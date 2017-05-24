/*
 * force_ps.h
 *
 *  Created on: Sep 20, 2016
 *      Author: isivkov
 */

#ifndef __FORCES_H__
#define __FORCES_H__

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/beta_projectors_gradient.h"
#include "../potential.h"
#include "../density.h"

#include "Non_local_functor.h"
#include "omp_reducer.h"

namespace sirius {

using namespace geometry3d;

class Forces_PS
{
  private:
    Simulation_context& ctx_;

    Density& density_;

    Potential& potential_;

    K_point_set& kset_;

    mdarray<double, 2> local_forces_;
    mdarray<double, 2> ultrasoft_forces_;
    mdarray<double, 2> nonlocal_forces_;
    mdarray<double, 2> nlcc_forces_;
    mdarray<double, 2> ewald_forces_;
    mdarray<double, 2> total_forces_;
    mdarray<double, 2> us_nl_forces_;

    template <typename T>
    void add_k_point_contribution_to_nonlocal2(K_point& kpoint, mdarray<double, 2>& forces)
    {
        Beta_projectors_gradient bp_grad(ctx_, kpoint.gkvec(), kpoint.beta_projectors());

        Non_local_functor<T, Beta_projectors_gradient::num_> nlf(ctx_, bp_grad);

        nlf.add_k_point_contribution(kpoint, forces);
    }

    inline void allocate()
    {
        local_forces_     = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
        ultrasoft_forces_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
        nonlocal_forces_  = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
        nlcc_forces_      = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
        ewald_forces_     = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
        total_forces_     = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
        us_nl_forces_     = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    }

    inline void symmetrize_forces(mdarray<double, 2>& unsym_forces, mdarray<double, 2>& sym_forces)
    {
        matrix3d<double> const& lattice_vectors         = ctx_.unit_cell().symmetry().lattice_vectors();
        matrix3d<double> const& inverse_lattice_vectors = ctx_.unit_cell().symmetry().inverse_lattice_vectors();

        sym_forces.zero();

        #pragma omp parallel for
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            vector3d<double> cart_force(&unsym_forces(0, ia));
            vector3d<double> lat_force =
                inverse_lattice_vectors * (cart_force / (double)ctx_.unit_cell().symmetry().num_mag_sym());

            for (int isym = 0; isym < ctx_.unit_cell().symmetry().num_mag_sym(); isym++) {
                int ja = ctx_.unit_cell().symmetry().sym_table(ia, isym);

                auto& R                    = ctx_.unit_cell().symmetry().magnetic_group_symmetry(isym).spg_op.R;
                vector3d<double> rot_force = lattice_vectors * (R * lat_force);

                #pragma omp atomic update
                sym_forces(0, ja) += rot_force[0];

                #pragma omp atomic update
                sym_forces(1, ja) += rot_force[1];

                #pragma omp atomic update
                sym_forces(2, ja) += rot_force[2];
            }
        }
    }

  public:
    Forces_PS(Simulation_context& ctx__, Density& density__, Potential& potential__, K_point_set& kset__)
        : ctx_(ctx__)
        , density_(density__)
        , potential_(potential__)
        , kset_(kset__)
    {
        allocate();
        calc_forces_contributions();
        sum_forces();
    }

    inline void calc_local_forces(mdarray<double, 2>& forces)
    {
        PROFILE("sirius::Forces_PS::calc_local_forces");

        const Periodic_function<double>* valence_rho = density_.rho();

        Radial_integrals_vloc ri(ctx_.unit_cell(), ctx_.pw_cutoff(), 100);

        Unit_cell& unit_cell = ctx_.unit_cell();

        Gvec const& gvecs = ctx_.gvec();

        int gvec_count  = gvecs.gvec_count(ctx_.comm().rank());
        int gvec_offset = gvecs.gvec_offset(ctx_.comm().rank());

        if (forces.size(0) != 3 || (int)forces.size(1) != unit_cell.num_atoms()) {
            TERMINATE("forces array has wrong number of elements");
        }

        forces.zero();

        double fact = valence_rho->gvec().reduced() ? 2.0 : 1.0;

        /* here the calculations are in lattice vectors space */
        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            Atom& atom = unit_cell.atom(ia);

            int iat = atom.type_id();

            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig  = gvec_offset + igloc;
                int igs = gvecs.shell(ig);

                /* cartesian form for getting cartesian force components */
                vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

                /* scalar part of a force without multipying by G-vector */
                double_complex z = fact * fourpi * ri.value(iat, gvecs.gvec_len(ig)) *
                                   std::conj(valence_rho->f_pw_local(igloc)) *
                                   std::conj(ctx_.gvec_phase_factor(ig, ia));

                /* get force components multiplying by cartesian G-vector  */
                forces(0, ia) -= (gvec_cart[0] * z).imag();
                forces(1, ia) -= (gvec_cart[1] * z).imag();
                forces(2, ia) -= (gvec_cart[2] * z).imag();
            }
        }

        ctx_.comm().allreduce(&forces(0, 0), static_cast<int>(forces.size()));
    }

    inline void calc_ultrasoft_forces(mdarray<double, 2>& forces)
    {
        PROFILE("sirius::Forces_PS::calc_ultrasoft_forces");

        const mdarray<double_complex, 4>& density_matrix = density_.density_matrix();
        const Periodic_function<double>* veff_full = potential_.effective_potential();

        Unit_cell& unit_cell = ctx_.unit_cell();

        Gvec const& gvecs = ctx_.gvec();

        int gvec_count  = gvecs.gvec_count(ctx_.comm().rank());
        int gvec_offset = gvecs.gvec_offset(ctx_.comm().rank());

        forces.zero();

        double reduce_g_fact = veff_full->gvec().reduced() ? 2.0 : 1.0;

        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            Atom& atom = unit_cell.atom(ia);

            int iat = atom.type_id();
            if (!unit_cell.atom_type(iat).pp_desc().augment) {
                continue;
            }

            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig = gvec_offset + igloc;

                /* cartesian form for getting cartesian force components */
                vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

                /* scalar part of a force without multipying by G-vector and Qij
                   omega * V_conj(G) * exp(-i G Rn) */
                double_complex g_atom_part = reduce_g_fact * ctx_.unit_cell().omega() *
                                             std::conj(veff_full->f_pw_local(igloc)) *
                                             std::conj(ctx_.gvec_phase_factor(ig, ia));

                const Augmentation_operator& aug_op = ctx_.augmentation_op(iat);

                /* iterate over trangle matrix Qij */
                for (int ib2 = 0; ib2 < atom.type().indexb().size(); ib2++) {
                    for (int ib1 = 0; ib1 <= ib2; ib1++) {
                        int iqij = (ib2 * (ib2 + 1)) / 2 + ib1;

                        double diag_fact = ib1 == ib2 ? 1.0 : 2.0;

                        /* [omega * V_conj(G) * exp(-i G Rn) ] * rho_ij * Qij(G) */
                        double_complex z =
                            diag_fact * g_atom_part * density_matrix(ib1, ib2, 0, ia).real() *
                            double_complex(aug_op.q_pw(iqij, 2 * igloc), aug_op.q_pw(iqij, 2 * igloc + 1));

                        /* get force components multiplying by cartesian G-vector */
                        forces(0, ia) -= (gvec_cart[0] * z).imag();
                        forces(1, ia) -= (gvec_cart[1] * z).imag();
                        forces(2, ia) -= (gvec_cart[2] * z).imag();
                    }
                }
            }
        }

        ctx_.comm().allreduce(&forces(0, 0), static_cast<int>(forces.size()));
    }

    inline void calc_nonlocal_forces(mdarray<double, 2>& forces)
    {
        PROFILE("sirius::Forces_PS::calc_nonlocal_forces");

        mdarray<double, 2> unsym_forces(forces.size(0), forces.size(1));

        unsym_forces.zero();
        forces.zero();

        auto& spl_num_kp = kset_.spl_num_kpoints();

        for (int ikploc = 0; ikploc < spl_num_kp.local_size(); ikploc++) {
            K_point* kp = kset_.k_point(spl_num_kp[ikploc]);

            if (ctx_.gamma_point()) {
                add_k_point_contribution_to_nonlocal2<double>(*kp, unsym_forces);
            } else {
                add_k_point_contribution_to_nonlocal2<double_complex>(*kp, unsym_forces);
            }
        }

        ctx_.comm().allreduce(&unsym_forces(0, 0), static_cast<int>(unsym_forces.size()));

        symmetrize_forces(unsym_forces, forces);
    }

    inline void calc_nlcc_forces(mdarray<double, 2>& forces)
    {
        PROFILE("sirius::Forces_PS::calc_nlcc_force");

        /* get main arrays */
        auto xc_pot = potential_.xc_potential();

        /* transform from real space to reciprocal */
        xc_pot->fft_transform(-1);

        Unit_cell& unit_cell = ctx_.unit_cell();

        Gvec const& gvecs = ctx_.gvec();

        int gvec_count  = gvecs.count();
        int gvec_offset = gvecs.offset();

        forces.zero();

        double fact = gvecs.reduced() ? 2.0 : 1.0;

        auto ri = Radial_integrals_rho_core_pseudo<false>(ctx_.unit_cell(), ctx_.pw_cutoff(), 20);

        /* here the calculations are in lattice vectors space */
        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            Atom& atom = unit_cell.atom(ia);

            int iat = atom.type_id();

            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig  = gvec_offset + igloc;
                int igs = gvecs.shell(ig);

                if (ig == 0) {
                    continue;
                }

                /* cartesian form for getting cartesian force components */
                vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

                /* scalar part of a force without multipying by G-vector */
                double_complex z = fact * fourpi * ri.value(iat, gvecs.gvec_len(ig)) *
                                   std::conj(xc_pot->f_pw_local(igloc)) *
                                   std::conj(ctx_.gvec_phase_factor(
                                       ig, ia)); // std::exp(double_complex(0.0, -twopi * (gvec * atom.position())));

                /* get force components multiplying by cartesian G-vector */
                forces(0, ia) -= (gvec_cart[0] * z).imag();
                forces(1, ia) -= (gvec_cart[1] * z).imag();
                forces(2, ia) -= (gvec_cart[2] * z).imag();
            }
        }
        ctx_.comm().allreduce(&forces(0, 0), static_cast<int>(forces.size()));
    }

    inline void calc_ewald_forces(mdarray<double, 2>& forces)
    {
        PROFILE("sirius::Forces_PS::calc_ewald_forces");

        Unit_cell& unit_cell = ctx_.unit_cell();

        forces.zero();

        #pragma omp declare reduction(+ : mdarray <double, 2> : add_mdarray2d(omp_in, omp_out)) initializer(init_mdarray2d(omp_priv, omp_orig))

        /* alpha = 1 / ( 2 sigma^2 ) , selecting alpha here for better convergence*/
        double alpha = 1.0;
        double gmax = ctx_.pw_cutoff();
        double upper_bound = 0.0;
        double charge = 0.0;

        /* total charge */
        for(int ia = 0; ia < unit_cell.num_atoms(); ia++){
            charge += unit_cell.atom(ia).zn();
        }

        /* iterate to find alpha */
        do{
            alpha += 0.1;
            upper_bound = charge*charge * std::sqrt( 2.0 * alpha / twopi) * gsl_sf_erfc( gmax * std::sqrt(1.0 / (4.0 * alpha)) );
            //std::cout<<"alpha " <<alpha<<" ub "<<upper_bound<<std::endl;
        }while(upper_bound < 1.0e-8);

        if(alpha < 1.5){
            std::cout<<"Ewald forces error: probably, pw_cutoff is too small."<<std::endl;
        }

        double prefac = (ctx_.gvec().reduced() ? 4.0 : 2.0) * (twopi / unit_cell.omega());

        #pragma omp parallel for reduction(+ : forces)
        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;

            if (ig == 0) {
                continue;
            }

            double g2 = std::pow(ctx_.gvec().shell_len(ctx_.gvec().shell(ig)), 2);

            /* cartesian form for getting cartesian force components */
            vector3d<double> gvec_cart = ctx_.gvec().gvec_cart(ig);
            double_complex rho(0, 0);

            for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
                rho += ctx_.gvec_phase_factor(ig, ja) * static_cast<double>(unit_cell.atom(ja).zn());
            }

            rho = std::conj(rho);

            for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
                double scalar_part = prefac * (rho * ctx_.gvec_phase_factor(ig, ja)).imag() *
                                     static_cast<double>(unit_cell.atom(ja).zn()) * std::exp(-g2 / (4.0 * alpha)) / g2;

                for (int x : {0, 1, 2}) {
                    forces(x, ja) += scalar_part * gvec_cart[x];
                }
            }
        }

        ctx_.comm().allreduce(&forces(0, 0), static_cast<int>(forces.size()));

        double invpi = 1. / pi;

        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            for (int i = 1; i < unit_cell.num_nearest_neighbours(ia); i++) {
                int ja = unit_cell.nearest_neighbour(i, ia).atom_id;

                double d  = unit_cell.nearest_neighbour(i, ia).distance;
                double d2 = d * d;

                vector3d<double> t = unit_cell.lattice_vectors() * unit_cell.nearest_neighbour(i, ia).translation;

                double scalar_part =
                    static_cast<double>(unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn()) / d2 *
                    (gsl_sf_erfc(std::sqrt(alpha) * d) / d + 2.0 * std::sqrt(alpha * invpi) * std::exp(-d2 * alpha));

                for (int x : {0, 1, 2}) {
                    forces(x, ia) += scalar_part * t[x];
                }
            }
        }
    }

    inline void calc_forces_contributions()
    {
        calc_local_forces(local_forces_);
        calc_ultrasoft_forces(ultrasoft_forces_);
        calc_nonlocal_forces(nonlocal_forces_);
        calc_nlcc_forces(nlcc_forces_);
        calc_ewald_forces(ewald_forces_);
    }

    inline mdarray<double, 2> const& local_forces()
    {
        return local_forces_;
    }

    inline mdarray<double, 2> const& ultrasoft_forces()
    {
        return ultrasoft_forces_;
    }

    inline mdarray<double, 2> const& nonlocal_forces()
    {
        return nonlocal_forces_;
    }

    inline mdarray<double, 2> const& nlcc_forces()
    {
        return nlcc_forces_;
    }

    inline mdarray<double, 2> const& ewald_forces()
    {
        return ewald_forces_;
    }

    inline mdarray<double, 2> const& total_forces()
    {
        return total_forces_;
    }

    inline mdarray<double, 2> const& us_nl_forces()
    {
        return us_nl_forces_;
    }

    inline void sum_forces()
    {
        mdarray<double, 2> total_forces_unsym(3, ctx_.unit_cell().num_atoms());

        #pragma omp parallel for
        for (size_t i = 0; i < local_forces_.size(); i++) {
            us_nl_forces_[i] = ultrasoft_forces_[i] + nonlocal_forces_[i];
            total_forces_unsym[i] =
                local_forces_[i] + ultrasoft_forces_[i] + nonlocal_forces_[i] + nlcc_forces_[i] + ewald_forces_[i];
        }

        symmetrize_forces(total_forces_unsym, total_forces_);
    }

    inline void print_info()
    {
        PROFILE("sirius::DFT_ground_state::forces");

        if (ctx_.comm().rank() == 0) {
            auto print_forces = [&](mdarray<double, 2> const& forces) {
                for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                    printf("Atom %4i    force = %15.7f  %15.7f  %15.7f \n", ctx_.unit_cell().atom(ia).type_id(),
                           forces(0, ia), forces(1, ia), forces(2, ia));
                }
            };

            std::cout << "===== Total Forces in Ha/bohr =====" << std::endl;
            print_forces(total_forces());

            std::cout << "===== Forces: ultrasoft contribution from Qij =====" << std::endl;
            print_forces(ultrasoft_forces());

            std::cout << "===== Forces: non-local contribution from Beta-projectors =====" << std::endl;
            print_forces(nonlocal_forces());

            std::cout << "===== Forces: local contribution from local potential=====" << std::endl;
            print_forces(local_forces());

            std::cout << "===== Forces: nlcc contribution from core density=====" << std::endl;
            print_forces(nlcc_forces());

            std::cout << "===== Forces: Ewald forces from ions =====" << std::endl;
            print_forces(ewald_forces());
        }
    }
};
}

#endif // __FORCES_H__
