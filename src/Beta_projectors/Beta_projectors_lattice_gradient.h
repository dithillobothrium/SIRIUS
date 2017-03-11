/*
 * Beta_projectors_stress.h
 *
 *  Created on: Feb 22, 2017
 *      Author: isivkov
 */

#ifndef __BETA_PROJECTORS_STRESS_H__
#define __BETA_PROJECTORS_STRESS_H__

#include "../utils.h"
#include "Beta_projectors_array.h"
#include "../Test/Stress_radial_integrals.h"

namespace sirius
{

class Beta_projectors_lattice_gradient: public Beta_projectors_array<6>
{
    protected:
        const Simulation_context* ctx_;

        mdarray<double_complex, 3> beta_gk_t_;

        int ind(int i, int j)
        {
            return j*this->num_ + i;
        }

        // dimensions of a 3d matrix (f.e. stress tensor)
        const int nu_ = 3;
        const int nv_ = 3;

    public:

        struct beta_besselJ_gaunt_coefs_t
        {
            int l1;
            int m1;
            int l2;
            int m2;
            std::array<double, 3> gaunt_coefs;
        };

        Beta_projectors_lattice_gradient(Beta_projectors* bp__, const Simulation_context* ctx__)
        : Beta_projectors_array<6>(bp__),
          ctx_(ctx__)
        {
            initialize();
        }


        virtual void initialize()
        {
            init_beta_gk_t();
            init_beta_gk();

            this->is_initialized_ = true;
        }

        void init_beta_gk_t()
        {
            int num_beta_t = bp_->num_beta_by_atom_types();

            //const std::vector<lpair>& lpairs = rad_int.radial_integrals_lpairs();
            /* allocate array */
            beta_gk_t_ = mdarray<double_complex, 3>(bp_->num_gkvec_loc(), num_beta_t, nu_);

            double const_prefac = fourpi * std::sqrt(fourpi / ( 3.0 * ctx_->unit_cell().omega() ));

            //auto m1m2idx = [](int l1, int m1, int l2, int m2){ return (m1+l1) * (2*l2+1) + m2+l2; };

            // compute
            for (int iat = 0; iat < ctx_->unit_cell().num_atom_types(); iat++){
                auto& atom_type = ctx_->unit_cell().atom_type(iat);
                Stress_radial_integrals rad_int(ctx_, &ctx_->unit_cell());
                rad_int.generate_beta_radial_integrals(iat);

                // + 1 because max BesselJ l is lbeta + 1 in summation
                int lmax_jl = atom_type.indexr().lmax() + 1;

                // get iterators of radial integrals
                auto rad_int_iterators = rad_int.radial_integral_iterators();

                // TODO: make common array of gaunt coefs for all atom types
                // array to store gaunt coeffs
                std::vector<std::vector<beta_besselJ_gaunt_coefs_t>> beta_besselJ_gaunt_coefs;

                // generate gaunt coeffs and store
                for(auto rad_int_it = rad_int_iterators.first; rad_int_it != rad_int_iterators.second; rad_int_it++){
                    lpair lp = rad_int_it->first;
                    int idxrf = lp.first;
                    int l1 = atom_type.indexr(idxrf).l;
                    int l2 = lp.second;
                    std::vector<beta_besselJ_gaunt_coefs_t> bj_gc;

                    for (int m1 = -l1; m1 <= l1; m1++){
                        for (int m2 = -l2; m2 <= l2; m2++){
                            bj_gc.push_back({
                                l1,m1,l2,m2,
                                {SHT::gaunt_rlm(l1, 1, l2, m1,  1, m2), // for x component
                                        SHT::gaunt_rlm(l1, 1, l2, m1, -1, m2),  // for y component
                                        SHT::gaunt_rlm(l1, 1, l2, m1,  0, m2)}  // for z component
                            });
                        }
                    }

                    beta_besselJ_gaunt_coefs.push_back(bj_gc);
                }

                // iterate over gk vectors
                #pragma omp parallel for private(rad_int_iterators)
                for (int igkloc = 0; igkloc < bp_->num_gkvec_loc(); igkloc++)
                {
                    int igk   = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
                    double gk = bp_->gk_vectors().gvec_len(igk);

                    // vs = {r, theta, phi}
                    auto vs = SHT::spherical_coordinates(bp_->gk_vectors().gkvec_cart(igk));

                    // compute real spherical harmonics for G+k vector
                    std::vector<double> gkvec_rlm(Utils::lmmax(lmax_jl));
                    SHT::spherical_harmonics(lmax_jl, vs[1], vs[2], &gkvec_rlm[0]);

                    // iterator over
                    int beta_j_gc_it = 0;

                    // SHOULD WORK
                    for(auto rad_int_it = rad_int_iterators.first; rad_int_it != rad_int_iterators.second; rad_int_it++){
                        lpair lp = rad_int_it->first;
                        int idxrf = lp.first;
                        int l1 = atom_type.indexr(idxrf).l;
                        int l2 = lp.second;
                        int idx_bf_start = atom_type.indexb().index_by_idxrf(idxrf);

                        // get gaunt coefs between beta l1 and besselJ l2
                        auto& bj_gc = beta_besselJ_gaunt_coefs[beta_j_gc_it++];

                        //TODO optimize std::pow
                        // multiply radial integral and constants
                        double_complex prefact = const_prefac * std::pow(double_complex(0, -1), l2) * rad_int.beta_radial_integral(rad_int_it, gk);

                        // get start index for basis functions
                        int xi = atom_type.indexb().index_by_idxrf(idxrf);

                        // m1-m2 iterator for gaunt coefficients
                        int m1m2_it = 0;

                        // iterate over m-components of l1 and l2
                        for (int m1 = -l1; m1 <= l1; m1++, xi++){
                            for (int m2 = -l2; m2 <= l2; m2++, m1m2_it++){
                                for(int comp=0; comp< nu_; comp++){
                                    beta_gk_t_(igkloc, atom_type.offset_lo() + xi, comp) += prefact *
                                            gkvec_rlm[ Utils::lm_by_l_m(l2, m2) ] * bj_gc[ m1m2_it ].gaunt_coefs[comp];
                                }
                            }
                        }
                    }
                }
            }
        }

        void init_beta_gk()
        {
            auto& unit_cell = ctx_->unit_cell();
            int num_gkvec_loc = bp_->num_gkvec_loc();

            for(int i = 0; i < this->num_; i++)
            {
                components_gk_a_[i] = matrix<double_complex>(num_gkvec_loc, unit_cell.mt_lo_basis_size());
            }

            auto cell_matrix = unit_cell.lattice_vectors();
            auto inv_cell_matrix = unit_cell.reciprocal_lattice_vectors();

            double omega = unit_cell.omega();

            double_complex const_fact = double_complex(0.0,1.0) / std::sqrt( std::pow( omega, 3));

            #pragma omp parallel for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {

                // prepare G+k phases
                auto vk = bp_->gk_vectors().vk();
                double phase = twopi * (vk * unit_cell.atom(ia).position());
                double_complex phase_k = std::exp(double_complex(0.0, phase));

                std::vector<double_complex> phase_gk(num_gkvec_loc);

                for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
                    int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igk_loc;
                    auto G = bp_->gk_vectors().gvec(igk);
                    phase_gk[igk_loc] = std::conj(ctx_->gvec_phase_factor(G, ia) * phase_k);
                }

                // cartesian atomic coordinate
                auto Rcart = cell_matrix * unit_cell.atom(ia).position();
                auto vk_cart = inv_cell_matrix * vk;

                // TODO: need to optimize order of loops
                // calc beta lattice gradient
                for (int igkloc = 0; igkloc < bp_->num_gkvec_loc(); igkloc++) {
                    int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
                    auto Gcart = bp_->gk_vectors().gvec_cart(igk);

                    for (int xi = 0; xi < unit_cell.atom(ia).mt_lo_basis_size(); xi++) {

                        // iteration over tensor components
                        for(int u = 0; u < nu_; u++){
                            for(int v = 0; v <= u; v++){
                                // complicate formula
                                components_gk_a_[ind(u,v)](igkloc, unit_cell.atom(ia).offset_lo() + xi) +=
                                        beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, v) *
                                        phase_gk[igkloc] * Gcart[u] * const_fact -
                                        bp_->beta_gk_a()(igkloc, unit_cell.atom(ia).offset_lo() + xi) *
                                        double_complex(0.0 , 1.0) * vk_cart[u] * Rcart[v] / omega;
                            }

                            components_gk_a_[ind(u,u)](igkloc, unit_cell.atom(ia).offset_lo() + xi) +=
                                    bp_->beta_gk_a()(igkloc, unit_cell.atom(ia).offset_lo() + xi) / omega * 0.5;
                        }


                    }
                }
            }

        }


};

}


#endif /* __BETA_PROJECTORS_STRESS_H__ */
