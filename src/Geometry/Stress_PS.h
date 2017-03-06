/*
 * Stress_PS.h
 *
 *  Created on: Feb 21, 2017
 *      Author: isivkov
 */

#ifndef __GEOMETRY_STRESS_PS_H__
#define __GEOMETRY_STRESS_PS_H__

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/beta_projectors_gradient.h"
#include "../Beta_projectors/Beta_projectors_lattice_gradient.h"
#include "../potential.h"
#include "../density.h"

#include "../Test/Stress_radial_integrals.h"
#include "../Test/Vloc_radial_integrals.h"

namespace sirius
{

class Stress_PS
{
    private:
        Simulation_context* ctx_;
        Density* density_;
        Potential* potential_;
        K_point_set* kset_;

    public:
        Stress_PS(Simulation_context* ctx__,
                  Density* density__,
                  Potential* potential__,
                  K_point_set* kset__)
        : ctx_(ctx__),
          density_(density__),
          potential_(potential__),
          kset_(kset__)
        {
            Stress_radial_integrals rad_int(ctx_, &ctx_->unit_cell());

            // tmp
            rad_int.generate_beta_radial_integrals(0);

            Beta_projectors_lattice_gradient bplg(&kset_->k_point(0)->beta_projectors(), ctx__);




        }

        template<class ProcessFuncT>
        void process_in_g_space(ProcessFuncT func__)
        {
            Unit_cell &unit_cell = ctx_->unit_cell();
            Gvec const& gvecs = ctx_->gvec();

            int gvec_count = gvecs.gvec_count(ctx_->comm().rank());
            int gvec_offset = gvecs.gvec_offset(ctx_->comm().rank());

            double fact = gvecs.reduced() ? 2.0 : 1.0 ;

            // here the calculations are in lattice vectors space
            #pragma omp parallel for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++){
                Atom &atom = unit_cell.atom(ia);
                int iat = atom.type_id();

                // mpi distributed
                for (int igloc = 0; igloc < gvec_count; igloc++){
                    int ig = gvec_offset + igloc;
                    int igs = gvecs.shell(ig);

                    func__(igs, igloc, ig, iat);
                }
            }
        }

        void calc_local_stress()
        {
            // calc vloc energy
            // TODO get it from the finished DFT loop
            const Periodic_function<double>* valence_rho = density_->rho();

            double vloc_energy = valence_rho->inner(&potential_->local_potential());

            // generate gradient radial vloc integrals
            Vloc_radial_integrals vlri(ctx_, &ctx_->unit_cell());
            vlri.generate_g_gradient_radial_integrals();

            // d vloc(G) / d G
            splindex<block> spl_ngv(gvecs.num_gvec(), ctx_->comm().size(), ctx_->comm().rank());
            mdarray<double_complex> dvloc_dg_local(spl_ngv.local_size());
            dvloc_dg_local.zero();

            ctx_->unit_cell().make_periodic_function_local(dvloc_dg_local, spl_ngv, g_gradient_vloc_radial_integrals__, ctx_->gvec());

            // create derivative over lattice matrix components
            for(int i=0; i<spl_ngv.local_size(); i++)
            {

            }

            //ctx_->comm().allreduce(&forces(0,0), static_cast<int>(forces.size()));
        }


        // TODO use MKL and replace it to linalg?
        double_complex inner_pw(double_complex* f_pw_local1, double_complex* f_pw_local2, size_t size)
        {
            double_complex res{0};

            #pragma omp parallel for reduction(+:res)
            for (size_t i = 0; i < size; i++) {
                res += f_pw_local1[i] * std::conj(f_pw_local2[i]);
            }

        }


};

}


#endif /* SRC_GEOMETRY_STRESS_PS_H_ */
