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

#include "../SDDK/geometry3d.hpp"

#include "Non_local_functor.h"

#include <fstream>

namespace sirius
{

//-----------------------------------------
// for omp reduction
//------------------------------------------
//template<typename T>
//inline void init_matrix3d(geometry3d::matrix3d<T>& priv, geometry3d::matrix3d<T>& orig )
//{
//    priv = geometry3d::matrix3d<T>();
//}
//
//template<typename T>
//inline void add_matrix3d(geometry3d::matrix3d<T>& in, geometry3d::matrix3d<T>& out)
//{
//    out += in;
//}

class Stress_PS
{
    private:
        Simulation_context* ctx_;
        Density* density_;
        Potential* potential_;
        K_point_set* kset_;

        geometry3d::matrix3d<double> sigma_loc_;
        geometry3d::matrix3d<double> sigma_non_loc_;

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

            //calc_local_stress();


        }

//        template<class ProcessFuncT>
//        void process_in_g_space(ProcessFuncT func__)
//        {
//            Unit_cell &unit_cell = ctx_->unit_cell();
//            Gvec const& gvecs = ctx_->gvec();
//
//            int gvec_count = gvecs.gvec_count(ctx_->comm().rank());
//            int gvec_offset = gvecs.gvec_offset(ctx_->comm().rank());
//
//            double fact = gvecs.reduced() ? 2.0 : 1.0 ;
//
//            // here the calculations are in lattice vectors space
//            #pragma omp parallel for
//            for (int ia = 0; ia < unit_cell.num_atoms(); ia++){
//                Atom &atom = unit_cell.atom(ia);
//                int iat = atom.type_id();
//
//                // mpi distributed
//                for (int igloc = 0; igloc < gvec_count; igloc++){
//                    int ig = gvec_offset + igloc;
//                    int igs = gvecs.shell(ig);
//
//                    func__(igs, igloc, ig, iat);
//                }
//            }
//        }



        void calc_local_stress()
        {
            // density for convolution with potentials
            const Periodic_function<double>* valence_rho = density_->rho();

            //------------------------------------------
            // calc dV/dG contributions to stress tensor
            //------------------------------------------

            // generate gradient radial vloc integrals
            Vloc_radial_integrals vlri(ctx_, &ctx_->unit_cell());
            vlri.generate_g_gradient_radial_integrals();

            // d vloc(G) / d G
            splindex<block> spl_ngv(ctx_->gvec().num_gvec(), ctx_->comm().size(), ctx_->comm().rank());

            mdarray<double_complex, 1> dvloc_dg_local(spl_ngv.local_size());
            dvloc_dg_local.zero();
            ctx_->unit_cell().make_periodic_function_local(dvloc_dg_local, spl_ngv, vlri.g_gradient_vloc_radial_integrals(), ctx_->gvec());

            // prepare for reduction
            sigma_loc_.zero();
            geometry3d::matrix3d<double>& sigma_loc = sigma_loc_;

            // reduction of matrix3d type
            #pragma omp declare reduction( + : geometry3d::matrix3d<double> : omp_out += omp_in) \
                                           initializer( omp_priv = geometry3d::matrix3d<double>() )

            #pragma omp parallel for reduction( + : sigma_loc )
            for(int igloc = 0; igloc < spl_ngv.local_size(); igloc++){
                int ig = spl_ngv.global_offset() + igloc;

                if( ig==0 ){
                    continue;
                }

                auto gvec_cart = ctx_->gvec().gvec_cart(ig);
                auto gvec_norm = gvec_cart.length();

                double scalar_part = ( dvloc_dg_local(igloc) *  std::conj(valence_rho->f_pw_local(ig))  ).real() / gvec_norm;

                for(int i=0; i<3; i++){
                    for(int j=0; j<=i; j++){
                        sigma_loc(i,j) += gvec_cart[i] * gvec_cart[j] * scalar_part;
                    }
                }
            }


            // if G-vectors are reduced
            double fact = ctx_->gvec().reduced() ? 2.0 : 1.0 ;

            // multiply by scalars
            for(int i=0; i<3; i++){
                for(int j=0; j<=i; j++){
                    sigma_loc_(i,j) *= fact;// * fourpi / ctx_->unit_cell().omega();
                    sigma_loc_(j,i) = sigma_loc_(i,j);
                }
            }

            // reduce over MPI (was distributed by G-vector)
            ctx_->comm().allreduce(&sigma_loc_(0,0), 9 );

            //------------------------------------------
            // calc Evloc contributions to stress tensor
            //------------------------------------------

            // TODO get it from the finished DFT loop
            double vloc_energy = valence_rho->inner(&potential_->local_potential());

            for(int i: {0,1,2}){
                sigma_loc_(i,i) += vloc_energy / ctx_->unit_cell().omega();
            }

            std::cout<<"local stress:"<<std::endl;
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    std::cout<< sigma_loc_(i,j)<<" ";
                }
                std::cout<<std::endl;
            }
        }



        void calc_non_local_stress()
        {


            auto& spl_num_kp = kset_->spl_num_kpoints();

            std::cout<<"nl stress start"<<std::endl;
            for(int ikploc=0; ikploc < spl_num_kp.local_size() ; ikploc++){
                K_point* kp = kset_->k_point(spl_num_kp[ikploc]);
                Beta_projectors_lattice_gradient bplg(&kp->beta_projectors(), ctx_);
                Non_local_functor<double_complex, Beta_projectors_lattice_gradient::num_> nlf(ctx_,kset_,&bplg);

                std::vector<double> stress_nl(bplg.num_, 0.0);

                nlf.add_k_point_contribution(*kp, [&](int comp__, int ia__, double_complex val__)
                                             {
                                                 stress_nl[comp__] += val__.real();
                                             });

                for(int i=0; i<3; i++){
                    for(int j=0; j<3; j++){
                        sigma_non_loc_(i,j) += stress_nl[bplg.ind(i,j)];
                        //sigma_non_loc_(j,i) = sigma_non_loc_(i,j);
                    }
                }
            }




            std::cout<<"non-local stress:"<<std::endl;
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    std::cout<< sigma_non_loc_(i,j)<<" ";
                }
                std::cout<<std::endl;
            }
        }


};

}


#endif /* SRC_GEOMETRY_STRESS_PS_H_ */
