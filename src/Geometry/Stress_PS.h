/*
 * Stress_PS.h
 *
 *  Created on: Feb 21, 2017
 *      Author: isivkov
 */

#ifndef __STRESS_PS_H__
#define __STRESS_PS_H__

#include <fstream>

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/beta_projectors_gradient.h"
#include "../potential.h"
#include "../density.h"

//#include "../Test/Stress_radial_integrals.h"
//#include "../Test/Vloc_radial_integrals.h"

#include "../Radial_integrals/radial_integrals.h"
#include "../SDDK/geometry3d.hpp"
#include "../Beta_projectors/Beta_projectors_strain_deriv_gaunt.h"

#include "Non_local_functor.h"
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


        inline void symmetrize(matrix3d<double>& mtrx__)
        {
            if (!ctx_->use_symmetry()) {
                return;
            }

            matrix3d<double> result;

            for (int i = 0; i < ctx_->unit_cell().symmetry().num_mag_sym(); i++) {
                auto R = ctx_->unit_cell().symmetry().magnetic_group_symmetry(i).spg_op.rotation;
                result = result + transpose(R) * mtrx__ * R;
            }

            mtrx__ = result * (1.0 / ctx_->unit_cell().symmetry().num_mag_sym());
        }

//        void calc_local_stress()
//        {
//            // density for convolution with potentials
//            const Periodic_function<double>* valence_rho = density_->rho();
//
//            //------------------------------------------
//            // calc dV/dG contributions to stress tensor
//            //------------------------------------------
//
//            // generate gradient radial vloc integrals
//            Vloc_radial_integrals vlri(ctx_, &ctx_->unit_cell());
//            vlri.generate_g_gradient_radial_integrals();
//
//            Radial_integrals_vloc_dg ri_vloc_dg
//
//            std::vector<double_complex> dvloc_dg_local = ctx_->make_periodic_function<index_domain_t::local>();
//
//            // prepare for reduction
//            sigma_loc_.zero();
//            geometry3d::matrix3d<double>& sigma_loc = sigma_loc_;
//
//            // reduction of matrix3d type
//            #pragma omp declare reduction( + : geometry3d::matrix3d<double> : omp_out += omp_in) \
//                                           initializer( omp_priv = geometry3d::matrix3d<double>() )
//
//            #pragma omp parallel for reduction( + : sigma_loc )
//            for(int igloc = 0; igloc < ctx_->gvec_count(); igloc++){
//                int ig = ctx_->gvec_offset() + igloc;
//
//                if( ig==0 ){
//                    continue;
//                }
//
//                auto gvec_cart = ctx_->gvec().gvec_cart(ig);
//                auto gvec_norm = gvec_cart.length();
//
//                double scalar_part = ( dvloc_dg_local(igloc) *  std::conj(valence_rho->f_pw_local(ig))  ).real() / gvec_norm;
//
//                for(int i=0; i<3; i++){
//                    for(int j=0; j<=i; j++){
//                        sigma_loc(i,j) += gvec_cart[i] * gvec_cart[j] * scalar_part;
//                    }
//                }
//            }
//
//
//            // if G-vectors are reduced
//            double fact = ctx_->gvec().reduced() ? 2.0 : 1.0 ;
//
//            // multiply by scalars
//            for(int i=0; i<3; i++){
//                for(int j=0; j<=i; j++){
//                    sigma_loc_(i,j) *= fact;// * fourpi / ctx_->unit_cell().omega();
//                    sigma_loc_(j,i) = sigma_loc_(i,j);
//                }
//            }
//
//            // reduce over MPI (was distributed by G-vector)
//            ctx_->comm().allreduce(&sigma_loc_(0,0), 9 );
//
//            //------------------------------------------
//            // calc Evloc contributions to stress tensor
//            //------------------------------------------
//
//            // TODO get it from the finished DFT loop
//            double vloc_energy = valence_rho->inner(&potential_->local_potential());
//
//            for(int i: {0,1,2}){
//                sigma_loc_(i,i) += vloc_energy / ctx_->unit_cell().omega();
//            }
//
//            std::cout<<"local stress:"<<std::endl;
//            for(int i=0; i<3; i++){
//                for(int j=0; j<3; j++){
//                    std::cout<< sigma_loc_(i,j)<<" ";
//                }
//                std::cout<<std::endl;
//            }
//        }



        void calc_non_local_stress()
        {
            mdarray<double, 2> collect_result(Beta_projectors_strain_deriv_gaunt::num_, ctx_->unit_cell().num_atoms() );
            //collect_result.zero();

            auto& spl_num_kp = kset_->spl_num_kpoints();

            // create reduction operation for matrix3d
            #pragma omp declare reduction (+: geometry3d::matrix3d<double>: omp_out+=omp_in )

            geometry3d::matrix3d<double> sigma_non_loc_priv;

            // iterate over k-points on the current MPI-rank
            for(int ikploc=0; ikploc < spl_num_kp.local_size() ; ikploc++){
                collect_result.zero();
                sigma_non_loc_priv.zero();

                std::cout<<"------------ non-local stress kp ------------:"<<std::endl;

                K_point* kp = kset_->k_point(spl_num_kp[ikploc]);
                Beta_projectors_strain_deriv_gaunt bplg(&kp->beta_projectors(), ctx_);
                Non_local_functor<double_complex, Beta_projectors_strain_deriv_gaunt::num_> nlf(ctx_,kset_,&bplg);

                nlf.add_k_point_contribution(*kp,collect_result);

                // iterate over atoms and collect result to tensor sigma_non_loc_priv
                //#pragma omp parallel for reduction(+:sigma_non_loc_priv)
                for(size_t ia=0; ia < collect_result.size(1); ia++){
                    for(size_t comp=0; comp < collect_result.size(0); comp++){
                        std::cout<< collect_result(comp, ia) * (1.0 / ctx_->unit_cell().omega())<<" ";
                    }
                    std::cout<<std::endl;

                    for(size_t i=0; i<3; i++){
                        for(size_t j=0; j<3; j++){
                            sigma_non_loc_priv(i,j) += collect_result(Beta_projectors_strain_deriv_gaunt::ind(i,j), ia) * (1.0 / ctx_->unit_cell().omega());
                        }
                    }
                }


                for(int i=0; i<3; i++){
                    for(int j=0; j<3; j++){
                        std::cout<< sigma_non_loc_priv(i,j)<<" ";
                    }
                    std::cout<<std::endl;
                }

                sigma_non_loc_ +=  sigma_non_loc_priv ;
            }






            // here one need to allreduce sigma_non_loc_priv or sigma_non_loc_
            // if we calculate only upper(lower)-triangular values of sigma_non_loc_priv
            // then need to make a loop for adding lower(upper)-triangulaer values to sigma_non_loc_


            symmetrize(sigma_non_loc_);

            std::cout<<"non-local stress:"<<std::endl;
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    std::cout<< sigma_non_loc_(i,j)<<" ";
                }
                std::cout<<std::endl;
            }
        }


};
/*
 sirius
 0.0248946 -0.000114811 -0.000114811
-0.000114951 0.0250291 8.14407e-05
-0.000114951 8.14407e-05 0.0250291

 qe
   3.3496639936778497E-002     8.1733454090695417E-004     8.1498363770990683E-004
   8.1733454090695417E-004     3.3231721321851974E-002    -1.0659863137523703E-003
   8.1498363770990683E-004    -1.0659863137523703E-003     3.3232328524179650E-002

 */

}


#endif /* SRC_GEOMETRY_STRESS_PS_H_ */
