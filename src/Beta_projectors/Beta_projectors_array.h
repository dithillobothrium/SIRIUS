/*
 * Beta_projectors_array.h
 *
 *  Created on: Feb 22, 2017
 *      Author: isivkov
 */

#ifndef __BETA_PROJECTORS_ARRAY_H__
#define __BETA_PROJECTORS_ARRAY_H__

#include "beta_projectors.h"

namespace sirius
{

template<int N>
class Beta_projectors_array
{
    protected:

        /// local array of gradient components. dimensions: 0 - gk, 1-orbitals
        std::array<matrix<double_complex>, N> components_gk_a_;

        /// the same but for one chunk
        std::array<matrix<double_complex>, N> chunk_comp_gk_a_;

        /// the same but for one chunk on gpu
        std::array<matrix<double_complex>, N> chunk_comp_gk_a_gpu_;

        /// inner product store
        std::array<mdarray<double, 1>, N> beta_phi_;

        Beta_projectors *bp_;

    public:

        static const int num_ = N;

        Beta_projectors_array(Beta_projectors* bp)
        : bp_(bp)
        {
            // on GPU we create arrays without allocation, it will before use
            for(int comp=0; comp<N; comp++){
                components_gk_a_[comp] = matrix<double_complex>( bp_->beta_gk_a().size(0), bp_->beta_gk_a().size(1) );
                #ifdef __GPU
                chunk_comp_gk_a_gpu_[comp] = matrix<double_complex>( bp_->num_gkvec_loc() , bp_->max_num_beta() , memory_t::none);
                #endif
            }

        }


        void generate(int chunk__, int calc_component__)
        {
            if (bp_->proc_unit() == CPU){
                chunk_comp_gk_a_[calc_component__] = mdarray<double_complex, 2>(&components_gk_a_[calc_component__](0, bp_->beta_chunk(chunk__).offset_),
                                                                                bp_->num_gkvec_loc(),
                                                                                bp_->beta_chunk(chunk__).num_beta_);
            }

            #ifdef __GPU
            if (bp_->proc_unit() == GPU){
                chunk_comp_gk_a_[calc_component__] = mdarray<double_complex, 2>(&components_gk_a_[calc_component__](0, bp_->beta_chunk(chunk__).offset_),
                                                                                chunk_comp_gk_a_gpu_[calc_component__].at<GPU>(),
                                                                                bp_->num_gkvec_loc(),
                                                                                bp_->beta_chunk(chunk__).num_beta_);

                chunk_comp_gk_a_[calc_component__].copy_to_device();
            }
            #endif
        }

        void generate(int chunk__){
            for(int comp=0; comp<N; comp++) {
                generate(chunk__, comp);
            }
        }

        /// Calculates inner product <beta_grad | Psi>.
        template <typename T>
        void inner(int chunk__, wave_functions& phi__, int idx0__, int n__, int calc_component__)
        {
            bp_->inner<T>(chunk__, phi__, idx0__, n__, chunk_comp_gk_a_[calc_component__], beta_phi_[calc_component__]);
        }

        //void inner(int chunk__,  wave_functions& phi__, int idx0__, int n__, mdarray<double_complex, 2> &beta_gk, mdarray<double, 1> &beta_phi);
        template <typename T>
        void inner(int chunk__, wave_functions& phi__, int idx0__, int n__)
        {
            for(int comp=0; comp<N; comp++) inner<T>(chunk__, phi__, idx0__, n__, comp);
        }

        template <typename T>
        matrix<T> beta_phi(int chunk__, int n__, int calc_component__)
        {
            int nbeta = bp_->beta_chunk(chunk__).num_beta_;

            if (bp_->proc_unit() == GPU) {
                return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_[calc_component__].template at<CPU>()),
                                           reinterpret_cast<T*>(beta_phi_[calc_component__].template at<GPU>()),
                                           nbeta, n__));
            } else {
                return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_[calc_component__].template at<CPU>()),
                                           nbeta, n__));
            }
        }

        template <typename T>
        std::array<matrix<T>,N> beta_phi(int chunk__, int n__)
        {
            std::array<matrix<T>,N> chunk_beta_phi;

            for(int comp=0; comp<N; comp++) chunk_beta_phi[comp] = beta_phi<T>(chunk__, n__, comp);

            return std::move(chunk_beta_phi);
        }

        void prepare()
        {
            #ifdef __GPU
            if (bp_->proc_unit() == GPU){
                for(int comp=0; comp<N; comp++){
                    chunk_comp_gk_a_gpu_[comp].allocate(memory_t::device);
                    beta_phi_[comp].allocate(memory_t::device);
                }
            }
            #endif
        }

        void dismiss()
        {
            #ifdef __GPU
            if (bp_->proc_unit() == GPU){
                for(int comp=0; comp<N; comp++){
                    chunk_comp_gk_a_gpu_[comp].deallocate_on_device();
                    beta_phi_[comp].deallocate_on_device();
                }
            }
            #endif
        }
};


}


#endif /* SRC_BETA_PROJECTORS_BETA_PROJECTORS_ARRAY_H_ */
