/*
 * Non_local_functor.h
 *
 *  Created on: Mar 13, 2017
 *      Author: isivkov
 */

#ifndef __NON_LOCAL_FUNCTOR_H__
#define __NON_LOCAL_FUNCTOR_H__

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/Beta_projectors_array.h"
#include "../potential.h"
#include "../density.h"
#include "Chunk_linalg.h"

namespace sirius
{

template<class T, int N>
class Non_local_functor
{
    private:
        Simulation_context* ctx_;
        K_point_set* kset_;
        Beta_projectors_array<N>* bpa_;

    public:

        Non_local_functor(Simulation_context* ctx__,
                          K_point_set* kset__,
                          Beta_projectors_array<N>* bpa__)
        : ctx_(ctx__),
          kset_(kset__),
          bpa_(bpa__)
        {
        }

        /// static const can be public
        static const int num_ = N;

        /// collect summation result in an array
        void add_k_point_contribution(K_point& kpoint__, mdarray<double, 2>& collect_res__)
        {
            Unit_cell &unit_cell = ctx_->unit_cell();

            Beta_projectors &bp = kpoint__.beta_projectors();

            // from formula
            double main_two_factor = -2.0;

            #ifdef __GPU
            for (int ispn = 0; ispn < ctx_->num_spins(); ispn++)
            {
                if( bp.proc_unit() == GPU )
                {
                    int nbnd = kpoint__.num_occupied_bands(ispn);
                    kpoint__.spinor_wave_functions(ispn).allocate_on_device();
                    kpoint__.spinor_wave_functions(ispn).copy_to_device(0, nbnd);
                }
            }
            #endif

            bpa_->prepare();
            bp.prepare();

            for (int icnk = 0; icnk < bp.num_beta_chunks(); icnk++)
            {
                // generate chunk for inner product of beta gradient
                bpa_->generate(icnk);

                // generate chunk for inner product of beta
                bp.generate(icnk);

                for (int ispn = 0; ispn < ctx_->num_spins(); ispn++)
                {
                    /* total number of occupied bands for this spin */
                    int nbnd = kpoint__.num_occupied_bands(ispn);

                    // inner product of beta gradient and WF
                    bpa_->template inner<T>(icnk, kpoint__.spinor_wave_functions(ispn), 0, nbnd);

                    // get inner product
                    std::array<matrix<T>, N> bp_grad_phi_chunk = bpa_->template beta_phi<T>(icnk, nbnd);

                    // inner product of beta and WF
                    bp.template inner<T>(icnk, kpoint__.spinor_wave_functions(ispn), 0, nbnd);

                    // get inner product
                    matrix<T> bp_phi_chunk = bp.template beta_phi<T>(icnk, nbnd);

                    splindex<block> spl_nbnd(nbnd, kpoint__.comm().size(), kpoint__.comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();

                    int bnd_offset = spl_nbnd.global_offset();

                    //#pragma omp parallel for
                    for(int ia_chunk = 0; ia_chunk < bp.beta_chunk(icnk).num_atoms_; ia_chunk++)
                    {
                        int ia = bp.beta_chunk(icnk).desc_(3, ia_chunk);
                        int offs = bp.beta_chunk(icnk).desc_(1, ia_chunk);
                        int nbf = bp.beta_chunk(icnk).desc_(0, ia_chunk);
                        int iat = unit_cell.atom(ia).type_id();

                        // mpi
                        // TODO make in smart way with matrix multiplication
                        for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++)
                        {
                            int ibnd = spl_nbnd[ibnd_loc];

                            auto D_aug_mtrx = [&](int i, int j)
                                {
                                    if (unit_cell.atom(ia).type().pp_desc().augment) {
                                        return unit_cell.atom(ia).d_mtrx(i, j, ispn) - kpoint__.band_energy(ibnd) *
                                                ctx_->augmentation_op(iat).q_mtrx(i, j);
                                    } else {
                                        return unit_cell.atom(ia).d_mtrx(i, j, ispn);
                                    }
                                };

                            for(int ibf = 0; ibf < unit_cell.atom(ia).type().mt_lo_basis_size(); ibf++ )
                            {
                                for(int jbf = 0; jbf < unit_cell.atom(ia).type().mt_lo_basis_size(); jbf++ )
                                {
                                    // calc scalar part of the forces
                                    double_complex scalar_part = main_two_factor *
                                            kpoint__.band_occupancy(ibnd + ispn * ctx_->num_fv_states()) * kpoint__.weight() *
                                            D_aug_mtrx(ibf, jbf) *
                                            std::conj(bp_phi_chunk(offs + jbf, ibnd));

                                    for(int comp=0; comp < N; comp++){
                                        collect_res__(comp, ia) += (scalar_part * bp_grad_phi_chunk[comp](offs + ibf, ibnd)).real();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            bp.dismiss();
            bpa_->dismiss();

            #ifdef __GPU
            for (int ispn = 0; ispn < ctx_->num_spins(); ispn++)
            {
                if( bp.proc_unit() == GPU )
                {
                    kpoint__.spinor_wave_functions(ispn).deallocate_on_device();
                }
            }
            #endif
        }
};

}


#endif /* __NON_LOCAL_FUNCTOR_H__ */
