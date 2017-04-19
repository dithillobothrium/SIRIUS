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
    Simulation_context& ctx_;
    K_point_set& kset_;
    Beta_projectors_base<N>& bp_base_;

public:

    Non_local_functor(Simulation_context& ctx__,
                      K_point_set& kset__,
                      Beta_projectors_base<N>& bp_base__)
    : ctx_(ctx__),
      kset_(kset__),
      bp_base_(bp_base__)
    {}

    /// static const can be public
    static const int num_ = N;

    /// collect summation result in an array
    void add_k_point_contribution(K_point& kpoint__, mdarray<double, 2>& collect_res__)
    {
        Unit_cell &unit_cell = ctx_.unit_cell();

        Beta_projectors& bp = kpoint.beta_projectors();

        auto& bp_chunks = bp.beta_projector_chunks();

        // from formula
        double main_two_factor = -2.0;

        #ifdef __GPU
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (ctx_.processing_unit() == GPU) {
                int nbnd = kpoint.num_occupied_bands(ispn);
                kpoint.spinor_wave_functions(ispn).copy_to_device(0, nbnd);
            }
        }
        #endif

        bp_base_.prepare();
        bp.prepare();

        for (int icnk = 0; icnk < bp_chunks.num_chunks(); icnk++) {
            /* generate chunk for inner product of beta */
            bp.generate(icnk);

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* total number of occupied bands for this spin */
                int nbnd = kpoint.num_occupied_bands(ispn);

                // inner product of beta and WF
                auto bp_phi_chunk = bp.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), 0, nbnd);

                for (int x = 0; x < N; x++) {
                    /* generate chunk for inner product of beta gradient */
                    bp_base_.generate(icnk, x);

                    // inner product of beta gradient and WF
                    auto bp_base_phi_chunk = bp_base_.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), 0, nbnd);

                    splindex<block> spl_nbnd(nbnd, kpoint.comm().size(), kpoint.comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();

                    int bnd_offset = spl_nbnd.global_offset();

                    #pragma omp parallel for
                    for(int ia_chunk = 0; ia_chunk < bp_chunks(icnk).num_atoms_; ia_chunk++) {
                        int ia   = bp_chunks(icnk).desc_(3, ia_chunk);
                        int offs = bp_chunks(icnk).desc_(1, ia_chunk);
                        int nbf  = bp_chunks(icnk).desc_(0, ia_chunk);
                        int iat  = unit_cell.atom(ia).type_id();

                        // mpi
                        // TODO make in smart way with matrix multiplication
                        for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++) {
                            int ibnd = spl_nbnd[ibnd_loc];

                            auto D_aug_mtrx = [&](int i, int j)
                                                                    {
                                if (unit_cell.atom(ia).type().pp_desc().augment) {
                                    return unit_cell.atom(ia).d_mtrx(i, j, ispn) - kpoint.band_energy(ibnd) *
                                            ctx_.augmentation_op(iat).q_mtrx(i, j);
                                } else {
                                    return unit_cell.atom(ia).d_mtrx(i, j, ispn);
                                }
                                                                    };

                            for (int ibf = 0; ibf < unit_cell.atom(ia).type().mt_lo_basis_size(); ibf++) {
                                for (int jbf = 0; jbf < unit_cell.atom(ia).type().mt_lo_basis_size(); jbf++) {
                                    /* calculate scalar part of the forces */
                                    double_complex scalar_part = main_two_factor *
                                            kpoint.band_occupancy(ibnd + ispn * ctx_.num_fv_states()) * kpoint.weight() *
                                            D_aug_mtrx(ibf, jbf) * std::conj(bp_phi_chunk(offs + jbf, ibnd));

                                    /* multiply scalar part by gradient components */
                                    collect_res__(x, ia) += (scalar_part * bp_base_phi_chunk(offs + ibf, ibnd)).real();
                                }
                            }
                        }
                    }
                }
            }
        }

        bp.dismiss();
        bp_base_.dismiss();
    }
};

}


#endif /* __NON_LOCAL_FUNCTOR_H__ */
