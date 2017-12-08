// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file force.cpp
 *
 *  \brief Contains implementation of sirius::Force class.
 */

#include "force.h"

namespace sirius {

void Force::compute_dmat(Simulation_parameters const& parameters__,
                         K_point* kp__,
                         dmatrix<double_complex>& dm__)
{
    PROFILE("sirius::Force::compute_dmat");

    dm__.zero();

    STOP();

    ///* trivial case */
    //if (!parameters__.need_sv())
    //{
    //    for (int i = 0; i < parameters__.num_fv_states(); i++) dm__.set(i, i, double_complex(kp__->band_occupancy(i), 0));
    //}
    //else
    //{
    //    if (parameters__.num_mag_dims() != 3)
    //    {
    //        dmatrix<double_complex> ev1(parameters__.num_fv_states(), parameters__.num_fv_states(), kp__->blacs_grid(), parameters__.cyclic_block_size(), parameters__.cyclic_block_size());
    //        for (int ispn = 0; ispn < parameters__.num_spins(); ispn++)
    //        {
    //            auto& ev = kp__->sv_eigen_vectors(ispn);
    //            /* multiply second-variational eigen-vectors with band occupancies */
    //            for (int j = 0; j < ev.num_cols_local(); j++)
    //            {
    //                /* up- or dn- band index */
    //                int jb = ev.icol(j);
    //                for (int i = 0; i < ev.num_rows_local(); i++)
    //                    ev1(i, j) = conj(ev(i, j)) * kp__->band_occupancy(jb + ispn * parameters__.num_fv_states());
    //            }

    //            linalg<CPU>::gemm(0, 1, parameters__.num_fv_states(), parameters__.num_fv_states(), parameters__.num_fv_states(),
    //                              linalg_const<double_complex>::one(), ev1, ev, linalg_const<double_complex>::one(), dm__);
    //        }
    //    }
    //    else
    //    {
    //        dmatrix<double_complex> ev1(parameters__.num_bands(), parameters__.num_bands(), kp__->blacs_grid(), parameters__.cyclic_block_size(), parameters__.cyclic_block_size());
    //        auto& ev = kp__->sv_eigen_vectors(0);
    //        /* multiply second-variational eigen-vectors with band occupancies */
    //        for (int j = 0; j < ev.num_cols_local(); j++)
    //        {
    //            /* band index */
    //            int jb = ev.icol(j);
    //            for (int i = 0; i < ev.num_rows_local(); i++) ev1(i, j) = conj(ev(i, j)) * kp__->band_occupancy(jb);
    //        }
    //        for (int ispn = 0; ispn < parameters__.num_spins(); ispn++)
    //        {
    //            int offs = ispn * parameters__.num_fv_states();

    //            linalg<CPU>::gemm(0, 1, parameters__.num_fv_states(), parameters__.num_fv_states(), parameters__.num_bands(),
    //                              linalg_const<double_complex>::one(), ev1, offs, 0, ev, offs, 0, linalg_const<double_complex>::one(), dm__, 0, 0);
    //        }
    //    }
    //}
}

void Force::ibs_force(Simulation_context_base& ctx__,
                      Hamiltonian* H__,
                      K_point* kp__,
                      mdarray<double, 2>& ffac__,
                      mdarray<double, 2>& forcek__)
{
    PROFILE("sirius::Force::ibs_force");

    auto& param = ctx__;
    auto& uc = ctx__.unit_cell();
    //auto rl = ctx__.reciprocal_lattice();

    forcek__.zero();

    dmatrix<double_complex> dm(ctx__.num_fv_states(), ctx__.num_fv_states(), ctx__.blacs_grid(), param.cyclic_block_size(), param.cyclic_block_size());
    compute_dmat(param, kp__, dm);

    auto& fv_evec = kp__->fv_eigen_vectors();

    dmatrix<double_complex> h(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx__.blacs_grid(), param.cyclic_block_size(), param.cyclic_block_size());
    dmatrix<double_complex> o(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx__.blacs_grid(), param.cyclic_block_size(), param.cyclic_block_size());

    dmatrix<double_complex> h1(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx__.blacs_grid(), param.cyclic_block_size(), param.cyclic_block_size());
    dmatrix<double_complex> o1(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx__.blacs_grid(), param.cyclic_block_size(), param.cyclic_block_size());

    dmatrix<double_complex> zm1(kp__->gklo_basis_size(), ctx__.num_fv_states(), ctx__.blacs_grid(), param.cyclic_block_size(), param.cyclic_block_size());
    dmatrix<double_complex> zf(ctx__.num_fv_states(), ctx__.num_fv_states(), ctx__.blacs_grid(), param.cyclic_block_size(), param.cyclic_block_size());

    mdarray<double_complex, 2> alm_row(kp__->num_gkvec_row(), uc.max_mt_aw_basis_size());
    mdarray<double_complex, 2> alm_col(kp__->num_gkvec_col(), uc.max_mt_aw_basis_size());
    mdarray<double_complex, 2> halm_col(kp__->num_gkvec_col(), uc.max_mt_aw_basis_size());

    for (int ia = 0; ia < uc.num_atoms(); ia++)
    {
        h.zero();
        o.zero();

        auto& atom = uc.atom(ia);
        auto& type = atom.type();

        /* generate matching coefficients for current atom */
        kp__->alm_coeffs_row().generate(ia, alm_row);
        kp__->alm_coeffs_col().generate(ia, alm_col);

        /* setup apw-lo and lo-apw blocks */
        H__->set_fv_h_o_apw_lo(kp__, type, atom, ia, alm_row, alm_col, h, o);

        /* apply MT Hamiltonian to column coefficients */
        H__->apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_col(), alm_col, halm_col);

        /* conjugate row (<bra|) matching coefficients */
        for (int i = 0; i < type.mt_aw_basis_size(); i++)
        {
            for (int igk = 0; igk < kp__->num_gkvec_row(); igk++) alm_row(igk, i) = std::conj(alm_row(igk, i));
        }

        /* apw-apw block of the overlap matrix */
        linalg<CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                          alm_row.at<CPU>(), alm_row.ld(), alm_col.at<CPU>(), alm_col.ld(), o.at<CPU>(), o.ld());

        /* apw-apw block of the Hamiltonian matrix */
        linalg<CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                          alm_row.at<CPU>(), alm_row.ld(), halm_col.at<CPU>(), halm_col.ld(), h.at<CPU>(), h.ld());

        int iat = type.id();

        for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) // loop over columns
        {
            auto gvec_col       = kp__->gkvec().gvec(kp__->igk_col(igk_col));
            auto gkvec_col_cart = kp__->gkvec().gkvec_cart(kp__->igk_col(igk_col));
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) // for each column loop over rows
            {
                auto gvec_row       = kp__->gkvec().gvec(kp__->igk_row(igk_row));
                auto gkvec_row_cart = kp__->gkvec().gkvec_cart(kp__->igk_row(igk_row));

                int ig12 = ctx__.gvec().index_g12(gvec_row, gvec_col);

                int igs = ctx__.gvec().shell(ig12);

                double_complex zt = std::conj(ctx__.gvec_phase_factor(ig12, ia)) * ffac__(iat, igs) * fourpi / uc.omega();

                double t1 = 0.5 * dot(gkvec_row_cart, gkvec_col_cart);

                h(igk_row, igk_col) -= t1 * zt;
                o(igk_row, igk_col) -= zt;
            }
        }

        for (int x = 0; x < 3; x++)
        {
            for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) // loop over columns
            {
                auto gvec_col = kp__->gkvec().gvec(kp__->igk_col(igk_col));
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) // for each column loop over rows
                {
                    auto gvec_row = kp__->gkvec().gvec(kp__->igk_row(igk_row));
                    int ig12 = ctx__.gvec().index_g12(gvec_row, gvec_col);

                    vector3d<double> vg = ctx__.gvec().gvec_cart(ig12);
                    h1(igk_row, igk_col) = double_complex(0.0, vg[x]) * h(igk_row, igk_col);
                    o1(igk_row, igk_col) = double_complex(0.0, vg[x]) * o(igk_row, igk_col);
                }
            }

            for (int icol = 0; icol < kp__->num_lo_col(); icol++) {
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) {
                    auto gkvec_row_cart = kp__->gkvec().gkvec_cart(kp__->igk_row(igk_row));
                    h1(igk_row, icol + kp__->num_gkvec_col()) = double_complex(0.0, gkvec_row_cart[x]) * h(igk_row, icol + kp__->num_gkvec_col());
                    o1(igk_row, icol + kp__->num_gkvec_col()) = double_complex(0.0, gkvec_row_cart[x]) * o(igk_row, icol + kp__->num_gkvec_col());
                }
            }

            for (int irow = 0; irow < kp__->num_lo_row(); irow++)
            {
                for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++)
                {
                    auto gkvec_col_cart = kp__->gkvec().gkvec_cart(kp__->igk_col(igk_col));
                    h1(irow + kp__->num_gkvec_row(), igk_col) = double_complex(0.0, -gkvec_col_cart[x]) * h(irow + kp__->num_gkvec_row(), igk_col);
                    o1(irow + kp__->num_gkvec_row(), igk_col) = double_complex(0.0, -gkvec_col_cart[x]) * o(irow + kp__->num_gkvec_row(), igk_col);
                }
            }

            /* zm1 = H * V */
            linalg<CPU>::gemm(0, 0, kp__->gklo_basis_size(), ctx__.num_fv_states(), kp__->gklo_basis_size(),
                              linalg_const<double_complex>::one(), h1, fv_evec, linalg_const<double_complex>::zero(), zm1);

            /* F = V^{+} * zm1 = V^{+} * H * V */
            linalg<CPU>::gemm(2, 0, ctx__.num_fv_states(), ctx__.num_fv_states(), kp__->gklo_basis_size(),
                              linalg_const<double_complex>::one(), fv_evec, zm1, linalg_const<double_complex>::zero(), zf);

            /* zm1 = O * V */
            linalg<CPU>::gemm(0, 0, kp__->gklo_basis_size(), ctx__.num_fv_states(), kp__->gklo_basis_size(),
                              linalg_const<double_complex>::one(), o1, fv_evec, linalg_const<double_complex>::zero(), zm1);

            STOP();
            ///* multiply by energy */
            //for (int i = 0; i < (int)kp__->spl_fv_states().local_size(); i++)
            //{
            //    int ist = kp__->spl_fv_states(i);
            //    for (int j = 0; j < kp__->gklo_basis_size_row(); j++) zm1(j, i) = zm1(j, i) * kp__->fv_eigen_value(ist);
            //}

            /* F = F - V^{+} * zm1 = F - V^{+} * O * (E*V) */
            linalg<CPU>::gemm(2, 0, ctx__.num_fv_states(), ctx__.num_fv_states(), kp__->gklo_basis_size(),
                              double_complex(-1, 0), fv_evec, zm1, double_complex(1, 0), zf);

            for (int i = 0; i < dm.num_cols_local(); i++)
            {
                for (int j = 0; j < dm.num_rows_local(); j++)
                    forcek__(x, ia) += kp__->weight() * real(dm(j, i) * zf(j, i));
            }
        }
    } //ia
}

void Force::total_force(Simulation_context_base& ctx__,
                        Potential* potential__,
                        Density* density__,
                        K_point_set* ks__,
                        mdarray<double, 2>& force__)
{
    PROFILE("sirius::Force::total_force");

    auto& uc = ctx__.unit_cell();

    //auto ffac = ctx__.step_function().get_step_function_form_factors(ctx__.gvec().num_shells(), ctx__.unit_cell(), ctx__.gvec(), ctx__.comm());
    STOP();

    force__.zero();

    mdarray<double, 2> forcek(3, uc.num_atoms());
    for (int ikloc = 0; ikloc < ks__->spl_num_kpoints().local_size(); ikloc++)
    {
        //int ik = ks__->spl_num_kpoints(ikloc);
        //ibs_force(ctx__, band_, (*ks__)[ik], ffac, forcek);
        STOP();
        for (int ia = 0; ia < uc.num_atoms(); ia++)
        {
            for (int x: {0, 1, 2}) force__(x, ia) += forcek(x, ia);
        }
    }
    ctx__.comm().allreduce(&force__(0, 0), (int)force__.size());

    if (ctx__.control().verbosity_ > 2 && ctx__.comm().rank() == 0) {
        printf("\n");
        printf("Forces\n");
        printf("\n");
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            printf("ia : %i, IBS : %12.6f %12.6f %12.6f\n", ia, force__(0, ia), force__(1, ia), force__(2, ia));
        }
    }

    mdarray<double, 2> forcehf(3, uc.num_atoms());

    forcehf.zero();
    for (int ialoc = 0; ialoc < (int)uc.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = uc.spl_num_atoms(ialoc);
        auto g = gradient(potential__->hartree_potential_mt(ialoc));
        for (int x = 0; x < 3; x++) forcehf(x, ia) = uc.atom(ia).zn() * g[x](0, 0) * y00;
    }
    ctx__.comm().allreduce(&forcehf(0, 0), (int)forcehf.size());

    if (ctx__.control().verbosity_ > 2 && ctx__.comm().rank() == 0) {
        printf("\n");
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            printf("ia : %i, Hellmann–Feynman : %12.6f %12.6f %12.6f\n", ia, forcehf(0, ia), forcehf(1, ia), forcehf(2, ia));
        }
    }

    mdarray<double, 2> forcerho(3, uc.num_atoms());
    forcerho.zero();
    for (int ialoc = 0; ialoc < (int)uc.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = uc.spl_num_atoms(ialoc);
        auto g = gradient(density__->density_mt(ialoc));
        for (int x = 0; x < 3; x++) forcerho(x, ia) = inner(potential__->effective_potential_mt(ialoc), g[x]);
    }
    ctx__.comm().allreduce(&forcerho(0, 0), (int)forcerho.size());

    if (ctx__.control().verbosity_ > 2 && ctx__.comm().rank() == 0) {
        printf("\n");
        printf("rho force\n");
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            printf("ia : %i, density contribution : %12.6f %12.6f %12.6f\n", ia, forcerho(0, ia), forcerho(1, ia), forcerho(2, ia));
        }
    }

    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        for (int x = 0; x < 3; x++) {
            force__(x, ia) += (forcehf(x, ia) + forcerho(x, ia));
        }
    }
}

}
