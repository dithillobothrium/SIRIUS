// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file band.h
 *   
 *   \brief Contains declaration and partial implementation of sirius::Band class.
 */

#ifndef __BAND_H__
#define __BAND_H__

#include "periodic_function.h"
#include "k_point_set.h"
#include "potential.h"
#include "local_operator.h"
#include "non_local_operator.h"

namespace sirius
{

// TODO: Band problem is a mess and needs more formal organizaiton. We have different basis functions. 
//       We can do first- and second-variation or a full variation. We can do iterative or exact diagonalization.
//       This has to be organized. 

/// Setup and solve the eigen value problem.
class Band
{
    private:

        /// Simulation context.
        Simulation_context& ctx_;

        /// Alias for the unit cell.
        Unit_cell& unit_cell_;

        /// BLACS grid for distributed linear algebra operations.
        BLACS_grid const& blacs_grid_;

        /// Non-zero Gaunt coefficients
        std::unique_ptr<Gaunt_coefficients<double_complex>> gaunt_coefs_;
        
        /// Interface to a standard eigen-value solver.
        std::unique_ptr<Eigenproblem> std_evp_solver_; 

        /// Interface to a generalized eigen-value solver.
        std::unique_ptr<Eigenproblem> gen_evp_solver_;

        /// Local part of the Hamiltonian operator.
        std::unique_ptr<Local_operator> local_op_;

        /// Solve the band diagonalziation problem with single (full) variation.
        inline int solve_with_single_variation(K_point& kp__, Potential& potential__) const;

        /// Solve the band diagonalziation problem with second variation approach.
        /** This is only used by the FP-LAPW method. */
        inline void solve_with_second_variation(K_point& kp__, Potential& potential__) const;

        /// Solve the first-variational (non-magnetic) problem with exact diagonalization.
        /** This is only used by the LAPW method. */
        inline void diag_fv_exact(K_point* kp__,
                                  Potential& potential__) const;
        
        /// Solve the first-variational (non-magnetic) problem with iterative Davidson diagonalization.
        inline void diag_fv_davidson(K_point* kp__) const;

        /// Apply effective magentic field to the first-variational state.
        /** Must be called first because hpsi is overwritten with B|fv_j>. */
        void apply_magnetic_field(wave_functions& fv_states__,
                                  Gvec const& gkvec__,
                                  std::vector<wave_functions>& hpsi__) const;

        /// Apply SO correction to the first-variational states.
        /** Raising and lowering operators:
         *  \f[
         *      L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
         *  \f]
         */
        void apply_so_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);
        
        /// Apply UJ correction to scalar wave functions
        template <spin_block_t sblock>
        void apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);

        /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap
        inline void set_fv_h_o_it(K_point* kp__,
                                  Potential const& potential__, 
                                  matrix<double_complex>& h__,
                                  matrix<double_complex>& o__) const;

        inline void set_o_it(K_point* kp, mdarray<double_complex, 2>& o) const;

        template <spin_block_t sblock>
        inline void set_h_it(K_point* kp,
                             Periodic_function<double>* effective_potential, 
                             Periodic_function<double>* effective_magnetic_field[3],
                             matrix<double_complex>& h) const;
        
        /// Setup lo-lo block of Hamiltonian and overlap matrices
        inline void set_fv_h_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o) const;

        template <spin_block_t sblock>
        inline void set_h_lo_lo(K_point* kp, mdarray<double_complex, 2>& h) const;
        
        inline void set_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& o) const;
       
        inline void set_o(K_point* kp, mdarray<double_complex, 2>& o);
    
        template <spin_block_t sblock> 
        inline void set_h(K_point* kp,
                          Periodic_function<double>* effective_potential, 
                          Periodic_function<double>* effective_magnetic_field[3],
                          mdarray<double_complex, 2>& h);
       
        inline void apply_fv_o(K_point* kp__,
                               bool apw_only__,
                               bool add_o1__,
                               int N__,
                               int n__,
                               wave_functions& phi__,
                               wave_functions& ophi__) const;

        /// Get singular components of the LAPW overlap matrix.
        /** Singular components are the eigen-vectors with a very small eigen-value. */
        inline void get_singular_components(K_point* kp__) const;

        /// Exact (not iterative) diagonalization of the Hamiltonian.
        template <typename T>
        inline void diag_pseudo_potential_exact(K_point* kp__,
                                                int ispn__,
                                                D_operator<T>& d_op__,
                                                Q_operator<T>& q_op__) const;

        /// Iterative Davidson diagonalization.
        template <typename T>
        inline int diag_pseudo_potential_davidson(K_point* kp__,
                                                  D_operator<T>& d_op__,
                                                  Q_operator<T>& q_op__) const;
        /// RMM-DIIS diagonalization.
        template <typename T>
        inline void diag_pseudo_potential_rmm_diis(K_point* kp__,
                                                   int ispn__,
                                                   D_operator<T>& d_op__,
                                                   Q_operator<T>& q_op__) const;

        template <typename T>
        inline void diag_pseudo_potential_chebyshev(K_point* kp__,
                                                    int ispn__,
                                                    D_operator<T>& d_op__,
                                                    Q_operator<T>& q_op__,
                                                    P_operator<T>& p_op__) const;

        template <typename T>
        inline void apply_h(K_point* kp__,
                            int ispn__, 
                            int N__,
                            int n__,
                            wave_functions& phi__,
                            wave_functions& hphi__,
                            D_operator<T>& d_op) const;

        template <typename T>
        void apply_h_o(K_point* kp__,
                       int ispn__, 
                       int N__,
                       int n__,
                       Wave_functions& phi__,
                       Wave_functions& hphi__,
                       Wave_functions& ophi__,
                       D_operator<T>& d_op,
                       Q_operator<T>& q_op) const;

        /// Auxiliary function used internally by residuals() function.
        inline mdarray<double, 1> residuals_aux(K_point*             kp__,
                                                int                  ispn__,
                                                int                  num_bands__,
                                                std::vector<double>& eval__,
                                                wave_functions&      hpsi__,
                                                wave_functions&      opsi__,
                                                wave_functions&      res__,
                                                mdarray<double, 2>&  h_diag__,
                                                mdarray<double, 1>&  o_diag__) const;

        /// Auxiliary function used internally by residuals() function.
        inline mdarray<double, 1> residuals_aux(K_point*             kp__,
                                                int                  ispn__,
                                                int                  num_bands__,
                                                std::vector<double>& eval__,
                                                Wave_functions&      hpsi__,
                                                Wave_functions&      opsi__,
                                                Wave_functions&      res__,
                                                mdarray<double, 2>&  h_diag__,
                                                mdarray<double, 1>&  o_diag__) const;
        
        template <typename T, typename wave_functions_t>
        int residuals_common(K_point*             kp__,
                             int                  ispn__,
                             int                  N__,
                             int                  num_bands__,
                             std::vector<double>& eval__,
                             std::vector<double>& eval_old__,
                             dmatrix<T>&          evec__,
                             wave_functions_t&    hphi__,
                             wave_functions_t&    ophi__,
                             wave_functions_t&    hpsi__,
                             wave_functions_t&    opsi__,
                             wave_functions_t&    res__,
                             mdarray<double, 2>&  h_diag__,
                             mdarray<double, 1>&  o_diag__) const;

        /// Compute residuals.
        template <typename T>
        inline int residuals(K_point*             kp__,
                             int                  ispn__,
                             int                  N__,
                             int                  num_bands__,
                             std::vector<double>& eval__,
                             std::vector<double>& eval_old__,
                             dmatrix<T>&          evec__,
                             wave_functions&      hphi__,
                             wave_functions&      ophi__,
                             wave_functions&      hpsi__,
                             wave_functions&      opsi__,
                             wave_functions&      res__,
                             mdarray<double, 2>&  h_diag__,
                             mdarray<double, 1>&  o_diag__) const;

        /// Compute residuals.
        template <typename T>
        inline int residuals(K_point*             kp__,
                             int                  ispn__,
                             int                  N__,
                             int                  num_bands__,
                             std::vector<double>& eval__,
                             std::vector<double>& eval_old__,
                             dmatrix<T>&          evec__,
                             Wave_functions&      hphi__,
                             Wave_functions&      ophi__,
                             Wave_functions&      hpsi__,
                             Wave_functions&      opsi__,
                             Wave_functions&      res__,
                             mdarray<double, 2>&  h_diag__,
                             mdarray<double, 1>&  o_diag__) const;
        
        /** Compute \f$ O_{ii'} = \langle \phi_i | \hat O | \phi_{i'} \rangle \f$ operator matrix
         *  for the subspace spanned by the wave-functions \f$ \phi_i \f$. The matrix is always returned
         *  in the CPU pointer because most of the standard math libraries start from the CPU. */
        template <typename T, typename W>
        inline void set_subspace_mtrx(int         num_sc__,
                                      int         N__,
                                      int         n__,
                                      W&          phi__,
                                      W&          op_phi__,
                                      dmatrix<T>& mtrx__,
                                      dmatrix<T>& mtrx_old__) const
        {
            PROFILE("sirius::Band::set_subspace_mtrx");
            
            assert(n__ != 0);
            if (mtrx_old__.size()) {
                assert(&mtrx__.blacs_grid() == &mtrx_old__.blacs_grid());
            }

            /* copy old N x N distributed matrix */
            if (N__ > 0) {
                splindex<block_cyclic> spl_row(N__, mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
                splindex<block_cyclic> spl_col(N__, mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());

                #pragma omp parallel for
                for (int i = 0; i < spl_col.local_size(); i++) {
                    std::memcpy(&mtrx__(0, i), &mtrx_old__(0, i), spl_row.local_size() * sizeof(T));
                }
            }

            /* <{phi,phi_new}|Op|phi_new> */
            inner(num_sc__, phi__, 0, N__ + n__, op_phi__, N__, n__, mtrx__, 0, N__);
            //if (true) {
            //    if (mtrx__.blacs_grid().comm().size() == 1) {
            //        for (int i = 0; i < n__; i++) {
            //            for (int j = 0; j < n__ + N__; j++) {
            //                mtrx__(j, N__ + i) = Utils::round(mtrx__(j, N__ + i), 10);
            //            }
            //        }
            //    }
            //}

            //if (true) {
            //    if (mtrx__.blacs_grid().comm().size() == 1) {
            //        for (int i = 0; i < n__; i++) {
            //            for (int j = 0; j < n__; j++) {
            //                auto zij = mtrx__(N__ + i, N__ + j);
            //                auto zji = mtrx__(N__ + j, N__ + i);
            //                mtrx__(N__ + i, N__ + j) = 0.5 * (zij + std::conj(zji));
            //                mtrx__(N__ + j, N__ + i) = std::conj(mtrx__(N__ + i, N__ + j));
            //            }
            //        }
            //    }
            //}
            
            /* restore lower part */
            if (N__ > 0) {
                if (mtrx__.blacs_grid().comm().size() == 1) {
                    #pragma omp parallel for
                    for (int i = 0; i < N__; i++) {
                        for (int j = N__; j < N__ + n__; j++) {
                            mtrx__(j, i) = type_wrapper<T>::bypass(std::conj(mtrx__(i, j)));
                        }
                    }
                } else {
                    linalg<CPU>::tranc(n__, N__, mtrx__, 0, N__, mtrx__, N__, 0);
                }
            }

            //if (true) {
            //    splindex<block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
            //    splindex<block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());
            //    for (int i = 0; i < spl_col.local_size(); i++) {
            //        for (int j = 0; j < spl_row.local_size(); j++) {
            //            if (std::abs(mtrx__(j, i)) < 1e-11) {
            //                mtrx__(j, i) = 0;
            //            }
            //        }
            //    }
            //}

            if (ctx_.control().print_checksum_) {
                splindex<block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
                splindex<block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());
                double_complex cs(0, 0);
                for (int i = 0; i < spl_col.local_size(); i++) {
                    for (int j = 0; j < spl_row.local_size(); j++) {
                        cs += mtrx__(j, i);
                    }
                }
                mtrx__.blacs_grid().comm().allreduce(&cs, 1);
                if (mtrx__.blacs_grid().comm().rank() == 0) {
                    print_checksum("subspace_mtrx", cs);
                }
            }

            mtrx__.make_real_diag(N__ + n__);

            /* save new matrix */
            if (mtrx_old__.size()) {
                splindex<block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
                splindex<block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());

                #pragma omp parallel for
                for (int i = 0; i < spl_col.local_size(); i++) {
                    std::memcpy(&mtrx_old__(0, i), &mtrx__(0, i), spl_row.local_size() * sizeof(T));
                }
            }
        }
                
        /// Diagonalize a pseudo-potential Hamiltonian.
        template <typename T>
        int diag_pseudo_potential(K_point* kp__) const
        {
            PROFILE("sirius::Band::diag_pseudo_potential");

            local_op_->prepare(kp__->gkvec());
            ctx_.fft_coarse().prepare(kp__->gkvec().partition());

            D_operator<T> d_op(ctx_, kp__->beta_projectors());
            Q_operator<T> q_op(ctx_, kp__->beta_projectors());

            int niter{0};

            auto& itso = ctx_.iterative_solver_input();
            if (itso.type_ == "exact") {
                if (ctx_.num_mag_dims() != 3) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        diag_pseudo_potential_exact(kp__, ispn, d_op, q_op);
                    }
                } else {
                    STOP();
                }
            } else if (itso.type_ == "davidson") {
                niter = diag_pseudo_potential_davidson(kp__, d_op, q_op);
            } else if (itso.type_ == "rmm-diis") {
                if (ctx_.num_mag_dims() != 3) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        diag_pseudo_potential_rmm_diis(kp__, ispn, d_op, q_op);
                    }
                } else {
                    STOP();
                }
            } else if (itso.type_ == "chebyshev") {
                P_operator<T> p_op(ctx_, kp__->beta_projectors(), kp__->p_mtrx());
                if (ctx_.num_mag_dims() != 3) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        diag_pseudo_potential_chebyshev(kp__, ispn, d_op, q_op, p_op);

                    }
                } else {
                    STOP();
                }
            } else {
                TERMINATE("unknown iterative solver type");
            }

            ctx_.fft_coarse().dismiss();
            return niter;
        }

    public:
        
        /// Constructor
        Band(Simulation_context& ctx__)
            : ctx_(ctx__)
            , unit_cell_(ctx__.unit_cell())
            , blacs_grid_(ctx__.blacs_grid())
        {
            PROFILE("sirius::Band::Band");

            gaunt_coefs_ = std::unique_ptr<Gaunt_coefficients<double_complex>>(
                new Gaunt_coefficients<double_complex>(ctx_.lmax_apw(), 
                                                       ctx_.lmax_pot(), 
                                                       ctx_.lmax_apw(),
                                                       SHT::gaunt_hybrid));
            
            Eigenproblem* ptr;
            /* create standard eigen-value solver */
            switch (ctx_.std_evp_solver_type()) {
                case ev_lapack: {
                    ptr = new Eigenproblem_lapack(2 * linalg_base::dlamch('S'));
                    break;
                }
                #ifdef __SCALAPACK
                case ev_scalapack: {
                    ptr = new Eigenproblem_scalapack(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size(), 1e-12);
                    break;
                }
                #endif
                #ifdef __PLASMA
                case ev_plasma: {
                    ptr = new Eigenproblem_plasma();
                    break;
                }
                #endif
                #ifdef __MAGMA
                case ev_magma: {
                    ptr = new Eigenproblem_magma();
                    break;
                }
                #endif
                #ifdef __ELPA
                case ev_elpa1: {
                    ptr = new Eigenproblem_elpa1(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                case ev_elpa2: {
                    ptr = new Eigenproblem_elpa2(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                #endif
                default: {
                    TERMINATE("wrong standard eigen-value solver");
                }
            }
            std_evp_solver_ = std::unique_ptr<Eigenproblem>(ptr);
            
            /* create generalized eign-value solver */
            switch (ctx_.gen_evp_solver_type()) {
                case ev_lapack: {
                    ptr = new Eigenproblem_lapack(2 * linalg_base::dlamch('S'));
                    break;
                }
                #ifdef __SCALAPACK
                case ev_scalapack: {
                    ptr = new Eigenproblem_scalapack(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size(), 1e-12);
                    break;
                }
                #endif
                #ifdef __ELPA
                case ev_elpa1: {
                    ptr = new Eigenproblem_elpa1(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                case ev_elpa2: {
                    ptr = new Eigenproblem_elpa2(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                #endif
                #ifdef __MAGMA
                case ev_magma: {
                    ptr = new Eigenproblem_magma();
                    break;
                }
                #endif
                #ifdef __RS_GEN_EIG
                case ev_rs_gpu: {
                    ptr = new Eigenproblem_RS_GPU(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
                    break;
                }
                case ev_rs_cpu: {
                    ptr = new Eigenproblem_RS_CPU(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
                    break;
                }
                #endif
                default: {
                    TERMINATE("wrong generalized eigen-value solver");
                }
            }
            gen_evp_solver_ = std::unique_ptr<Eigenproblem>(ptr);

            if (std_evp_solver_->parallel() != gen_evp_solver_->parallel()) {
                TERMINATE("both eigen-value solvers must be serial or parallel");
            }

            if (!std_evp_solver_->parallel() && blacs_grid_.comm().size() > 1) {
                TERMINATE("eigen-value solvers must be parallel");
            }

            local_op_ = std::unique_ptr<Local_operator>(new Local_operator(ctx_, ctx_.fft_coarse()));
        }

        /// Apply the muffin-tin part of the Hamiltonian to the apw basis functions of an atom.
        /** The following matrix is computed:
         *  \f[
         *    b_{L_2 \nu_2}^{\alpha}({\bf G'}) = \sum_{L_1 \nu_1} \sum_{L_3} 
         *      a_{L_1\nu_1}^{\alpha}({\bf G'}) 
         *      \langle u_{\ell_1\nu_1}^{\alpha} | h_{L3}^{\alpha} |  u_{\ell_2\nu_2}^{\alpha}  
         *      \rangle  \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle
         *  \f] 
         */
        template <spin_block_t sblock>
        void apply_hmt_to_apw(Atom const&                 atom__,
                              int                         num_gkvec__,
                              mdarray<double_complex, 2>& alm__,
                              mdarray<double_complex, 2>& halm__) const
        {
            auto& type = atom__.type();

            // TODO: this is k-independent and can in principle be precomputed together with radial integrals if memory is available
            // TODO: for spin-collinear case hmt is Hermitian; compute upper triangular part and use zhemm
            mdarray<double_complex, 2> hmt(type.mt_aw_basis_size(), type.mt_aw_basis_size());
            /* compute the muffin-tin Hamiltonian */
            for (int j2 = 0; j2 < type.mt_aw_basis_size(); j2++) {
                int lm2    = type.indexb(j2).lm;
                int idxrf2 = type.indexb(j2).idxrf;
                for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
                    int lm1    = type.indexb(j1).lm;
                    int idxrf1 = type.indexb(j1).idxrf;
                    hmt(j1, j2) = atom__.radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
                }
            }
            linalg<CPU>::gemm(0, 1, num_gkvec__, type.mt_aw_basis_size(), type.mt_aw_basis_size(), alm__, hmt, halm__);
        }

        void apply_o1mt_to_apw(Atom const&                 atom__,
                               int                         num_gkvec__,
                               mdarray<double_complex, 2>& alm__,
                               mdarray<double_complex, 2>& oalm__) const
        {
            auto& type = atom__.type();

            for (int j = 0; j < type.mt_aw_basis_size(); j++) {
                int l     = type.indexb(j).l;
                int lm    = type.indexb(j).lm;
                int idxrf = type.indexb(j).idxrf;
                for (int order = 0; order < type.aw_order(l); order++) {
                    int j1 = type.indexb().index_by_lm_order(lm, order);
                    int idxrf1 = type.indexr().index_by_l_order(l, order);
                    for (int ig = 0; ig < num_gkvec__; ig++) {
                        oalm__(ig, j) += atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1) * alm__(ig, j1);
                    }
                }
            }
        }
 
        /// Setup apw-lo and lo-apw blocs of Hamiltonian and overlap matrices
        void set_fv_h_o_apw_lo(K_point* kp,
                               Atom_type const& type,
                               Atom const& atom,
                               int ia,
                               mdarray<double_complex, 2>& alm_row,
                               mdarray<double_complex, 2>& alm_col,
                               mdarray<double_complex, 2>& h,
                               mdarray<double_complex, 2>& o) const;
        
        template <spin_block_t sblock>
        void set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                          mdarray<double_complex, 2>& h);
        
        /// Set APW-lo and lo-APW blocks of the overlap matrix.
        void set_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                          mdarray<double_complex, 2>& o);

        /// Setup the Hamiltonian and overlap matrices in APW+lo basis
        /** The Hamiltonian matrix has the following expression:
         *  \f[
         *      H_{\mu' \mu}=\langle \varphi_{\mu' } | \hat H | \varphi_{\mu } \rangle  = 
         *      \left( \begin{array}{cc} 
         *         H_{\bf G'G} & H_{{\bf G'}j} \\
         *         H_{j'{\bf G}} & H_{j'j}
         *      \end{array} \right)
         *  \f]
         *  APW-APW block:
         *  \f{eqnarray*}{
         *      H_{{\bf G'} {\bf G}}^{\bf k} &=& \sum_{\alpha} \sum_{L'\nu', L\nu} a_{L'\nu'}^{\alpha *}({\bf G'+k}) 
         *      \langle  u_{\ell' \nu'}^{\alpha}Y_{\ell' m'}|\hat h^{\alpha} | u_{\ell \nu}^{\alpha}Y_{\ell m}  \rangle 
         *       a_{L\nu}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde V_{eff}({\bf G - G'}) \\
         *          &=& \sum_{\alpha} \sum_{\xi' } a_{\xi'}^{\alpha *}({\bf G'+k}) 
         *              b_{\xi'}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde V_{eff}({\bf G - G'})  
         *  \f}
         *  APW-lo block:
         *  \f[
         *      H_{{\bf G'} j}^{\bf k} = \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'+k}) 
         *      \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} |  \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j}  \rangle 
         *  \f]
         *  lo-APW block:
         *  \f[
         *      H_{j' {\bf G}}^{\bf k} = \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G+k}) 
         *  \f]
         *  lo-lo block:
         *  \f[
         *      H_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          |\hat h^{\alpha_{j}} |  \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j}  \rangle  \delta_{\alpha_j \alpha_{j'}}
         *  \f]
         *
         *  The overlap matrix has the following expression:
         *  \f[
         *      O_{\mu' \mu} = \langle \varphi_{\mu'} | \varphi_{\mu} \rangle
         *  \f]
         *  APW-APW block:
         *  \f[
         *      O_{{\bf G'} {\bf G}}^{\bf k} = \sum_{\alpha} \sum_{L\nu} a_{L\nu}^{\alpha *}({\bf G'+k}) 
         *      a_{L\nu}^{\alpha}({\bf G+k}) + \Theta({\bf G-G'})
         *  \f]
         *  
         *  APW-lo block:
         *  \f[
         *      O_{{\bf G'} j}^{\bf k} = \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'+k}) 
         *      \langle u_{\ell_j \nu'}^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} \rangle
         *  \f]
         *
         *  lo-APW block:
         *  \f[
         *      O_{j' {\bf G}}^{\bf k} = 
         *      \sum_{\nu'} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | u_{\ell_{j'} \nu'}^{\alpha_{j'}} \rangle
         *      a_{\ell_{j'} m_{j'} \nu'}^{\alpha_{j'}}({\bf G+k}) 
         *  \f]
         *
         *  lo-lo block:
         *  \f[
         *      O_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | 
         *      \phi_{\ell_{j}}^{\zeta_{j} \alpha_{j}} \rangle \delta_{\alpha_{j'} \alpha_j} 
         *      \delta_{\ell_{j'} \ell_j} \delta_{m_{j'} m_j}
         *  \f]
         */
        template <device_t pu, electronic_structure_method_t basis>
        inline void set_fv_h_o(K_point* kp,
                               Potential const& potential__,
                               dmatrix<double_complex>& h,
                               dmatrix<double_complex>& o) const;
        
        /// Apply LAPW Hamiltonain and overlap to the trial wave-functions.
        /** Check the documentation of Band::set_fv_h_o() for the expressions of Hamiltonian and overlap
         *  matrices and \ref basis for the definition of the LAPW+lo basis. 
         *
         *  For the set of wave-functions expanded in LAPW+lo basis (k-point index is dropped for simplicity)
         *  \f[
         *      \psi_{i} = \sum_{\mu} \phi_{\mu} C_{\mu i}
         *  \f]
         *  where \f$ \mu = \{ {\bf G}, j \} \f$ is a combined index of LAPW and local orbitals we want to contrusct
         *  a subspace Hamiltonian and overlap matrices:
         *  \f[
         *      H_{i' i} = \langle \psi_{i'} | \hat H | \psi_i \rangle =
         *          \sum_{\mu' \mu} C_{\mu' i'}^{*} \langle \phi_{\mu'} | \hat H | \phi_{\mu} \rangle C_{\mu i} = 
         *          \sum_{\mu'} C_{\mu' i'}^{*} h_{\mu' i}(\psi)
         *  \f]
         *  \f[
         *      O_{i' i} = \langle \psi_{i'} | \psi_i \rangle =
         *          \sum_{\mu' \mu} C_{\mu' i'}^{*} \langle \phi_{\mu'} | \phi_{\mu} \rangle C_{\mu i} = 
         *          \sum_{\mu'} C_{\mu' i'}^{*} o_{\mu' i}(\psi)
         *  \f]
         *  where
         *  \f[
         *      h_{\mu' i}(\psi) = \sum_{\mu} \langle \phi_{\mu'} | \hat H | \phi_{\mu} \rangle C_{\mu i}
         *  \f]
         *  and
         *  \f[
         *      o_{\mu' i}(\psi) = \sum_{\mu} \langle \phi_{\mu'} | \phi_{\mu} \rangle C_{\mu i}
         *  \f]
         *  For the APW block of \f$  h_{\mu' i}(\psi)  \f$ and \f$  o_{\mu' i}(\psi)  \f$ we have:
         *  \f[
         *       h_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} + 
         *          \sum_{j} \langle \phi_{\bf G'} | \hat H | \phi_{j} \rangle C_{j i}
         *  \f]
         *  \f[
         *       o_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \phi_{\bf G} \rangle C_{{\bf G} i} + 
         *          \sum_{j} \langle \phi_{\bf G'} | \phi_{j} \rangle C_{j i}
         *  \f]
         *  and for the lo block:
         *  \f[
         *       h_{j' i}(\psi) = \sum_{{\bf G}} \langle \phi_{j'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} + 
         *          \sum_{j} \langle \phi_{j'} | \hat H | \phi_{j} \rangle C_{j i}
         *  \f]
         *  \f[
         *       o_{j' i}(\psi) = \sum_{{\bf G}} \langle \phi_{j'} |  \phi_{\bf G} \rangle C_{{\bf G} i} + 
         *          \sum_{j} \langle \phi_{j'} | \phi_{j} \rangle C_{j i}
         *  \f]
         *
         *  APW-APW contribution, muffin-tin part:
         *  \f[
         *      h_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} = 
         *          \sum_{{\bf G}} \sum_{\alpha} \sum_{\xi'} a_{\xi'}^{\alpha *}({\bf G'}) b_{\xi'}^{\alpha}({\bf G}) 
         *           C_{{\bf G} i} 
         *  \f]
         *  \f[
         *      o_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \phi_{\bf G} \rangle C_{{\bf G} i} = 
         *          \sum_{{\bf G}} \sum_{\alpha} \sum_{\xi'} a_{\xi'}^{\alpha *}({\bf G'}) a_{\xi'}^{\alpha}({\bf G}) 
         *           C_{{\bf G} i} 
         *  \f]
         *  APW-APW contribution, interstitial effective potential part:
         *  \f[
         *      h_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} V({\bf r}) \psi_{i}({\bf r}) d{\bf r}
         *  \f]
         *  This is done by transforming \f$ \psi_i({\bf G}) \f$ to the real space, multiplying by effectvive potential
         *  and step function and transforming the result back to the \f$ {\bf G} \f$ domain.
         *
         *  APW-APW contribution, interstitial kinetic energy part:
         *  \f[
         *      h_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} \Big( -\frac{1}{2} \nabla \Big) 
         *          \Big( \nabla \psi_{i}({\bf r}) \Big) d{\bf r}
         *  \f]
         *  and the gradient of the wave-function is computed with FFT as:
         *  \f[
         *      \Big( \nabla \psi_{i}({\bf r}) \Big) = \sum_{\bf G} i{\bf G}e^{i{\bf G}{\bf r}}\psi_i({\bf G})  
         *  \f]
         *
         *  APW-APW contribution, interstitial overlap:
         *  \f[
         *      o_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} \psi_{i}({\bf r}) d{\bf r}
         *  \f]
         *
         *  APW-lo contribution:
         *  \f[
         *      h_{{\bf G'} i}(\psi) =  \sum_{j} \langle \phi_{\bf G'} | \hat H | \phi_{j} \rangle C_{j i} = 
         *      \sum_{j} C_{j i}   \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'}) 
         *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j} \rangle = 
         *      \sum_{j} C_{j i} \sum_{\xi'} a_{\xi'}^{\alpha_j *}({\bf G'}) h_{\xi' \xi_j}^{\alpha_j}  
         *  \f]
         *  \f[
         *      o_{{\bf G'} i}(\psi) =  \sum_{j} \langle \phi_{\bf G'} | \phi_{j} \rangle C_{j i} = 
         *      \sum_{j} C_{j i}   \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'}) 
         *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}| \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j} \rangle = 
         *      \sum_{j} C_{j i} \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'}) o_{\nu' \zeta_j \ell_j}^{\alpha_j}  
         *  \f]
         *  lo-APW contribution:
         *  \f[
         *     h_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{\xi} h_{\xi_{j'} \xi}^{\alpha_{j'}} a_{\xi}^{\alpha_{j'}}({\bf G})
         *  \f]
         *  \f[
         *     o_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} |  \phi_{\bf G} \rangle C_{{\bf G} i} = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{\nu} o_{\zeta_{j'} \nu \ell_{j'}}^{\alpha_{j'}} a_{\ell_{j'} m_{j'} \nu}^{\alpha_{j'}}({\bf G})
         *  \f]
         *  lo-lo contribution:
         *  \f[
         *      h_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \hat H | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i} h_{\xi_{j'} \xi_j}^{\alpha_j}
         *          \delta_{\alpha_j \alpha_{j'}}
         *  \f]
         *  \f[
         *      o_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i} 
         *          o_{\zeta_{j'} \zeta_{j} \ell_j}^{\alpha_j}
         *            \delta_{\alpha_j \alpha_{j'}} \delta_{\ell_j \ell_{j'}} \delta_{m_j m_{j'}}
         *  \f]
         */
        inline void apply_fv_h_o(K_point* kp__,
                                 int nlo,
                                 int N,
                                 int n,
                                 wave_functions& phi__,
                                 wave_functions& hphi__,
                                 wave_functions& ophi__) const;

        /// Solve second-variational problem.
        inline void diag_sv(K_point* kp,
                            Potential& potential__) const;
        
        /// Solve \f$ \hat H \psi = E \psi \f$ and find eigen-states of the Hamiltonian.
        inline void solve_for_kset(K_point_set& kset__,
                                   Potential& potential__,
                                   bool precompute__) const;

        inline Eigenproblem const& std_evp_solver() const
        {
            return *std_evp_solver_;
        }

        inline Eigenproblem const& gen_evp_solver() const
        {
            return *gen_evp_solver_;
        }

        /// Get diagonal elements of LAPW Hamiltonian.
        inline mdarray<double, 2> get_h_diag(K_point* kp__,
                                             double v0__,
                                             double theta0__) const;

        /// Get diagonal elements of LAPW overlap.
        inline mdarray<double, 1> get_o_diag(K_point* kp__,
                                             double theta0__) const;

        /// Get diagonal elements of pseudopotential Hamiltonian.
        template <typename T>
        inline mdarray<double, 2> get_h_diag(K_point*        kp__,
                                             Local_operator& v_loc__,
                                             D_operator<T>&  d_op__) const;

        /// Get diagonal elements of pseudopotential overlap matrix.
        template <typename T>
        inline mdarray<double, 1> get_o_diag(K_point*       kp__,
                                             Q_operator<T>& q_op__) const;

        /// Initialize the subspace for the entire k-point set.
        inline void initialize_subspace(K_point_set& kset__,
                                        Potential&   potential__) const;

        /// Initialize the wave-functions subspace.
        template <typename T>
        inline void initialize_subspace(K_point*                                        kp__,
                                        int                                             num_ao__,
                                        Radial_grid_lin<double>&                        qgrid__,
                                        std::vector<std::vector<Spline<double>>> const& rad_int__) const;
};

#include "Band/get_h_o_diag.hpp"
#include "Band/apply.hpp"
#include "Band/set_lapw_h_o.hpp"
#include "Band/residuals.hpp"
#include "Band/diag_full_potential.hpp"
#include "Band/diag_pseudo_potential.hpp"
#include "Band/initialize_subspace.hpp"
#include "Band/solve.hpp"

}

#endif // __BAND_H__
