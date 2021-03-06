/// Inner product between wave-functions.
/** The result is always returned in the CPU pointer. In case of a single MPI rank the result is also returned in the
 *  GPU pointer */
template <typename T>
inline void inner(wave_functions& bra__,
                  int             i0__,
                  int             m__,
                  wave_functions& ket__,
                  int             j0__,
                  int             n__,
                  double          beta__,
                  dmatrix<T>&     result__,
                  int             irow0__,
                  int             jcol0__)
{
    PROFILE("sddk::wave_functions::inner");

    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");
    
    assert(&bra__.comm() == &ket__.comm());
    assert(bra__.pw_coeffs().num_rows_loc() == ket__.pw_coeffs().num_rows_loc());
    if (bra__.has_mt()) {
        assert(bra__.mt_coeffs().num_rows_loc() == ket__.mt_coeffs().num_rows_loc());
    }

    auto& comm = bra__.comm();
    auto pu = bra__.pu();
    
    const char* sddk_pp_raw = std::getenv("SDDK_PRINT_PERFORMANCE");
    int sddk_pp = (sddk_pp_raw == NULL) ? 0 : std::atoi(sddk_pp_raw);

    const char* sddk_bs_raw = std::getenv("SDDK_BLOCK_SIZE");
    int sddk_block_size = (sddk_bs_raw == NULL) ? sddk_default_block_size : std::atoi(sddk_bs_raw);

    double ngop{0};
    if (std::is_same<T, double>::value) {
        ngop = 2e-9;
    }
    if (std::is_same<T, double_complex>::value) {
        ngop = 8e-9;
    }

    if (sddk_pp) {
        comm.barrier();
    }
    double time = -omp_get_wtime();

    T alpha = (std::is_same<T, double_complex>::value) ? 1 : 2;
    T beta = beta__;

    auto local_inner = [&](int i0__,
                           int m__,
                           int j0__,
                           int n__,
                           T*  buf__,
                           int ld__,
                           int stream_id){
        /* wave-functions are complex and inner product is complex */
        if (std::is_same<T, double_complex>::value) {
            switch (pu) {
                case CPU: {
                    linalg<CPU>::gemm(2, 0, m__, n__, bra__.pw_coeffs().num_rows_loc(),
                                      *reinterpret_cast<double_complex*>(&alpha),
                                      bra__.pw_coeffs().prime().at<CPU>(0, i0__), bra__.pw_coeffs().prime().ld(),
                                      ket__.pw_coeffs().prime().at<CPU>(0, j0__), ket__.pw_coeffs().prime().ld(),
                                      *reinterpret_cast<double_complex*>(&beta),
                                      reinterpret_cast<double_complex*>(buf__), ld__);
                    if (bra__.has_mt() && bra__.mt_coeffs().num_rows_loc()) {
                        linalg<CPU>::gemm(2, 0, m__, n__, bra__.mt_coeffs().num_rows_loc(),
                                          *reinterpret_cast<double_complex*>(&alpha),
                                          bra__.mt_coeffs().prime().at<CPU>(0, i0__), bra__.mt_coeffs().prime().ld(),
                                          ket__.mt_coeffs().prime().at<CPU>(0, j0__), ket__.mt_coeffs().prime().ld(),
                                          linalg_const<double_complex>::one(),
                                          reinterpret_cast<double_complex*>(buf__), ld__);
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, m__, n__, bra__.pw_coeffs().num_rows_loc(),
                                      reinterpret_cast<double_complex*>(&alpha),
                                      bra__.pw_coeffs().prime().at<GPU>(0, i0__), bra__.pw_coeffs().prime().ld(),
                                      ket__.pw_coeffs().prime().at<GPU>(0, j0__), ket__.pw_coeffs().prime().ld(),
                                      reinterpret_cast<double_complex*>(&beta),
                                      reinterpret_cast<double_complex*>(buf__), ld__,
                                      stream_id);
                    if (bra__.has_mt() && bra__.mt_coeffs().num_rows_loc()) {
                        linalg<GPU>::gemm(2, 0, m__, n__, bra__.mt_coeffs().num_rows_loc(),
                                          reinterpret_cast<double_complex*>(&alpha),
                                          bra__.mt_coeffs().prime().at<GPU>(0, i0__), bra__.mt_coeffs().prime().ld(),
                                          ket__.mt_coeffs().prime().at<GPU>(0, j0__), ket__.mt_coeffs().prime().ld(),
                                          &linalg_const<double_complex>::one(),
                                          reinterpret_cast<double_complex*>(buf__), ld__,
                                          stream_id);
                    }
                    #endif
                    break;
                }
            }
        }
        /* wave-functions are real and inner product is also real */
        if (std::is_same<T, double>::value) {
            if (bra__.has_mt() && bra__.mt_coeffs().num_rows_loc()) {
                TERMINATE("not implemented");
            }
            switch (pu) {
                case CPU: {
                    linalg<CPU>::gemm(2, 0, m__, n__, 2 * bra__.pw_coeffs().num_rows_loc(),
                                      *reinterpret_cast<double*>(&alpha),
                                      reinterpret_cast<double*>(bra__.pw_coeffs().prime().at<CPU>(0, i0__)), 2 * bra__.pw_coeffs().prime().ld(),
                                      reinterpret_cast<double*>(ket__.pw_coeffs().prime().at<CPU>(0, j0__)), 2 * ket__.pw_coeffs().prime().ld(),
                                      *reinterpret_cast<double*>(&beta),
                                      reinterpret_cast<double*>(buf__), ld__);
                    /* subtract one extra G=0 contribution */
                    if (comm.rank() == 0) {
                        linalg<CPU>::ger(m__, n__, -1.0,
                                         reinterpret_cast<double*>(bra__.pw_coeffs().prime().at<CPU>(0, i0__)), 2 * bra__.pw_coeffs().prime().ld(),
                                         reinterpret_cast<double*>(ket__.pw_coeffs().prime().at<CPU>(0, j0__)), 2 * ket__.pw_coeffs().prime().ld(),
                                         reinterpret_cast<double*>(buf__), ld__); 

                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    linalg<GPU>::gemm(2, 0, m__, n__, 2 * bra__.pw_coeffs().num_rows_loc(),
                                      reinterpret_cast<double*>(&alpha),
                                      reinterpret_cast<double*>(bra__.pw_coeffs().prime().at<GPU>(0, i0__)), 2 * bra__.pw_coeffs().prime().ld(),
                                      reinterpret_cast<double*>(ket__.pw_coeffs().prime().at<GPU>(0, j0__)), 2 * ket__.pw_coeffs().prime().ld(),
                                      reinterpret_cast<double*>(&beta),
                                      reinterpret_cast<double*>(buf__), ld__,
                                      stream_id);
                    /* subtract one extra G=0 contribution */
                    if (comm.rank() == 0) {
                        linalg<GPU>::ger(m__, n__, &linalg_const<double>::m_one(),
                                         reinterpret_cast<double*>(bra__.pw_coeffs().prime().at<GPU>(0, i0__)), 2 * bra__.pw_coeffs().prime().ld(),
                                         reinterpret_cast<double*>(ket__.pw_coeffs().prime().at<GPU>(0, j0__)), 2 * ket__.pw_coeffs().prime().ld(),
                                         reinterpret_cast<double*>(buf__), ld__,
                                         stream_id);
                    }
                    #endif
                    break;
                }
            }
        }
    };

    if (comm.size() == 1) {
        T* buf = (pu == CPU) ? result__.template at<CPU>(irow0__, jcol0__) : result__.template at<GPU>(irow0__, jcol0__);
        local_inner(i0__, m__, j0__, n__, buf, result__.ld(), -1);
        #ifdef __GPU
        if (pu == GPU) {
            acc::copyout(result__.template at<CPU>(irow0__, jcol0__), result__.ld(),
                         result__.template at<GPU>(irow0__, jcol0__), result__.ld(),
                         m__, n__);
        }
        #endif
        if (sddk_pp) {
            time += omp_get_wtime();
            int k = bra__.pw_coeffs().num_rows_loc();
            if (bra__.has_mt()) {
                k += bra__.mt_coeffs().num_rows_loc();
            }
            printf("inner() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n", ngop * m__ * n__ * k / time, m__, n__, k, time);
        }
        return;
    }
    
    const int BS = sddk_block_size;

    mdarray<T, 2> c_tmp(BS * BS, 2, memory_t::host_pinned, "inner::c_tmp");
    if (pu == GPU) {
        c_tmp.allocate(memory_t::device);
    }

    int nbr = m__ / BS + std::min(1, m__ % BS);
    int nbc = n__ / BS + std::min(1, n__ % BS);

    std::array<MPI_Request, 2> req = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    std::array<std::array<int, 4>, 2> dims;

    if (pu == GPU) {
        #ifdef __GPU
        /* state of the buffers:
         * state = 0: buffer is free
         * state = 1: buffer stores result of local zgemm */
        int buf_state[] = {0, 0};
         
        omp_set_nested(1);
        int nt = omp_get_max_threads();
        if (nt < 2) {
            TERMINATE("minimum two threads are required");
        }

        #pragma omp parallel num_threads(2) shared(buf_state)
        {
            if (omp_get_thread_num() == 0) {
                omp_set_num_threads(nt - 1);
            }

            int s{0};
            for (int ibc = 0; ibc < nbc; ibc++) {
                int j0 = ibc * BS;
                int ncol = std::min(n__, (ibc + 1) * BS) - j0;

                for (int ibr = 0; ibr < nbr; ibr++) {
                    int i0 = ibr * BS;
                    int nrow = std::min(m__, (ibr + 1) * BS) - i0;

                    /* this thread will call cudaZgemm */
                    if (omp_get_thread_num() == 1) {
                        int state{1};
                        /* wait for the release of the buffer */
                        while (state) {
                            #pragma omp atomic read
                            state = buf_state[s % 2];
                        }

                        T* buf = (pu == CPU) ? c_tmp.template at<CPU>(0, s % 2) : c_tmp.template at<GPU>(0, s % 2);
                        local_inner(i0__ + i0, nrow, j0__ + j0, ncol, buf, nrow, s % 2);

                        #ifdef __GPU
                        if (pu == GPU) {
                            acc::copyout(c_tmp.template at<CPU>(0, s % 2), c_tmp.template at<GPU>(0, s % 2), nrow * ncol, s % 2);
                        }
                        #endif

                        #pragma omp atomic write
                        /* lock the buffer */
                        buf_state[s % 2] = 1;
                    } else { /* this thread will do allreduce and store */
                        int state{0};
                        /* wait for the lock of the buffer */
                        while (!state) {
                            #pragma omp atomic read
                            state = buf_state[s % 2];
                        }
                        /* wait for the cuda stream */
                        #ifdef __GPU
                        if (pu == GPU) {
                            acc::sync_stream(s % 2);
                        }
                        #endif
                        
                        comm.allreduce(c_tmp.template at<CPU>(0, s % 2), nrow * ncol);

                        /* store panel */
                        #pragma omp parallel for
                        for (int jcol = 0; jcol < ncol; jcol++) {
                            for (int irow = 0; irow < nrow; irow++) {
                                result__.set(irow0__ + irow + i0, jcol0__ + jcol + j0,
                                             c_tmp(irow + nrow * jcol, s % 2));
                            }
                        }

                        #pragma omp atomic write
                        /* release the buffer */
                        buf_state[s % 2] = 0;
                    }
                    s++;
                }
            }
        }
        omp_set_nested(0);
        omp_set_num_threads(nt);
        #endif
    }
    
    if (pu == CPU) {
        auto store_panel = [&req, &result__, &dims, &c_tmp, irow0__, jcol0__](int s)
        {
            MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);

            #pragma omp parallel for
            for (int jcol = 0; jcol < dims[s % 2][3]; jcol++) {
                for (int irow = 0; irow < dims[s % 2][2]; irow++) {
                    result__.set(irow0__ + irow +  dims[s % 2][0], jcol0__ + jcol +  dims[s % 2][1],
                                 c_tmp(irow + dims[s % 2][2] * jcol, s % 2));
                }
            }
        };

        int s{0};
        for (int ibc = 0; ibc < nbc; ibc++) {
            int j0 = ibc * BS;
            int ncol = std::min(n__, (ibc + 1) * BS) - j0;

            for (int ibr = 0; ibr < nbr; ibr++) {
                int i0 = ibr * BS;
                int nrow = std::min(m__, (ibr + 1) * BS) - i0;

                if (req[s % 2] != MPI_REQUEST_NULL) {
                    store_panel(s);
                }

                dims[s % 2][0] = i0;
                dims[s % 2][1] = j0;
                dims[s % 2][2] = nrow;
                dims[s % 2][3] = ncol;

                T* buf = (pu == CPU) ? c_tmp.template at<CPU>(0, s % 2) : c_tmp.template at<GPU>(0, s % 2);
                local_inner(i0__ + i0, nrow, j0__ + j0, ncol, buf, nrow, -1);

                #ifdef __GPU
                if (pu == GPU) {
                    acc::copyout(c_tmp.template at<CPU>(0, s % 2), c_tmp.template at<GPU>(0, s % 2), nrow * ncol);
                }
                #endif

                comm.iallreduce(c_tmp.template at<CPU>(0, s % 2), nrow * ncol, &req[s % 2]);
                
                s++;
            }
        }

        for (int s: {0, 1}) {
            if (req[s % 2] != MPI_REQUEST_NULL) {
                store_panel(s);
            }
        }
    }

    if (sddk_pp) {
        comm.barrier();
        time += omp_get_wtime();
        int k = bra__.pw_coeffs().num_rows_loc();
        if (bra__.has_mt()) {
            k += bra__.mt_coeffs().num_rows_loc();
        }
        comm.allreduce(&k, 1);
        if (comm.rank() == 0) {
            printf("inner() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n", ngop * m__ * n__ * k / time / comm.size(), m__, n__, k, time);
        }
    }
}
template <typename T>
inline void inner(int             num_sc__,
                  wave_functions& bra__,
                  int             i0__,
                  int             m__,
                  wave_functions& ket__,
                  int             j0__,
                  int             n__,
                  dmatrix<T>&     result__,
                  int             irow0__,
                  int             jcol0__)
{
    inner(bra__, i0__, m__, ket__, j0__, n__, 0.0, result__, irow0__, jcol0__);
}

template <typename T>
inline void inner(int             num_sc__,
                  Wave_functions& bra__,
                  int             i0__,
                  int             m__,
                  Wave_functions& ket__,
                  int             j0__,
                  int             n__,
                  dmatrix<T>&     result__,
                  int             irow0__,
                  int             jcol0__)
{
    double beta{0};
    for (int is = 0; is < num_sc__; is++) {
        inner(bra__.component(is), i0__, m__, ket__.component(is), j0__, n__, beta, result__, irow0__, jcol0__);
        beta = 1;
    }
}
