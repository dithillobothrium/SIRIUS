#include "wave_functions.h"

namespace sirius {

void Wave_functions<false>::swap_forward(int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_forward");

    /* this is how n wave-functions will be distributed between columns */
    spl_n_ = splindex<block>(n__, comm_col_.size(), comm_col_.rank());

    /* trivial case */
    if (comm_col_.size() == 1)
    {
        wf_coeffs_swapped_ = mdarray<double_complex, 2>(&wf_coeffs_(0, idx0__), num_gvec_loc_, n__);
        return;
    }

    /* local number of columns */
    int n_loc = spl_n_.local_size();

    block_data_descriptor sd(comm_col_.size());
    block_data_descriptor rd(comm_col_.size());

    for (int j = 0; j < comm_col_.size(); j++)
    {
        sd.counts[j] = spl_n_.local_size(j) * gvec_slab_pile_.counts[comm_col_.rank()];
        rd.counts[j] = spl_n_.local_size(comm_col_.rank()) * gvec_slab_pile_.counts[j];
    }

    sd.calc_offsets();
    rd.calc_offsets();

    comm_col_.alltoall(&wf_coeffs_(0, idx0__), &sd.counts[0], &sd.offsets[0],
                       &send_recv_buf_[0], &rd.counts[0], &rd.offsets[0]);
                      
    /* reorder recieved blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++)
    {
        for (int j = 0; j < comm_col_.size(); j++)
        {
            int offset = gvec_slab_pile_.offsets[j];
            int count = gvec_slab_pile_.counts[j];
            std::memcpy(&wf_coeffs_swapped_(offset, i), &send_recv_buf_[offset * n_loc + count * i], count * sizeof(double_complex));
        }
    }
}

void Wave_functions<false>::swap_backward(int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_backward");

    if (comm_col_.size() == 1) return;

    /* this is how n wave-functions are distributed between column ranks */
    splindex<block> spl_n(n__, comm_col_.size(), comm_col_.rank());
    /* local number of columns */
    int n_loc = spl_n.local_size();

    /* reorder sending blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++)
    {
        for (int j = 0; j < comm_col_.size(); j++)
        {
            int offset = gvec_slab_pile_.offsets[j];
            int count = gvec_slab_pile_.counts[j];
            std::memcpy(&send_recv_buf_[offset * n_loc + count * i], &wf_coeffs_swapped_(offset, i), count * sizeof(double_complex));
        }
    }

    block_data_descriptor sd(comm_col_.size());
    block_data_descriptor rd(comm_col_.size());

    for (int j = 0; j < comm_col_.size(); j++)
    {
        sd.counts[j] = spl_n.local_size(comm_col_.rank()) * gvec_slab_pile_.counts[j];
        rd.counts[j] = spl_n.local_size(j) * gvec_slab_pile_.counts[comm_col_.rank()];
    }

    sd.calc_offsets();
    rd.calc_offsets();

    comm_col_.alltoall(&send_recv_buf_[0], &sd.counts[0], &sd.offsets[0],
                       &wf_coeffs_(0, idx0__), &rd.counts[0], &rd.offsets[0]);
}

void Wave_functions<false>::inner(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                                  mdarray<double_complex, 2>& result__, int irow__, int icol__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::inner");

    assert(num_gvec_loc() == ket__.num_gvec_loc());

    /* single rank, CPU: store result directly in the output matrix */
    if (mpi_grid_.size() == 1 && pu_ == CPU)
    {
        linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
                          &ket__(0, j0__), num_gvec_loc(), &result__(irow__, icol__), result__.ld());
    }
    else /* parallel or GPU */
    {
        /* block size to split <ket| states */
        int bs = 256;
        /* number of blocks of <ket| states */
        int nb = m__ / bs + std::min(m__ % bs, 1);
        /* reallocate buffer if necessary */
        if (static_cast<size_t>(2 * bs * n__) > inner_prod_buf_.size())
        {
            inner_prod_buf_ = mdarray<double_complex, 1>(2 * bs * n__);
            #ifdef __GPU
            if (pu_ == GPU) inner_prod_buf_.allocate_on_device();
            #endif
        }

        double t0 = -omp_get_wtime();
        
        /* state of the buffers:
         * state = 0: buffer is free
         * state = 1: buffer stores result of local zgemm */
        int buf_state[] = {0, 0};

        omp_set_nested(1);
        int nt = omp_get_max_threads();
        if (nt < 2) TERMINATE("minimum two threads are required");
        #pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0)
            {
                /* compute threads */
                omp_set_num_threads(nt - 1);
                for (int ib = 0; ib < nb; ib++)
                {
                    int is = ib % 2;
                    /* current block size */
                    int ibs = ((ib + 1) * bs > m__) ? m__ - ib * bs : bs;
                    /* wait for the release of the buffer */
                    while (true)
                    {
                        int st;
                        #pragma omp critical
                        st = buf_state[is];
                        if (st == 0) break;
                    }
                    switch (pu_)
                    {
                        case CPU:
                        {
                            /* perform local matrix multiplication */
                            linalg<CPU>::gemm(2, 0, ibs, n__, num_gvec_loc(), &wf_coeffs_(0, i0__ + ib * bs), num_gvec_loc(),
                                              &ket__(0, j0__), num_gvec_loc(), &inner_prod_buf_[is * bs * n__], ibs);
                            break;
                        }
                        case GPU:
                        {
                            #ifdef __GPU
                            linalg<GPU>::gemm(2, 0, ibs, n__, num_gvec_loc(), wf_coeffs_.at<GPU>(0, i0__ + ib * bs), num_gvec_loc(),
                                              ket__.wf_coeffs_.at<GPU>(0, j0__), num_gvec_loc(), inner_prod_buf_.at<GPU>(is * bs * n__), ibs);
                            acc::copyout(inner_prod_buf_.at<CPU>(is * bs * n__), inner_prod_buf_.at<GPU>(is * bs * n__), ibs * n__);
                            #else
                            TERMINATE_NO_GPU
                            #endif
                        }
                    }
                    /* lock the buffer */
                    #pragma omp critical
                    buf_state[is] = 1;
                }
            }
            else
            {
                for (int ib = 0; ib < nb; ib++)
                {
                    int is = ib % 2;
                    /* current block size */
                    int ibs = ((ib + 1) * bs > m__) ? m__ - ib * bs : bs;
                    /* wait for data */
                    while (true)
                    {
                        int st;
                        #pragma omp critical
                        st = buf_state[is];
                        if (st == 1) break;
                    }
                    /* sum from all ranks */
                    mpi_grid_.communicator().allreduce(&inner_prod_buf_[is * bs * n__], ibs * n__);
                    /* save result */
                    for (int i = 0; i < n__; i++)
                    {
                        std::memcpy(&result__(irow__ + ib * bs, icol__ + i),
                                    &inner_prod_buf_[is * bs * n__ + i * ibs],
                                    ibs * sizeof(double_complex));
                    }
                    /* release the buffer */
                    #pragma omp critical
                    buf_state[is] = 0;
                }
            }
        }
        omp_set_nested(0);

        t0 += omp_get_wtime();
        PRINT("[wf:inner] zgemm performance: %f GFlops", 8e-9 * m__ * n__ * gvec_.num_gvec() / t0);



            






        ///* reallocate buffer if necessary */
        //if (static_cast<size_t>(m__ * n__) > inner_prod_buf_.size())
        //{
        //    inner_prod_buf_ = mdarray<double_complex, 1>(m__ * n__);
        //    #ifdef __GPU
        //    if (pu_ == GPU) inner_prod_buf_.allocate_on_device();
        //    #endif
        //}
        //switch (pu_)
        //{
        //    case CPU:
        //    {
        //        linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
        //                          &ket__(0, j0__), num_gvec_loc(), &inner_prod_buf_[0], m__);
        //        break;
        //    }
        //    case GPU:
        //    {
        //        #ifdef __GPU
        //        linalg<GPU>::gemm(2, 0, m__, n__, num_gvec_loc(), wf_coeffs_.at<GPU>(0, i0__), num_gvec_loc(),
        //                          ket__.wf_coeffs_.at<GPU>(0, j0__), num_gvec_loc(), inner_prod_buf_.at<GPU>(), m__);
        //        inner_prod_buf_.copy_to_host(m__ * n__);
        //        #else
        //        TERMINATE_NO_GPU
        //        #endif
        //        break;
        //    }
        //}

        //if (mpi_grid_.size() > 1) mpi_grid_.communicator().allreduce(&inner_prod_buf_[0], m__ * n__);

        //for (int i = 0; i < n__; i++)
        //    std::memcpy(&result__(irow__, icol__ + i), &inner_prod_buf_[i * m__], m__ * sizeof(double_complex));
    }
}

};
