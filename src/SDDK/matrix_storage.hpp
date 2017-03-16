// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file matrix_storage.hpp
 *
 *  \brief Contains definition and implementaiton of matrix_storage class.
 */

#ifndef __MATRIX_STORAGE_HPP__
#define __MATRIX_STORAGE_HPP__

namespace sddk {

enum class matrix_storage_t
{
    slab,
    block_cyclic
};

/// Class declaration.
template <typename T, matrix_storage_t kind>
class matrix_storage;

/// Specialization of matrix storage class for slab data distribution.
/** \tparam T data type */
template <typename T>
class matrix_storage<T, matrix_storage_t::slab>
{
  private:
    /// Type of processing unit used.
    device_t pu_;

    /// Local number of rows.
    int num_rows_loc_{0};

    /// Total number of columns.
    int num_cols_;

    /// Primary storage of matrix.
    mdarray<T, 2> prime_;

    /// Auxiliary matrix storage.
    /** This distribution is used by the FFT driver */
    mdarray<T, 2> extra_;

    /// Raw buffer for the extra storage.
    mdarray<T, 1> extra_buf_;

    /// Raw send-recieve buffer.
    mdarray<T, 1> send_recv_buf_;

    /// Column distribution in auxiliary matrix.
    splindex<block> spl_num_col_;

  public:
    matrix_storage(int num_rows_loc__, int num_cols__, device_t pu__)
        : pu_(pu__)
        , num_rows_loc_(num_rows_loc__)
        , num_cols_(num_cols__)
    {
        PROFILE("sddk::matrix_storage::matrix_storage");
        /* primary storage of PW wave functions: slabs */
        prime_ = mdarray<T, 2>(num_rows_loc_, num_cols_, memory_t::host, "matrix_storage.prime_");
    }

    matrix_storage(T* ptr__, int num_rows_loc__, int num_cols__, device_t pu__)
        : pu_(pu__)
        , num_rows_loc_(num_rows_loc__)
        , num_cols_(num_cols__)
    {
        PROFILE("sddk::matrix_storage::matrix_storage");
        /* primary storage of PW wave functions: slabs */
        prime_ = mdarray<T, 2>(ptr__, num_rows_loc_, num_cols_, "matrix_storage.prime_");
    }

    /// Set the dimensions of the extra matrix storage.
    /** \param [in] num_rows Number of rows in the extra matrix storage.
     *  \param [in] comm     Communicator used to distribute columns of the matrix.
     *  \param [in] n        Number of matrix columns to distribute.
     *  \param [in] idx0     Starting column of the matrix.
     *
     *  \image html matrix_storage.png "Redistribution of wave-functions between MPI ranks"
     *
     *  In general, the extra storage is created in the CPU pointer. The only exception is when the data
     *  distribution doesn't change (no swapping between comm_col ranks is performed) – then the extra
     *  storage will mirror the prime storage (both on CPU and GPU).
     */
    inline void set_num_extra(int num_rows__, Communicator const& comm_col__, int n__, int idx0__ = 0)
    {
        /* this is how n columns of the matrix will be distributed between columns of the MPI grid */
        spl_num_col_ = splindex<block>(n__, comm_col__.size(), comm_col__.rank());

        /* trivial case */
        if (comm_col__.size() == 1) {
            assert(num_rows_loc_ == num_rows__);
            if (pu_ == GPU && prime_.on_device()) {
                extra_ = mdarray<T, 2>(prime_.template at<CPU>(0, idx0__), prime_.template at<GPU>(0, idx0__),
                                       num_rows__, n__, "matrix_storage.extra_");
            } else {
                extra_ = mdarray<T, 2>(prime_.template at<CPU>(0, idx0__), num_rows__, n__, "matrix_storage.extra_");
            }
        } else {
            /* maximum local number of matrix columns */
            int max_n_loc = splindex_base<int>::block_size(n__, comm_col__.size());
            /* upper limit for the size of swapped extra matrix */
            size_t sz = num_rows__ * max_n_loc;
            /* reallocate buffers if necessary */
            if (extra_buf_.size() < sz) {
                extra_buf_     = mdarray<T, 1>(sz);
                send_recv_buf_ = mdarray<T, 1>(sz, memory_t::host, "matrix_storage.send_recv_buf_");
            }
            extra_ = mdarray<T, 2>(extra_buf_.template at<CPU>(), num_rows__, max_n_loc, "matrix_storage.extra_");
        }
    }
    
    /// Remap data from prime to extra storage.
    /** Prime storage is expected on the CPU (for the MPI a2a communication). If prime storage was also allocated on
     *  the device, extra storage will be copied to device as well. 
     *  \param [in] row_distr Distribution of rows of prime matrix in the extra matrix storage. 
     *  \param [in] comm      Communicator used to distribute columns of the matrix.
     *  \param [in] n         Number of matrix columns to distribute.
     *  \param [in] idx0      Starting column of the matrix.
     *  \tparam mem_type      Indicates which type of memory is remapped to exrta storage.
     */
    template <memory_t mem_type = memory_t::host>
    inline void remap_forward(block_data_descriptor const& row_distr__,
                              Communicator const& comm_col__,
                              int n__,
                              int idx0__ = 0)
    {
        PROFILE("sddk::matrix_storage::remap_forward");

        set_num_extra(row_distr__.counts.back() + row_distr__.offsets.back(), comm_col__, n__, idx0__);

        /* trivial case when extra storage mirrors the prime storage */
        if (comm_col__.size() == 1) {
            return;
        }

        /* local number of columns */
        int n_loc = spl_num_col_.local_size();

        /* send and recieve dimensions */
        block_data_descriptor sd(comm_col__.size()), rd(comm_col__.size());
        for (int j = 0; j < comm_col__.size(); j++) {
            sd.counts[j] = spl_num_col_.local_size(j) * row_distr__.counts[comm_col__.rank()];
            rd.counts[j] = spl_num_col_.local_size(comm_col__.rank()) * row_distr__.counts[j];
        }
        sd.calc_offsets();
        rd.calc_offsets();

        T* send_buf = (num_rows_loc_ == 0) ? nullptr : prime_.template at<CPU>(0, idx0__);

        comm_col__.alltoall(send_buf, sd.counts.data(), sd.offsets.data(), send_recv_buf_.template at<CPU>(),
                            rd.counts.data(), rd.offsets.data());

        /* reorder recieved blocks */
        #pragma omp parallel for
        for (int i = 0; i < n_loc; i++) {
            for (int j = 0; j < comm_col__.size(); j++) {
                int offset = row_distr__.offsets[j];
                int count  = row_distr__.counts[j];
                if (count) {
                    std::memcpy(&extra_(offset, i), &send_recv_buf_[offset * n_loc + count * i], count * sizeof(T));
                }
            }
        }
        /* if prime storage was on device, copy extra storage to the device as well */
        if ((mem_type & memory_t::device) != memory_t::none) {
            extra_.allocate(memory_t::device);
            extra_.template copy<memory_t::host, memory_t::device>();
        }
    }

    /// Remap data from extra to prime storage.
    /** \param [in] row_distr Distribution of rows of prime matrix in the extra matrix storage. 
     *  \param [in] comm      Communicator used to distribute columns of the matrix.
     *  \param [in] n         Number of matrix columns to collect.
     *  \param [in] idx0      Starting column of the matrix.
     *  \tparam mem_type      Indicates which type of memory is remapped back to prime storage.
     */
    template <memory_t mem_type = memory_t::host>
    inline void remap_backward(block_data_descriptor const& row_distr__,
                               Communicator const& comm_col__,
                               int n__,
                               int idx0__ = 0)
    {
        PROFILE("sddk::matrix_storage::remap_backward");

        /* trivial case when extra storage mirrors the prime storage */
        if (comm_col__.size() == 1) {
            return;
        }
        
        /* move data to host to perfoem MPI a2a communication */
        if ((mem_type & memory_t::device) != memory_t::none) {
            extra_.template copy<memory_t::device, memory_t::host>();
        }

        assert(n__ == spl_num_col_.global_index_size());

        /* local number of columns */
        int n_loc = spl_num_col_.local_size();

        /* reorder sending blocks */
        #pragma omp parallel for
        for (int i = 0; i < n_loc; i++) {
            for (int j = 0; j < comm_col__.size(); j++) {
                int offset = row_distr__.offsets[j];
                int count  = row_distr__.counts[j];
                if (count) {
                    std::memcpy(&send_recv_buf_[offset * n_loc + count * i], &extra_(offset, i), count * sizeof(T));
                }
            }
        }

        /* send and recieve dimensions */
        block_data_descriptor sd(comm_col__.size()), rd(comm_col__.size());
        for (int j = 0; j < comm_col__.size(); j++) {
            sd.counts[j] = spl_num_col_.local_size(comm_col__.rank()) * row_distr__.counts[j];
            rd.counts[j] = spl_num_col_.local_size(j) * row_distr__.counts[comm_col__.rank()];
        }
        sd.calc_offsets();
        rd.calc_offsets();

        T* recv_buf = (num_rows_loc_ == 0) ? nullptr : prime_.template at<CPU>(0, idx0__);

        comm_col__.alltoall(send_recv_buf_.template at<CPU>(), sd.counts.data(), sd.offsets.data(), recv_buf,
                            rd.counts.data(), rd.offsets.data());

        /* move data back to device */
        if ((mem_type & memory_t::device) != memory_t::none && num_rows_loc()) {
            prime_.template copy<memory_t::host, memory_t::device>(idx0__ * num_rows_loc(), n__ * num_rows_loc());
        }
    }

    inline void remap_from(dmatrix<T> const& mtrx__, int irow0__)
    {
        PROFILE("sddk::matrix_storage::remap_from");

        if (mtrx__.num_cols() != num_cols_) {
            TERMINATE("different number of columns");
        }

        auto& comm = mtrx__.blacs_grid().comm();

        /* cache cartesian ranks */
        mdarray<int, 2> cart_rank(mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().num_ranks_col());
        for (int i = 0; i < mtrx__.blacs_grid().num_ranks_col(); i++) {
            for (int j = 0; j < mtrx__.blacs_grid().num_ranks_row(); j++) {
                cart_rank(j, i) = mtrx__.blacs_grid().cart_rank(j, i);
            }
        }

        if (send_recv_buf_.size() < prime_.size()) {
            send_recv_buf_ = mdarray<T, 1>(prime_.size(), memory_t::host, "matrix_storage::send_recv_buf_");
        }

        block_data_descriptor rd(comm.size());
        rd.counts[comm.rank()] = num_rows_loc_;
        comm.allgather(rd.counts.data(), comm.rank(), 1);
        rd.calc_offsets();

        block_data_descriptor sd(comm.size());

        /* global index of column */
        int j0 = 0;
        /* actual number of columns in the submatrix */
        int ncol = num_cols_;

        splindex<block_cyclic> spl_col_begin(j0, mtrx__.num_ranks_col(), mtrx__.rank_col(), mtrx__.bs_col());
        splindex<block_cyclic> spl_col_end(j0 + ncol, mtrx__.num_ranks_col(), mtrx__.rank_col(), mtrx__.bs_col());

        int local_size_col = spl_col_end.local_size() - spl_col_begin.local_size();

        for (int rank_row = 0; rank_row < comm.size(); rank_row++) {
            if (!rd.counts[rank_row]) {
                continue;
            }
            /* global index of column */
            int i0 = rd.offsets[rank_row];
            /* actual number of rows in the submatrix */
            int nrow = rd.counts[rank_row];

            assert(nrow != 0);

            splindex<block_cyclic> spl_row_begin(irow0__ + i0, mtrx__.num_ranks_row(), mtrx__.rank_row(),
                                                 mtrx__.bs_row());
            splindex<block_cyclic> spl_row_end(irow0__ + i0 + nrow, mtrx__.num_ranks_row(), mtrx__.rank_row(),
                                               mtrx__.bs_row());

            int local_size_row = spl_row_end.local_size() - spl_row_begin.local_size();

            mdarray<T, 1> buf(local_size_row * local_size_col);

            /* fetch elements of sub-matrix matrix */
            if (local_size_row) {
                for (int j = 0; j < local_size_col; j++) {
                    std::memcpy(&buf[local_size_row * j],
                                &mtrx__(spl_row_begin.local_size(), spl_col_begin.local_size() + j),
                                local_size_row * sizeof(T));
                }
            }

            sd.counts[comm.rank()] = local_size_row * local_size_col;
            comm.allgather(sd.counts.data(), comm.rank(), 1);
            sd.calc_offsets();

            /* collect buffers submatrix */
            T* send_buf = (buf.size() == 0) ? nullptr : &buf[0];
            T* recv_buf = (send_recv_buf_.size() == 0) ? nullptr : &send_recv_buf_[0];
            comm.gather(send_buf, recv_buf, sd.counts.data(), sd.offsets.data(), rank_row);

            if (comm.rank() == rank_row) {
                /* unpack data */
                std::vector<int> counts(comm.size(), 0);
                for (int jcol = 0; jcol < ncol; jcol++) {
                    auto pos_jcol = mtrx__.spl_col().location(j0 + jcol);
                    for (int irow = 0; irow < nrow; irow++) {
                        auto pos_irow = mtrx__.spl_row().location(irow0__ + i0 + irow);
                        int rank      = cart_rank(pos_irow.rank, pos_jcol.rank);

                        prime_(irow, jcol) = send_recv_buf_[sd.offsets[rank] + counts[rank]];
                        counts[rank]++;
                    }
                }
                for (int rank = 0; rank < comm.size(); rank++) {
                    assert(sd.counts[rank] == counts[rank]);
                }
            }
        }
    }

    inline T& prime(int irow__, int jcol__)
    {
        return prime_(irow__, jcol__);
    }

    inline T const& prime(int irow__, int jcol__) const
    {
        return prime_(irow__, jcol__);
    }

    mdarray<T, 2>& prime()
    {
        return prime_;
    }

    mdarray<T, 2> const& prime() const
    {
        return prime_;
    }

    mdarray<T, 2>& extra()
    {
        return extra_;
    }

    mdarray<T, 2> const& extra() const
    {
        return extra_;
    }
    
    template <memory_t mem_type>
    inline void zero(int i0__, int n__)
    {
        prime_.zero<mem_type>(i0__ * num_rows_loc(), n__ * num_rows_loc());
    }

    /// Local number of rows in prime matrix.
    inline int num_rows_loc() const
    {
        return num_rows_loc_;
    }

    inline splindex<block> const& spl_num_col() const
    {
        return spl_num_col_;
    }

    #ifdef __GPU
    /// Allocate prime storage on device.
    void allocate_on_device()
    {
        prime_.allocate(memory_t::device);
    }
    
    /// Deallocate storage on device.
    void deallocate_on_device()
    {
        prime_.deallocate_on_device();
        extra_.deallocate_on_device();
    }
    
    /// Copy prime storage to device memory.
    void copy_to_device(int i0__, int n__)
    {
        if (num_rows_loc()) {
            acc::copyin(prime_.template at<GPU>(0, i0__), prime_.template at<CPU>(0, i0__), n__ * num_rows_loc());
        }
    }

    /// Copy prime storage to host memory.
    void copy_to_host(int i0__, int n__)
    {
        if (num_rows_loc()) {
            acc::copyout(prime_.template at<CPU>(0, i0__), prime_.template at<GPU>(0, i0__), n__ * num_rows_loc());
        }
    }
    #endif
};

//== template <typename T>
//== class matrix_storage<T, matrix_storage_t::block_cyclic>
//== {
//==     private:
//==
//==         int num_rows_;
//==
//==         int num_cols_;
//==
//==         int bs_;
//==
//==         BLACS_grid const& blacs_grid_;
//==
//==         BLACS_grid const& blacs_grid_slice_;
//==
//==         dmatrix<T> prime_;
//==
//==         dmatrix<T> extra_;
//==
//==         /// Raw buffer for the extra storage.
//==         mdarray<T, 1> extra_buf_;
//==
//==         /// Column distribution in auxiliary matrix.
//==         splindex<block> spl_num_col_;
//==
//==     public:
//==
//==         matrix_storage(int num_rows__, int num_cols__, int bs__, BLACS_grid const& blacs_grid__, BLACS_grid const&
//blacs_grid_slice__)
//==             : num_rows_(num_rows__),
//==               num_cols_(num_cols__),
//==               bs_(bs__),
//==               blacs_grid_(blacs_grid__),
//==               blacs_grid_slice_(blacs_grid_slice__)
//==         {
//==             assert(blacs_grid_slice__.num_ranks_row() == 1);
//==
//==             prime_ = dmatrix<T>(num_rows_, num_cols_, blacs_grid_, bs_, bs_);
//==         }
//==
//==         /// Set extra-storage matrix.
//==         void set_num_extra(int n__)
//==         {
//==             /* this is how n wave-functions will be distributed between panels */
//==             spl_num_col_ = splindex<block>(n__, blacs_grid_slice_.num_ranks_col(), blacs_grid_slice_.rank_col());
//==
//==             int bs = splindex_base<int>::block_size(n__, blacs_grid_slice_.num_ranks_col());
//==             if (blacs_grid_.comm().size() > 1) {
//==                 size_t sz = num_rows_ * bs;
//==                 if (extra_buf_.size() < sz) {
//==                     extra_buf_ = mdarray<T, 1>(sz);
//==                 }
//==                 extra_ = dmatrix<T>(&extra_buf_[0], num_rows_, n__, blacs_grid_slice_, 1, bs);
//==             } else {
//==                 extra_ = dmatrix<T>(prime_.template at<CPU>(), num_rows_, n__, blacs_grid_slice_, 1, bs);
//==             }
//==         }
//==
//==         void remap_forward(int idx0__, int n__)
//==         {
//==             PROFILE("sddk::matrix_storage::remap_forward");
//==             set_num_extra(n__);
//==             if (blacs_grid_.comm().size() > 1) {
//==                 #ifdef __SCALAPACK
//==                 linalg<CPU>::gemr2d(num_rows_, n__, prime_, 0, idx0__, extra_, 0, 0, blacs_grid_.context());
//==                 #else
//==                 TERMINATE_NO_SCALAPACK
//==                 #endif
//==             }
//==         }
//==
//==         void remap_backward(int idx0__, int n__)
//==         {
//==             PROFILE("sddk::matrix_storage::remap_backward");
//==             if (blacs_grid_.comm().size() > 1) {
//==                 #ifdef __SCALAPACK
//==                 linalg<CPU>::gemr2d(num_rows_, n__, extra_, 0, 0, prime_, 0, idx0__, blacs_grid_.context());
//==                 #else
//==                 TERMINATE_NO_SCALAPACK
//==                 #endif
//==             }
//==         }
//==
//==         dmatrix<double_complex>& prime()
//==         {
//==             return prime_;
//==         }
//==
//==         dmatrix<double_complex>& extra()
//==         {
//==             return extra_;
//==         }
//==
//==         inline splindex<block> const& spl_num_col() const
//==         {
//==             return spl_num_col_;
//==         }
//== };

} // namespace sddk

#endif // __MATRIX_STORAGE_HPP__
