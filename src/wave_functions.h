#ifndef __WAVE_FUNCTIONS_H__
#define __WAVE_FUNCTIONS_H__

namespace sirius {

class Wave_functions
{
    private:
        
        /// Number of wave-functions.
        int num_wfs_;

        Gvec const& gvec_;
        
        /// MPI grid for wave-function storage.
        /** Assume that the 1st dimension is used to distribute wave-functions and 2nd to distribute G-vectors */
        MPI_grid const& mpi_grid_;

        /// Entire communicator.
        Communicator const& comm_;

        mdarray<double_complex, 2> psi_slab_;

        mdarray<double_complex, 2> psi_panel_;

        mdarray<double_complex, 1> send_recv_buf_;
        
        splindex<block> spl_num_wfs_;

        int num_gvec_loc_;

        int rank_;
        int rank_row_;
        int num_ranks_col_;

        block_data_descriptor gvec_slab_distr_;

    public:

        Wave_functions(int num_wfs__, Gvec const& gvec__, MPI_grid const& mpi_grid__)
            : num_wfs_(num_wfs__),
              gvec_(gvec__),
              mpi_grid_(mpi_grid__),
              comm_(mpi_grid_.communicator())
        {
            PROFILE();

            /* number of column ranks */
            num_ranks_col_ = mpi_grid_.communicator(1 << 0).size();

            spl_num_wfs_ = splindex<block>(num_wfs_, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());

            num_gvec_loc_ = gvec_.num_gvec(mpi_grid_.communicator().rank());

            psi_slab_ = mdarray<double_complex, 2>(num_gvec_loc_, num_wfs_);

            psi_panel_ = mdarray<double_complex, 2>(gvec_.num_gvec_fft(), spl_num_wfs_.local_size());
   
            send_recv_buf_ = mdarray<double_complex, 1>(gvec_.num_gvec_fft() * spl_num_wfs_.local_size());
            
            /* flat rank id */
            rank_ = comm_.rank();
            /* row rank */
            rank_row_ = mpi_grid_.communicator(1 << 1).rank();

            /* store the number of G-vectors to be received by this rank */
            gvec_slab_distr_ = block_data_descriptor(num_ranks_col_);
            for (int i = 0; i < num_ranks_col_; i++)
            {
                gvec_slab_distr_.counts[i] = gvec_.num_gvec(rank_row_ * num_ranks_col_ + i);
            }
            gvec_slab_distr_.calc_offsets();
        
            assert(gvec_slab_distr_.offsets[num_ranks_col_ - 1] + gvec_slab_distr_.counts[num_ranks_col_ - 1] == gvec__.num_gvec_fft());
        }

        ~Wave_functions()
        {
        }

        void slab_to_panel(int idx0__, int n__)
        {
            PROFILE();

            Timer t("slab_to_panel");

            /* this is how n wave-functions will be distributed between panels */
            splindex<block> spl_n(n__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());
            /* local number of columns */
            int n_loc = spl_n.local_size();

            /* send parts of slab
             * +---+---+--+
             * |   |   |  |  <- irow = 0
             * +---+---+--+
             * |   |   |  |
             * ............
             * ranks in flat and 2D grid are related as: rank = irow * ncol + icol */
            for (int icol = 0; icol < num_ranks_col_; icol++)
            {
                int dest_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
                comm_.isend(&psi_slab_(0, idx0__ + spl_n.global_offset(icol)),
                            num_gvec_loc_ * spl_n.local_size(icol),
                            dest_rank, rank_ % num_ranks_col_);
            }
            
            /* receive parts of panel
             *                 n_loc
             *                 +---+  
             *                 |   |
             * gvec_slab_distr +---+
             *                 |   | 
             *                 +---+ */
            if (num_ranks_col_ > 1)
            {
                for (int i = 0; i < num_ranks_col_; i++)
                {
                    int src_rank = rank_row_ * num_ranks_col_ + i;
                    comm_.recv(&send_recv_buf_(gvec_slab_distr_.offsets[i] * n_loc), gvec_slab_distr_.counts[i] * n_loc, src_rank, i);
                }
                
                /* reorder received blocks to make G-vector index continuous */
                #pragma omp parallel for
                for (int i = 0; i < n_loc; i++)
                {
                    for (int j = 0; j < num_ranks_col_; j++)
                    {
                        std::memcpy(&psi_panel_(gvec_slab_distr_.offsets[j], i),
                                    &send_recv_buf_(gvec_slab_distr_.offsets[j] * n_loc + gvec_slab_distr_.counts[j] * i),
                                    gvec_slab_distr_.counts[j] * sizeof(double_complex));
                    }
                }
            }
            else
            {
                int src_rank = rank_row_ * num_ranks_col_;
                comm_.recv(&psi_panel_(0, 0), gvec_slab_distr_.counts[0] * n_loc, src_rank, 0);
            }
        }

        void panel_to_slab(int idx0__, int n__)
        {
            PROFILE();

            Timer t("panel_to_slab");
        
            /* this is how n wave-functions are distributed between panels */
            splindex<block> spl_n(n__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());
            /* local number of columns */
            int n_loc = spl_n.local_size();
            
            if (num_ranks_col_ > 1)
            {
                /* reorder sending blocks */
                #pragma omp parallel for
                for (int i = 0; i < n_loc; i++)
                {
                    for (int j = 0; j < num_ranks_col_; j++)
                    {
                        std::memcpy(&send_recv_buf_(gvec_slab_distr_.offsets[j] * n_loc + gvec_slab_distr_.counts[j] * i),
                                    &psi_panel_(gvec_slab_distr_.offsets[j], i),
                                    gvec_slab_distr_.counts[j] * sizeof(double_complex));
                    }
                }
        
                for (int i = 0; i < num_ranks_col_; i++)
                {
                    int dest_rank = rank_row_ * num_ranks_col_ + i;
                    comm_.isend(&send_recv_buf_(gvec_slab_distr_.offsets[i] * n_loc), gvec_slab_distr_.counts[i] * n_loc, dest_rank, i);
                }
            }
            else
            {
                int dest_rank = rank_row_ * num_ranks_col_;
                comm_.isend(&psi_panel_(0, 0), gvec_slab_distr_.counts[0] * n_loc, dest_rank, 0);
            }
            
            for (int icol = 0; icol < num_ranks_col_; icol++)
            {
                int src_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
                comm_.recv(&psi_slab_(0, idx0__ + spl_n.global_offset(icol)),
                           num_gvec_loc_ * spl_n.local_size(icol),
                           src_rank, rank_ % num_ranks_col_);
            }
        }

        inline double_complex& operator()(int igloc__, int i__)
        {
            return psi_slab_(igloc__, i__);
        }

        inline double_complex& panel(int igloc__, int i__)
        {
            return psi_panel_(igloc__, i__);
        }

        inline int num_gvec_loc() const
        {
            return num_gvec_loc_;
        }

        inline Gvec const& gvec() const
        {
            return gvec_;
        }
};

};

#endif
