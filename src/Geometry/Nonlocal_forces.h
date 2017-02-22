#ifndef __NONLOCAL_FORCES__
#define __NONLOCAL_FORCES__

#include <memory>

#include "../simulation_context.h"
#include "../k_point.h"
#include "Chunk_linalg.h"

namespace sirius
{


template <typename T>
class Nonlocal_forces
{
    protected:

        std::array<mdarray<T, 1>, 3> forces_matrix_;

        //-------------------------------------------
        /// offsets, indicating first elements of force matrix belonging to each atom
        //-------------------------------------------
        mdarray<int, 1> forces_mtrx_offset_;

        //-------------------------------------------
        /// external classes pointers
        //-------------------------------------------
        Simulation_context* ctx_;
        Unit_cell* unit_cell_;
        K_point* kpoint_;
        D_operator<T>* operator_D_;
        Q_operator<T>* operator_Q_;


        //-------------------------------------------
        /// temporary store mult result
        //-------------------------------------------
        mdarray<T, 2> beta_phi_D_;
        mdarray<T, 2> beta_phi_Q_;

        device_t pu_;

    public:
        Nonlocal_forces(Simulation_context* ctx__, K_point* kpoint__, D_operator<T>* operator_D__, Q_operator<T>* operator_Q__, device_t pu__):
        ctx_(ctx__),
        unit_cell_(&ctx__->unit_cell()),
        kpoint_(kpoint__),
        operator_D_(operator_D__),
        operator_Q_(operator_Q__),
        pu_(pu__)
        {
            // allocate forces matrix
            for(int comp: {0,1,2})
            {
                forces_matrix_[comp] = mdarray<T, 1>( operator_Q__->packed_mtrx_size());
                forces_matrix_[comp].zero();
            }

            // allocate forces matrix offsets similar to Q offsets
            forces_mtrx_offset_ = mdarray<int, 1>(unit_cell_->num_atoms());

            for(int ia = 0; ia < unit_cell_->num_atoms(); ia++)
            {
                forces_mtrx_offset_(ia) = operator_Q__->packed_mtrx_offset(ia);
            }

        }


        void calc_contribution_to_forces_matrix(int ispn__,
                                                const Beta_projectors::beta_chunk_t& chunk__,
                                                matrix<T>& bp_phi_chunk__,
                                                std::array<matrix<T>, 3>& bp_grad_phi_chunk__,
                                                int bands_offset__,
                                                int num_bands__);

        void calc_contribution_to_forces(mdarray<double,2>& forces__, const Beta_projectors::beta_chunk_t& chunk__);



};

template<typename T>
void Nonlocal_forces<T>::calc_contribution_to_forces_matrix(int ispn__,
                                                            const Beta_projectors::beta_chunk_t& chunk__,
                                                            matrix<T>& bp_phi_chunk__,
                                                            std::array<matrix<T>, 3>& bp_grad_phi_chunk__,
                                                            int bands_offset__,
                                                            int num_bands__)
{

    // allocate beta_phi_D_ / Q_
    if( static_cast<size_t>( (bands_offset__ + num_bands__) * chunk__.num_beta_) > beta_phi_D_.size() )
    {
        beta_phi_D_ = mdarray<T, 2>(chunk__.num_beta_, bands_offset__ + num_bands__);
    }

    if( static_cast<size_t>( (bands_offset__ + num_bands__) * chunk__.num_beta_) > beta_phi_Q_.size() )
    {
        beta_phi_Q_ = mdarray<T, 2>(chunk__.num_beta_, bands_offset__ + num_bands__);
    }


    // |bd> = D * |beta>
    if( operator_D_->processing_unit() == CPU)
    {
        mdarray<T, 1> op(operator_D_->operator_matrix_unsafe().template at<CPU>(0,ispn__),
                           operator_D_->operator_matrix_unsafe().size(0)  );

        test::Chunk_linalg<CPU,T>::apply_op_to_chunked_state(op,
                                                             operator_D_->packed_mtrx_offset(),
                                                             chunk__,
                                                             bp_phi_chunk__,
                                                             beta_phi_D_,
                                                             bands_offset__,
                                                             num_bands__);
    }

    // |bq> = Q * |beta>
    if( operator_Q_->processing_unit() == CPU)
    {
        mdarray<T, 1> op(operator_Q_->operator_matrix_unsafe().template at<CPU>(0,0),
                           operator_Q_->operator_matrix_unsafe().size(0)  );

        test::Chunk_linalg<CPU,T>::apply_op_to_chunked_state(op,
                                                             operator_Q_->packed_mtrx_offset(),
                                                             chunk__,
                                                             bp_phi_chunk__,
                                                             beta_phi_Q_,
                                                             bands_offset__,
                                                             num_bands__);
    }


//    for(int ibnd = bands_offset__; ibnd < bands_offset__ + num_bands__; ibnd++)
//    {
//        for(int ibeta = 0; ibeta < chunk__.num_beta_; ibeta++)
//        {
//            std::cout<<beta_phi_D_(ibeta, ibnd)<<" ";
//        }
//    }
    // |bd> += E(bnd) * |bq>
    #pragma omp parallel for
    for(int ibnd = bands_offset__; ibnd < bands_offset__ + num_bands__; ibnd++)
    {
        for(int ibeta = 0; ibeta < chunk__.num_beta_; ibeta++)
        {
            beta_phi_D_(ibeta, ibnd) = kpoint_->band_occupancy(ibnd + ispn__ * ctx_->num_fv_states()) *
                    ( beta_phi_D_(ibeta, ibnd) - kpoint_->band_energy(ibnd) * beta_phi_Q_(ibeta, ibnd) );
        }
    }


    // <bd| * |beta_grad>[0,1,2]
    for(int comp: {0,1,2})
    {
        if(pu_ == CPU)
        {
            test::Chunk_linalg<CPU,T>::chunked_states_product(0, 2, chunk__,
                                                              beta_phi_D_,
                                                              bp_grad_phi_chunk__[comp],
                                                              forces_matrix_[comp],
                                                              forces_mtrx_offset_,
                                                              bands_offset__,
                                                              num_bands__);
        }
    }
}


template<typename T>
void Nonlocal_forces<T>::calc_contribution_to_forces(mdarray<double,2>& forces__, const Beta_projectors::beta_chunk_t& chunk__)
{

    double factor = 2.0;

    for(int comp: {0,1,2})
    {
        //#pragma omp parallel for
        for (int i = 0; i < chunk__.num_atoms_; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = chunk__.desc_(0, i);
            int offs = chunk__.desc_(1, i);
            int ia = chunk__.desc_(3, i);

//            std::cout<<ia<<" "<<comp;
            for(int ibf = 0; ibf < nbf; ibf++ )
            {
//                for(int jbf = 0; jbf < nbf; jbf++ )
//                {
                    forces__(comp, ia) += forces_matrix_[comp](forces_mtrx_offset_(ia) + ibf * nbf + ibf).real();
//                }
            }
            //std::cout<<" "<<forces__(comp, ia)<<std::endl;

            forces__(comp, ia) *= factor * kpoint_->weight();


            for(int ibf = 0; ibf < nbf; ibf++ )
            {
                for(int jbf = 0; jbf < nbf; jbf++ )
                {
                    std::cout<<forces_matrix_[comp](forces_mtrx_offset_(ia) + ibf * nbf + jbf)<<" \t";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
    }

}
/*
 Atom    0    force =       0.0031518        0.0070574        0.0070574
 Atom    1    force =       0.0370748       -0.0107797       -0.0107797

Atom    0    force =       0.0362378        0.0364083        0.0364083
Atom    1    force =      -0.0025062       -0.0028173       -0.0028173

 etalon
Atom    0    force =      -0.0185157       -0.0250644       -0.0250644
Atom    1    force =      -0.0311765       -0.0316160       -0.0316160
 * */

}
#endif
