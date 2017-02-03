/*
 * Operator.h
 *
 *  Created on: Feb 2, 2017
 *      Author: isivkov
 */

#ifndef __OPERATOR_H__
#define __OPERATOR_H__

#include "../simulation_context.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../unit_cell.h"
#include "../non_local_operator.h"

namespace sirius
{

namespace test
{

template <typename T>
class Non_local_operator_ext
{
    protected:
        device_t pu_;

        /// Non-local operator matrix.
        mdarray<T, 2> *op_;

        Non_local_operator<T> *non_local_op_;

        mdarray<T, 1> work_;

        bool is_null_{false};

        Unit_cell *unit_cell_;

        Non_local_operator_ext& operator=(Non_local_operator_ext const& src) = delete;
        Non_local_operator_ext(Non_local_operator_ext const& src) = delete;

    public:
        Non_local_operator_ext(Non_local_operator<T> non_local_op__, Unit_cell *unit_cell__)
        : pu_(non_local_op__.processing_unit()),
          unit_cell_(unit_cell__),
          op_(&non_local_op__.operator_matrix_unsafe())
        {
            //PROFILE("sirius::Non_local_operator::Non_local_operator");
        }

        ~Non_local_operator_ext()
        {
        }



        inline T operator()(int xi1__, int xi2__, int ia__)
        {
            return (*non_local_op_)(xi1__, xi2__, ia__);
        }

        inline T operator()(int xi1__, int xi2__, int ispn__, int ia__)
        {
            return (*non_local_op_)(xi1__, xi2__, ispn__, ia__);
        }

        inline size_t operator_raw_size()
        {
            return op_->size();
        }



        inline void apply_op_to_beta_phi(int ispn__,
                                    Beta_projectors::beta_chunk_t& chunk__,
                                    mdarray<double_complex, 2>& ket__,
                                    mdarray<double_complex, 2>& result__,
                                    int n__);

        inline void apply_beta_phi_to_beta_phi(int ispn__,
                                     Beta_projectors::beta_chunk_t& chunk__,
                                     mdarray<double_complex, 2>& bra__,
                                     mdarray<double_complex, 2>& ket__,
                                     mdarray<double_complex, 2>& result__,
                                     int n__);
};



template<>
inline void Non_local_operator_ext<double_complex>::apply_op_to_beta_phi(int ispn__,
                                                                Beta_projectors::beta_chunk_t& chunk__,
                                                                mdarray<double_complex, 2>& beta_phi_ket__,
                                                                mdarray<double_complex, 2>& result__,
                                                                int n__)
{
    if (beta_phi_ket__.size(0) != result__.size(0))
    {
        TERMINATE("ERROR from tes::Non_local_operator<double_complex>::apply_op_to_ket :  result__ and ket__ must have the same 'nbeta' leading dimmensions");
    }

    if (static_cast<size_t>(chunk__.num_beta_ * n__) > result__.size() )
    {
        TERMINATE("ERROR from tes::Non_local_operator<double_complex>::apply_op_to_ket :  result__ has wrong dimensions");
    }

    if (pu_ == CPU)
    {
        #pragma omp parallel for
        for (int i = 0; i < chunk__.num_atoms_; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = chunk__.desc_(0, i);
            int offs = chunk__.desc_(1, i);
            int ia = chunk__.desc_(3, i);

            /* compute |work> = O * |ket> */
            linalg<CPU>::gemm(0, 0, nbf, n__, nbf,
                              op_->at<CPU>(non_local_op_->packed_mtrx_offset(ia), ispn__), nbf,
                              beta_phi_ket__.at<CPU>(offs, 0), nbeta,
                              result__.at<CPU>(offs), nbeta);

        }
    }
}




template<>
inline void Non_local_operator_ext<double_complex>::apply_beta_phi_to_beta_phi(int ispn__,
                                                                     Beta_projectors::beta_chunk_t& chunk__,
                                                                     mdarray<double_complex, 2>& beta_phi_bra__,
                                                                     mdarray<double_complex, 2>& beta_phi_ket__,
                                                                     mdarray<double_complex, 2>& result__,
                                                                     int n__)
{
    if (beta_phi_ket__.size(0) != beta_phi_bra__.size(0))
    {
        TERMINATE("ERROR from tes::Non_local_operator<double_complex>::apply_ket_to_bra :  ket__ and bra__ have wrong dimensions");
    }

    if ( op_.size() != result__.size() )
    {
        TERMINATE("ERROR from tes::Non_local_operator<double_complex>::apply_ket_to_bra :  result__ must have the same size as operator");
    }

    #pragma omp parallel for
    for (int i = 0; i < chunk__.num_atoms_; i++)
    {
        /* number of beta functions for a given atom */
        int nbf = chunk__.desc_(0, i);
        int offs = chunk__.desc_(1, i);
        int ia = chunk__.desc_(3, i);

        /* compute |result> = |op_ket> * <bra| where |op_ket> = O*|ket> or |ket>*O   */
        linalg<CPU>::gemm(0, 2, nbf, nbf, n__,
                          beta_phi_ket__.at<CPU>(offs, 0), nbeta,
                          beta_phi_bra__.at<CPU>(offs, 0), nbeta,
                          result__.at<CPU>(non_local_op_->packed_mtrx_offset(ia), ispn__), nbf);

    }
}









}

}


#endif /* SRC_OPERATOR_OPERATOR_H_ */
