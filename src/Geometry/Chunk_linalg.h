/*
 * Operator.h
 *
 *  Created on: Feb 2, 2017
 *      Author: isivkov
 */

#ifndef __CHUNK_LINALG_H__
#define __CHUNK_LINALG_H__

#include "../simulation_context.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../unit_cell.h"
#include "../non_local_operator.h"

namespace sirius
{

namespace test
{

template <device_t pu, typename T>
class Chunk_linalg
{
    public:
        //---------------------------------------------------------
        /// apply operator, containing matrices for all atoms to a chunk of a state
        //---------------------------------------------------------
//        template<typename T>
//        static void apply_op_to_chunked_state(mdarray<T, 1>& operator__,
//                                              mdarray<int, 1>& packed_mtrx_offsets__,
//                                              Beta_projectors::beta_chunk_t& chunk__,
//                                              mdarray<T, 2>& ket__,
//                                              mdarray<T, 2>& result__,
//                                              int bands_offset__,
//                                              int num_bands__);

        static void apply_op_to_chunked_state(mdarray<T, 1>& operator__,
                                              const mdarray<int, 1>& packed_mtrx_offsets__,
                                              const Beta_projectors::beta_chunk_t& chunk__,
                                              mdarray<T, 2>& ket__,
                                              mdarray<T, 2>& result__,
                                              int bands_offset__,
                                              int num_bands__);




        static void chunked_states_product(int transpose_bra__,
                                           int transpose_ket__,
                                           const Beta_projectors::beta_chunk_t& chunk__,
                                           matrix<T>& bra__,
                                           matrix<T>& ket__,
                                           mdarray<T, 1>& operator__,
                                           const mdarray<int, 1>& packed_mtrx_offsets__,
                                           int bands_offset__,
                                           int num_bands__);

};




template<device_t pu>
class Chunk_linalg<pu, double_complex>
{
    public:

        static void apply_op_to_chunked_state(mdarray<double_complex, 1>& operator__,
                                              const mdarray<int, 1>& packed_mtrx_offsets__,
                                              const Beta_projectors::beta_chunk_t& chunk__,
                                              mdarray<double_complex, 2>& ket__,
                                              mdarray<double_complex, 2>& result__,
                                              int bands_offset__,
                                              int num_bands__)
        {
            //            if (ket__.size(0) != result__.size(0))
            //            {
            //                TERMINATE("ERROR from tes::Non_local_operator<double_complex>::apply_op_to_ket :  result__ and ket__ must have the same 'nbeta' leading dimmensions");
            //            }

            if (static_cast<size_t>(chunk__.num_beta_ * (bands_offset__ + num_bands__) ) > result__.size() ||
                    static_cast<size_t>(chunk__.num_beta_ * (bands_offset__ + num_bands__) ) > ket__.size()  )
            {
                TERMINATE("ERROR from tes::Non_local_operator<double_complex>::apply_op_to_ket :  result__ has too small size");
            }

            int nbeta = chunk__.num_beta_;

            if (pu == CPU)
            {
                #pragma omp parallel for
                for (int i = 0; i < chunk__.num_atoms_; i++)
                {
                    /* number of beta functions for a given atom */
                    int nbf = chunk__.desc_(0, i);
                    int offs = chunk__.desc_(1, i);
                    int ia = chunk__.desc_(3, i);

                    /* compute |work> = O * |ket> */
//                    linalg<pu>::gemm(0, 0, nbf, num_bands__, nbf,
//                                     operator__.at<CPU>(packed_mtrx_offsets__(ia)), nbf,
//                                     ket__.at<CPU>(offs, bands_offset__), nbeta,
//                                     result__.at<CPU>(offs, bands_offset__), nbeta);

                    linalg<pu>::gemm(0, 0,  nbf, num_bands__, nbf,
                                     operator__.at<CPU>(packed_mtrx_offsets__(ia)), nbf,
                                     ket__.at<CPU>(offs, bands_offset__), nbeta,
                                     result__.at<CPU>(offs, bands_offset__), nbeta);

                }
            }
        }


        static void chunked_states_product(int transpose_bra__,
                                           int transpose_ket__,
                                           const Beta_projectors::beta_chunk_t& chunk__,
                                           matrix<double_complex>& bra__,
                                           matrix<double_complex>& ket__,
                                           mdarray<double_complex, 1>& operator__,
                                           const mdarray<int, 1>& packed_mtrx_offsets__,
                                           int bands_offset__,
                                           int num_bands__)
        {
            //            if (ket__.size(0) != bra__.size(0))
            //            {
            //                TERMINATE("ERROR from tes::Non_local_operator<double_complex>::chunked_states_product :  ket__ and bra__ have wrong dimensions");
            //            }

            //            if ( op_->size() != result__.size() )
            //            {
            //                TERMINATE("ERROR from tes::Non_local_operator<double_complex>::chunked_states_product :  result__ must have the same size as operator");
            //            }

            if (static_cast<size_t>(chunk__.num_beta_ * (bands_offset__ + num_bands__) ) > bra__.size() ||
                    static_cast<size_t>(chunk__.num_beta_ * (bands_offset__ + num_bands__) ) > ket__.size()  )
            {
                TERMINATE("ERROR from tes::Non_local_operator<double_complex>::apply_op_to_ket :  result__ has too small size");
            }

            int nbeta = chunk__.num_beta_;

            if (pu == CPU)
            {
                #pragma omp parallel for
                for (int i = 0; i < chunk__.num_atoms_; i++)
                {
                    /* number of beta functions for a given atom */
                    int nbf = chunk__.desc_(0, i);
                    int offs = chunk__.desc_(1, i);
                    int ia = chunk__.desc_(3, i);

                    /* compute |result> = |ket> * <bra| or <bra| * |ket>   */
                    linalg<pu>::gemm(transpose_bra__, transpose_ket__,
                                      nbf, nbf, num_bands__,
                                      bra__.at<CPU>(offs, bands_offset__), nbeta,
                                      ket__.at<CPU>(offs, bands_offset__), nbeta,
                                      operator__.at<CPU>(packed_mtrx_offsets__(ia)), nbf);

                }
            }
        }
};




}

}


#endif /* SRC_OPERATOR_OPERATOR_H_ */
