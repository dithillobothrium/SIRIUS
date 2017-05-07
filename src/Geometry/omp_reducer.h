/*
 * omp_reductors.h
 *
 *  Created on: May 7, 2017
 *      Author: isivkov
 */

#ifndef SRC_GEOMETRY_OMP_REDUCER_H_
#define SRC_GEOMETRY_OMP_REDUCER_H_

#include "SDDK/mdarray.hpp"

namespace sirius
{
//------------------------------------------
template<typename T>
void init_mdarray2d(mdarray<T,2> &priv, mdarray<T,2> &orig )
{
    priv = mdarray<double,2>(orig.size(0),orig.size(1)); priv.zero();
}

template<typename T>
void add_mdarray2d(mdarray<T,2> &in, mdarray<T,2> &out)
{
    for(size_t i = 0; i < in.size(1); i++ ) {
        for(size_t j = 0; j < in.size(0); j++ ) {
            out(j,i) += in(j,i);
        }
    }
}

#pragma omp declare reduction( + : mdarray<double,2> : add_mdarray2d(omp_in, omp_out))  initializer( init_mdarray2d(omp_priv, omp_orig) )


}


#endif /* SRC_GEOMETRY_OMP_REDUCER_H_ */
