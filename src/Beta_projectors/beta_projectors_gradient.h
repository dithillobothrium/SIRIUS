/*
 * Beta_projectors_gradient.h
 *
 *  Created on: Oct 14, 2016
 *      Author: isivkov
 */

#ifndef __BETA_PROJECTORS_GRADIENT_H__
#define __BETA_PROJECTORS_GRADIENT_H__


#include "beta_projectors.h"
#include "Beta_projectors_array.h"

namespace sirius
{

/// Stores gradient components of beta over atomic positions d <G+k | Beta > / d Rn
class Beta_projectors_gradient : public Beta_projectors_array<3>
{


public:

    Beta_projectors_gradient(Beta_projectors* bp)
    : Beta_projectors_array<3>(bp)
    {
        for(int comp: {0,1,2})
        {
            initialize(comp);
        }
    }


    void initialize(int calc_component__)
    {
        Gvec const& gkvec = bp_->gk_vectors();

        matrix<double_complex> const& beta_comps = bp_->beta_gk_a();

        double_complex Im(0, 1);

        #pragma omp parallel for
        for (size_t ibf = 0; ibf < bp_->beta_gk_a().size(1); ibf++) {
            for (int igk_loc = 0; igk_loc < bp_->num_gkvec_loc(); igk_loc++) {
                int igk = gkvec.gvec_offset(bp_->comm().rank()) + igk_loc;

                double gkvec_comp = gkvec.gkvec_cart(igk)[calc_component__];

                components_gk_a_[calc_component__](igk_loc, ibf) = - Im * gkvec_comp * beta_comps(igk_loc,ibf);
            }
        }
    }
};

}

#endif /* SRC_BETA_PROJECTORS_BETA_PROJECTORS_GRADIENT_H_ */
