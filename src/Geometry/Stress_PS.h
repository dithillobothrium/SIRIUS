/*
 * Stress_PS.h
 *
 *  Created on: Feb 21, 2017
 *      Author: isivkov
 */

#ifndef __GEOMETRY_STRESS_PS_H__
#define __GEOMETRY_STRESS_PS_H__

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/beta_projectors_gradient.h"
#include "../Beta_projectors/Beta_projectors_lattice_gradient.h"
#include "../potential.h"
#include "../density.h"
#include "../Stress_radial_integrals.h"

namespace sirius
{

class Stress_PS
{
    private:
        Simulation_context* ctx_;
        Density* density_;
        Potential* potential_;
        K_point_set* kset_;

    public:
        Stress_PS(Simulation_context* ctx__,
                  Density* density__,
                  Potential* potential__,
                  K_point_set* kset__)
        : ctx_(ctx__),
          density_(density__),
          potential_(potential__),
          kset_(kset__)
        {
            Stress_radial_integrals rad_int(ctx_, &ctx_->unit_cell());

            // tmp
            rad_int.generate_beta_radial_integrals(0);

            Beta_projectors_lattice_gradient(&kset_->k_point(0)->beta_projectors(), ctx__);
        }



};

}


#endif /* SRC_GEOMETRY_STRESS_PS_H_ */
