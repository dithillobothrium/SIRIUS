/*
 * Beta_projectors_stress.h
 *
 *  Created on: Feb 22, 2017
 *      Author: isivkov
 */

#ifndef __BETA_PROJECTORS_STRESS_H__
#define __BETA_PROJECTORS_STRESS_H__

#include "Beta_projectors_array.h"
#include "../Stress_radial_integrals.h"

namespace sirius
{

class Beta_projectors_lattice_gradient: public Beta_projectors_array<6>
{
    protected:
        const Simulation_context* ctx_;

    public:
        Beta_projectors_lattice_gradient(Beta_projectors* bp__, const Simulation_context* ctx__)
        : Beta_projectors_array<6>(bp__),
          ctx_(ctx__)
        {
            initialize();
        }

        virtual void initialize()
        {
            Stress_radial_integrals rad_int(ctx_, &ctx_->unit_cell());

            //const std::vector<lpair>& lpairs = rad_int.radial_integrals_lpairs();

            auto rad_int_iterators = rad_int.radial_integral_iterators();

            for(auto rad_int_it = rad_int_iterators.first; rad_int_it != rad_int_iterators.second; rad_int_it++)
            {

            }
        }
};

}


#endif /* __BETA_PROJECTORS_STRESS_H__ */
