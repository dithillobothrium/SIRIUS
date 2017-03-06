/*
 * Gspace_radial_integrals.h
 *
 *  Created on: Mar 3, 2017
 *      Author: isivkov
 */

#ifndef __RADIAL_INTEGRALS_BASE_H__
#define __RADIAL_INTEGRALS_BASE_H__

#include "../sbessel.h"

namespace sirius
{


class Radial_integrals_base
{
    protected:
        /// Basic parameters.
        const Simulation_context* ctx_;

        /// Unit cell.
        const Unit_cell* unit_cell_;

        /// Linear grid up to |G+k|_{max}
        Radial_grid grid_gkmax_;

        /// Linear grid up to |G|_{max}
        Radial_grid grid_gmax_;

    public:
        /// Constructor.
        Radial_integrals_base(const Simulation_context* ctx__,
                              const Unit_cell* unit_cell__)
            : ctx_(ctx__)
            , unit_cell_(unit_cell__)
        {
            grid_gmax_  = Radial_grid(linear_grid, static_cast<int>(12 * ctx_->pw_cutoff()), 0, ctx_->pw_cutoff());
            grid_gkmax_ = Radial_grid(linear_grid, static_cast<int>(12 * ctx_->gk_cutoff()), 0, ctx_->gk_cutoff());
        }

        inline std::pair<int, double> iqdq_gkmax(double q__) const
        {
            std::pair<int, double> result;
            result.first = static_cast<int>((grid_gkmax_.num_points() - 1) * q__ / ctx_->gk_cutoff());
            /* delta q = q - q_i */
            result.second = q__ - grid_gkmax_[result.first];
            return std::move(result);
        }

        inline std::pair<int, double> iqdq_gmax(double q__) const
        {
            std::pair<int, double> result;
            result.first = static_cast<int>((grid_gmax_.num_points() - 1) * q__ / ctx_->pw_cutoff());
            /* delta q = q - q_i */
            result.second = q__ - grid_gmax_[result.first];
            return std::move(result);
        }

        inline double integral_at(Spline<double>& integral__, double q__)
        {
            auto iqdq = iqdq_gkmax(q__);
            return integral__(iqdq.first, iqdq.second);
        }
};

}
#endif /* SRC_TEST_GSPACE_RADIAL_INTEGRALS_H_ */
