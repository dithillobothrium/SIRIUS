/*
 * Base_radial_integral.h
 *
 *  Created on: Feb 20, 2017
 *      Author: isivkov
 */

#ifndef __BASE_RADIAL_INTEGRALS_H__
#define __BASE_RADIAL_INTEGRALS_H__

#include <unordered_map>
#include <functional>
#include <memory>
#include <algorithm>

#include "sbessel.h"

namespace sirius {

using lpair = std::pair<int, int>;

// simple hash for unordered map which will store pairs of indices
struct lpair_hash
{
        const int num = 1000;

        std::size_t operator () (const lpair& p) const
        {
            return p.first * num + p.second;
        }
};


using Lpair_spline_map = std::unordered_map<lpair, Spline<double>, lpair_hash >;

/// generates and stores splines in G-space of beta radial integrals according to
/// l-selection rules for Gaunt coefficients { {l(idxrf), 1, l2} , {m1, m, m2}}
/// idxrf - radial function index, l2 - quantum number of J Bessel function
class Stress_radial_integrals
{
    private:
        /// Basic parameters.
        const Simulation_parameters* param_;

        /// Unit cell.
        const Unit_cell* unit_cell_;

        /// Linear grid up to |G+k|_{max}
        Radial_grid grid_gkmax_;

        /// Linear grid up to |G|_{max}
        Radial_grid grid_gmax_;

        Lpair_spline_map atom_type_radial_integrals_;

        std::vector<lpair> radial_integrals_lpairs_;

        inline std::pair<int, double> iqdq_gkmax(double q__) const
            {
            std::pair<int, double> result;
            result.first = static_cast<int>((grid_gkmax_.num_points() - 1) * q__ / param_->gk_cutoff());
            /* delta q = q - q_i */
            result.second = q__ - grid_gkmax_[result.first];
            return std::move(result);
            }

        inline std::pair<int, double> iqdq_gmax(double q__) const
            {
            std::pair<int, double> result;
            result.first = static_cast<int>((grid_gmax_.num_points() - 1) * q__ / param_->pw_cutoff());
            /* delta q = q - q_i */
            result.second = q__ - grid_gmax_[result.first];
            return std::move(result);
            }

    public:
        /// Constructor.
        Stress_radial_integrals(const Simulation_parameters* param__,
                              const Unit_cell* unit_cell__)
        : param_(param__)
        , unit_cell_(unit_cell__)
        {
            grid_gmax_  = Radial_grid(linear_grid, static_cast<int>(12 * param_->pw_cutoff()), 0, param_->pw_cutoff());
            grid_gkmax_ = Radial_grid(linear_grid, static_cast<int>(12 * param_->gk_cutoff()), 0, param_->gk_cutoff());
        }

        /// generate beta radial integral splines for given atom type index
        inline void generate_beta_radial_integrals(int atom_type_idx__)
        {
            PROFILE("sirius::Base_radial_integrals::generate_beta_radial_stress_integrals");

            int iat = atom_type_idx__;

            atom_type_radial_integrals_.clear();
            radial_integrals_lpairs_.clear();

            auto& atom_type = unit_cell_->atom_type(iat);

            int nrb = atom_type.mt_radial_basis_size();


            for (int idxrf = 0; idxrf < nrb; idxrf++)
            {
                int l  = atom_type.indexr(idxrf).l;

                std::vector<int> l2s;

                if( l == 0)
                {
                    l2s.push_back(1);
                }
                else
                {
                    l2s.push_back(l-1);
                    l2s.push_back(l+1);
                }

                // create radial spline
                Spline<double> rdist(atom_type.radial_grid());

                int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];

                for (int ir = 0; ir < nr; ir++)
                {
                    rdist[ir] = atom_type.pp_desc().beta_radial_functions(ir, idxrf);
                }

                rdist.interpolate();

                // create g-grid spline
                for(auto l2 : l2s)
                {
                    Spline<double> gdist(grid_gkmax_);

                    for (int iq = 0; iq < grid_gkmax_.num_points(); iq++)
                    {
                        // just for a case since
                        Spherical_Bessel_functions jl(unit_cell_->lmax() + 1, atom_type.radial_grid(), grid_gkmax_[iq]);

                        // compute \int j_l2(|G+k|r) beta_l(r) r^3 dr
                        // remeber that beta(r) are defined as miltiplied by r
                        gdist[iq] = sirius::inner(jl[l2], rdist, 2, nr);
                    }

                    atom_type_radial_integrals_.insert( std::make_pair( lpair(idxrf, l2), std::move(gdist) ) );
                    radial_integrals_lpairs_.push_back(lpair(idxrf, l2));
                }
            }
        }

        /// return value of radial intefgral at point g__
        /// key is a pair < idxrf - radial function index, l2 - quantum number>
        inline double beta_radial_integral(const lpair& key, double q__)
        {
            auto iqdq = iqdq_gkmax(q__);
            return atom_type_radial_integrals_[key](iqdq.first, iqdq.second);
        }

        inline double beta_radial_integral(const Lpair_spline_map::iterator& it__, double q__)
        {
            auto iqdq = iqdq_gkmax(q__);
            return it__->second(iqdq.first, iqdq.second);
        }

        /// return pairs < idxrf - radial function index, l2 - quantum number>
        inline const std::vector<lpair>& radial_integrals_lpairs()
        {
            return radial_integrals_lpairs_;
        }

        inline std::vector<lpair> find_lpair_by_radial_index(int idxrf)
        {
            std::vector<lpair> lpairs;
            for(auto lpair: radial_integrals_lpairs_)
            {
                if(lpair.first == idxrf)
                {
                    lpairs.push_back(lpair);
                }
            }
            return lpairs;
        }

        const Lpair_spline_map::iterator find_radial_integral(const lpair& key)
        {
            return atom_type_radial_integrals_.find(key);
        }

        std::pair<Lpair_spline_map::iterator,Lpair_spline_map::iterator> radial_integral_iterators()
        {
            return std::make_pair(atom_type_radial_integrals_.begin(), atom_type_radial_integrals_.end());
        }


};

}


#endif /* BASE_RADIAL_INTEGRAL_H_ */
