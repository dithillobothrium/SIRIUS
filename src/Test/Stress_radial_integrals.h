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

#include "../sbessel.h"

#include "Radial_integrals_base.h"

namespace sirius {

struct l2_rad_int_t
{
      int l2;
      Spline<double> rad_int;
};


/// generates and stores splines in G-space of beta radial integrals according to
/// l-selection rules for Gaunt coefficients { {l(idxrf), 1, l2} , {m1, m, m2}}
/// idxrf - radial function index, l2 - quantum number of J Bessel function
class Stress_radial_integrals : public Radial_integrals_base
{
    private:
        std::vector<std::vector<l2_rad_int_t>> atom_type_radial_integrals_;

    public:
        /// Constructor.
        Stress_radial_integrals(const Simulation_context* ctx__,
                              const Unit_cell* unit_cell__)
        : Radial_integrals_base(ctx__, unit_cell__)
        { }

        /// generate beta radial integral splines for given atom type index
        inline void generate_beta_radial_integrals(int atom_type_idx__)
        {
            PROFILE("sirius::Base_radial_integrals::generate_beta_radial_stress_integrals");

            int iat = atom_type_idx__;
            atom_type_radial_integrals_.clear();
            auto& atom_type = unit_cell_->atom_type(iat);
            int nrb = atom_type.mt_radial_basis_size();

            // beta radial splines
            std::vector<Spline<double>> rdists;

            for (int idxrf = 0; idxrf < nrb; idxrf++){
                // create beta radial spline
                Spline<double> rdist(atom_type.radial_grid());
                int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];

                for (int ir = 0; ir < nr; ir++){
                    rdist[ir] = atom_type.pp_desc().beta_radial_functions(ir, idxrf);
                }
                rdist.interpolate();
                rdists.push_back(std::move(rdist));

                // add l2 indices and allocated but not filled splines on g-grid for radial integrals
                int l  = atom_type.indexr(idxrf).l;
                std::vector<l2_rad_int_t> l2_rad_ints;

                if( l == 0){
                    l2_rad_ints.push_back({1,Spline<double>(grid_gkmax_)});
                } else {
                    l2_rad_ints.push_back({l-1,Spline<double>(grid_gkmax_)});
                    l2_rad_ints.push_back({l+1,Spline<double>(grid_gkmax_)});
                }

                atom_type_radial_integrals_.push_back(std::move(l2_rad_ints));
            }

            #pragma omp parallel for
            for (int iq = 0; iq < grid_gkmax_.num_points(); iq++){

                Spherical_Bessel_functions jl(unit_cell_->lmax() + 1, atom_type.radial_grid(), grid_gkmax_[iq]);

                for (int idxrf = 0; idxrf < nrb; idxrf++){
                    for(auto& l2_rad_int: atom_type_radial_integrals_[idxrf])
                    {
                        // compute \int j_l2(|G+k|r) beta_l(r) r^3 dr
                        // remember that beta(r) are defined as miltiplied by r
                        int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];
                        l2_rad_int.rad_int[iq] = sirius::inner(jl[l2_rad_int.l2], rdists[idxrf], 2, nr);
                    }
                }
                //printf("b_J rad_int %d %d ( ) = %f\n", l, l2, gdist[iq]);
            }

            for (auto& l2_rad_int_vec: atom_type_radial_integrals_){
                for(auto& l2_rad_int: l2_rad_int_vec)
                {
                    l2_rad_int.rad_int.interpolate();
                }
            }
        }


        inline double beta_radial_integral(int idxrf__, int l2idx__, double q__) const
        {
            return integral_at(atom_type_radial_integrals_[idxrf__][l2idx__].rad_int, q__);
        }

        inline double beta_radial_integral(const Spline<double>& rad_int__, double q__) const
        {
            auto iqdq = iqdq_gkmax(q__);
            return rad_int__(iqdq.first, iqdq.second);
        }

        inline const std::vector<std::vector<l2_rad_int_t>>& beta_l2_integrals() const
        {
            return atom_type_radial_integrals_;
        }



};

}


#endif /* BASE_RADIAL_INTEGRAL_H_ */
