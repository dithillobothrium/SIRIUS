/*
 * Vloc_radial_integrals.h
 *
 *  Created on: Mar 3, 2017
 *      Author: isivkov
 */

#ifndef __VLOC_RADIAL_INTEGRALS_H__
#define __VLOC_RADIAL_INTEGRALS_H__

#include "../sbessel.h"

#include "Radial_integrals_base.h"

namespace sirius
{

class Vloc_radial_integrals: public Radial_integrals_base
{

    private:
        matrix<double> vloc_radial_integrals_;
        matrix<double> g_gradient_vloc_radial_integrals_;

        template<class ConvolveFuncT>
        inline void process_for_nonzero_gshell(int igs, int iat, std::vector< Spline<double> >& sa, ConvolveFuncT func__)
        {
                auto& atom_type = unit_cell_->atom_type(iat);
                for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                    double x = atom_type.radial_grid(ir);
                    sa[iat][ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn() * gsl_sf_erf(x)) * func__( ctx_->gvec().shell_len(igs) , x);
                }
        }

        inline void process_for_zero_gshell(int igs, int iat, std::vector< Spline<double> >& sa)
        {
            auto& atom_type = unit_cell_->atom_type(iat);
            for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                double x = atom_type.radial_grid(ir);
                sa[iat][ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn()) * x;
            }
        }

        template<class ProcessFuncT>
        void convolve_with_func(matrix<double>& radial_integrals__, ProcessFuncT func__)
        {
            radial_integrals__ = mdarray<double, 2>(unit_cell_->num_atom_types(), ctx_->gvec().num_shells());

            /* split G-shells between MPI ranks */
            splindex<block> spl_gshells(ctx_->gvec().num_shells(), ctx_->comm().size(), ctx_->comm().rank());

            #pragma omp parallel
            {
                /* splines for all atom types */
                std::vector< Spline<double> > sa(unit_cell_->num_atom_types());

                for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++) {
                    sa[iat] = Spline<double>(unit_cell_->atom_type(iat).radial_grid());
                }

            #pragma omp for
                for (int igsloc = 0; igsloc < spl_gshells.local_size(); igsloc++) {
                    int igs = spl_gshells[igsloc];

                    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++) {
                        func__(igs, iat, sa);
                    }
                }
            }

            int ld = unit_cell_->num_atom_types();
            ctx_->comm().allgather(radial_integrals__.at<CPU>(), ld * spl_gshells.global_offset(), ld * spl_gshells.local_size());
        }

    public:
        Vloc_radial_integrals(const Simulation_context* ctx__,
                              const Unit_cell* unit_cell__)
        : Radial_integrals_base(ctx__, unit_cell__)
        { }

        void generate_radial_integrals()
        {
            convolve_with_func( vloc_radial_integrals_, [&](int igs, int iat, std::vector< Spline<double> >& sa)
            {
                if(igs == 0){
                    process_for_zero_gshell(igs, iat, sa);
                    vloc_radial_integrals_(iat, igs) = sa[iat].interpolate().integrate(0);
                } else {
                    process_for_nonzero_gshell(igs, iat, sa, [&](double g, double x){ return std::sin(g * x); });
                    double g = ctx_->gvec().shell_len(igs);
                    double g2 = std::pow(g, 2);
                    vloc_radial_integrals_(iat, igs) = (sa[iat].interpolate().integrate(0) / g - unit_cell_->atom_type(iat).zn() * std::exp(-g2 / 4) / g2);
                }
            } );
        }

        void generate_g_gradient_radial_integrals()
        {
            convolve_with_func( g_gradient_vloc_radial_integrals_, [&](int igs, int iat, std::vector< Spline<double> >& sa)
            {
                if(igs == 0){
                    g_gradient_vloc_radial_integrals_(iat, igs) = 0.0;
                } else {
                    process_for_nonzero_gshell(igs, iat, sa, [&](double g, double x){ return x * std::cos(g * x) - std::sin(g * x) / g; });
                    double g = ctx_->gvec().shell_len(igs);
                    double g2 = std::pow(g, 2);
                    g_gradient_vloc_radial_integrals_(iat, igs) = sa[iat].interpolate().integrate(0) / g +
                            0.5 * unit_cell_->atom_type(iat).zn() * std::exp(-g2 / 4) / std::pow(g,3) * (4 + g2);
                }
            } );
        }

        const matrix<double>& vloc_radial_integrals()
        {
            return vloc_radial_integrals_;
        }

        const matrix<double>& g_gradient_vloc_radial_integrals()
        {
            return g_gradient_vloc_radial_integrals_;
        }

};

}


#endif /* SRC_TEST_VLOC_RADIAL_INTEGRALS_H_ */
