/*
 * Beta_projectors_stress.h
 *
 *  Created on: Feb 22, 2017
 *      Author: isivkov
 */

#ifndef __BETA_PROJECTORS_STRESS_H__
#define __BETA_PROJECTORS_STRESS_H__

#include "../utils.h"
#include "Beta_projectors_array.h"
#include "../Test/Stress_radial_integrals.h"

namespace sirius
{

class Beta_projectors_lattice_gradient: public Beta_projectors_array<6>
{
    protected:
        const Simulation_context* ctx_;

        matrix<double_complex> beta_gk_t_;

    public:

        struct beta_J_gaunt_coefs_t
        {
            int l1;
            int m1;
            int l2;
            int m2;
            std::array<double, 3> gaunt_coefs;
        };

        Beta_projectors_lattice_gradient(Beta_projectors* bp__, const Simulation_context* ctx__)
        : Beta_projectors_array<6>(bp__),
          ctx_(ctx__)
        {
            initialize();
        }


        virtual void initialize()
        {
            int num_beta_t = bp_->num_beta_by_atom_types();

            //const std::vector<lpair>& lpairs = rad_int.radial_integrals_lpairs();
            /* allocate array */
            beta_gk_t_ = matrix<double_complex>(bp_->num_gkvec_loc(), num_beta_t);
            matrix3d<double> const& inverse_lattice_vectors = ctx_->unit_cell().symmetry().inverse_lattice_vectors();
            double const_prefac = fourpi * std::sqrt(fourpi / ( 3.0 * ctx_->unit_cell().omega() ));

            //auto m1m2idx = [](int l1, int m1, int l2, int m2){ return (m1+l1) * (2*l2+1) + m2+l2; };

            // compute
            for (int iat = 0; iat < ctx_->unit_cell().num_atom_types(); iat++){
                auto& atom_type = ctx_->unit_cell().atom_type(iat);
                Stress_radial_integrals rad_int(ctx_, &ctx_->unit_cell());
                rad_int.generate_beta_radial_integrals(iat);

                // + 1 because max BesselJ l is lbeta + 1 in summation
                int lmax_jl = atom_type.indexr().lmax() + 1;

                #pragma omp parallel
                {
                    // TODO: maybe put pair<iterator,iterator> in private clause
                    // and move gaunt rlm out from omp region (how about read access from different threads)
                    auto rad_int_iterators = rad_int.radial_integral_iterators();

                    // array to store gaunt coeffs
                    std::vector<std::vector<beta_J_gaunt_coefs_t>> betaj_gc;

                    // generate gaunt coeffs and store
                    for(auto rad_int_it = rad_int_iterators.first; rad_int_it != rad_int_iterators.second; rad_int_it++){
                        lpair lp = rad_int_it->first;
                        int idxrf = lp.first;
                        int l1 = atom_type.indexr(idxrf).l;
                        int l2 = lp.second;
                        std::vector<beta_J_gaunt_coefs_t> bj_gc;

                        for (int m1 = -l1; m1 <= l1; m1++){
                            for (int m2 = -l2; m2 <= l2; m2++){
                                bj_gc.push_back({
                                    l1,m1,l2,m2,
                                    {SHT::gaunt_rlm(l1, 1, l2, m1,  1, m2), // for x component
                                            SHT::gaunt_rlm(l1, 1, l2, m1, -1, m2),  // for y component
                                            SHT::gaunt_rlm(l1, 1, l2, m1,  0, m2)}  // for z component
                                });
                            }
                        }

                        betaj_gc.push_back(bj_gc);
                    }


                    #pragma omp for
                    for (int igkloc = 0; igkloc < bp_->num_gkvec_loc(); igkloc++)
                    {
                        int igk   = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
                        double gk = bp_->gk_vectors().gvec_len(igk);

                        // vs = {r, theta, phi}
                        auto vs = SHT::spherical_coordinates(bp_->gk_vectors().gkvec_cart(igk));

                        // compute real spherical harmonics for G+k vector
                        std::vector<double> gkvec_rlm(Utils::lmmax(lmax_jl));
                        SHT::spherical_harmonics(lmax_jl, vs[1], vs[2], &gkvec_rlm[0]);

                        //WILL NOT WORK
                        for(auto rad_int_it = rad_int_iterators.first; rad_int_it != rad_int_iterators.second; rad_int_it++){
                            lpair lp = rad_int_it->first;
                            int idxrf = lp.first;
                            int l1 = atom_type.indexr(idxrf).l;
                            int l2 = lp.second;
                            int idx_bf_start = atom_type.indexb().index_by_idxrf(idxrf);


                            // get radial integral value at gk
                            double rad_int_val = rad_int.beta_radial_integral(rad_int_it, gk);
                            int beta_j_gc_it = 0;
                            auto& bj_gc = betaj_gc[beta_j_gc_it++];

                            // temp variables
                            double_complex rlm_gaunt[]={0.0, 0.0, 0.0};
                            double_complex prefact = const_prefac * std::pow(double_complex(0, -1), l2) * fourpi / std::sqrt(ctx_->unit_cell().omega());

                            // get start index for basis functions
                            int xi = atom_type.indexb().index_by_idxrf(idxrf);
                            int m1m2_it = 0;

                            // iterate over m-components of l1 and l2
                            for (int m1 = -l1; m1 <= l1; m1++, xi++){
                                for (int m2 = -l2; m2 <= l2; m2++){
                                    for(auto comp: {0,1,2}){
                                        rlm_gaunt[comp] = gkvec_rlm[ Utils::lm_by_l_m(l2, m2) ] * bj_gc[ m1m2_it++ ].gaunt_coefs[comp];
                                    }
                                    prefact * beta_gk_t_(igkloc, atom_type.offset_lo() + xi);
                                }
                            }

                            //                           for (int xi = 0; xi < atom_type.mt_basis_size(); xi++)
                            //                           {
                            //                               int l     = atom_type.indexb(xi).l;
                            //                               int lm    = atom_type.indexb(xi).lm;
                            //                               int idxrf = atom_type.indexb(xi).idxrf;
                            //
                            //                               double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_->unit_cell().omega());
                            //                               beta_gk_t_(igkloc, atom_type.offset_lo() + xi) += z * gkvec_rlm[lm] * ctx__.radial_integrals().beta_radial_integral(idxrf, iat, gk);
                            //                           }
                        }
                    }
                }
            }
        }
};

}


#endif /* __BETA_PROJECTORS_STRESS_H__ */
