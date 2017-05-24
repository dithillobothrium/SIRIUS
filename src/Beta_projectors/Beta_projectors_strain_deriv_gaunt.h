/*
 * Beta_projectors_stress.h
 *
 *  Created on: Feb 22, 2017
 *      Author: isivkov
 */

#ifndef __BETA_PROJECTORS_STRESS_H__
#define __BETA_PROJECTORS_STRESS_H__

#include "../utils.h"
#include "../Radial_integrals/Stress_radial_integrals.h"
#include "beta_projectors_base.h"
#include "beta_projectors.h"

namespace sirius
{

class Beta_projectors_strain_deriv_gaunt: public Beta_projectors_base<9>
{
protected:
    // dimensions of a 3d matrix (f.e. stress tensor)
    static const int nu_ = 3;
    static const int nv_ = 3;

public:

    struct lm2_gaund_coef_t
    {
        int lm2;
        double_complex prefac_gaunt_coef;
    };

    Beta_projectors_strain_deriv_gaunt(Simulation_context& ctx__,
                                       Gvec const&         gkvec__,
                                       Beta_projectors&    beta__)
    : Beta_projectors_base<9>(ctx__, gkvec__)
      {
        generate_pw_coefs_t(beta__);
      }

    static int ind(int i, int j)
    {
        return i * nu_ + j;
        //return (i + 1) * i / 2 + j ;
    }

    template<class ProcFuncT>
    inline void foreach_tensor(ProcFuncT func__)
    {
        for(size_t u=0; u < nu_; u++ ){
            for(size_t v=0; v < nv_; v++ ){
                func__(u,v);
            }
        }
    }

    template<class ProcFuncT>
    inline void foreach_m1m2(int l1__, int l2__, ProcFuncT func__)
    {
        for(int u=0; u < nu_; u++ ){
            for(int v=0; v < nv_; v++ ){
                func__(u,v);
            }
        }
    }

    void generate_pw_coefs_t(Beta_projectors& beta__)
    {
        PROFILE("sirius::Beta_projectors_strain_deriv_gaunt::generate_pw_coefs_t");

        double_complex const_prefac = double_complex(0.0 , 1.0) * fourpi * std::sqrt(fourpi / ( 3.0 * ctx_.unit_cell().omega() ));

        auto& comm = gkvec_.comm();

        // compute
        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++){
            auto& atom_type = ctx_.unit_cell().atom_type(iat);

            // TODO remove magic number
            Stress_radial_integrals stress_radial_integrals(ctx_.unit_cell(), ctx_.gk_cutoff(), 20);
            stress_radial_integrals.generate_beta_radial_integrals(iat);

            // + 1 because max BesselJ l is lbeta + 1 in summation
            int lmax_jl = atom_type.indexr().lmax() + 1;

            // get iterators of radial integrals
            auto& rad_ints = stress_radial_integrals.beta_l2_integrals();

            // array to store gaunt coeffs
            std::vector< std::vector< std::array< std::vector<lm2_gaund_coef_t>, 3> > > rbidx_m1_l2_m2_betaJ_gaunt_coefs;

            int nrb = atom_type.mt_radial_basis_size();

            // generate gaunt coeffs and store
            for (int idxrf = 0; idxrf < nrb; idxrf++){
                int l1 = atom_type.indexr(idxrf).l;
                std::vector<std::array<std::vector<lm2_gaund_coef_t>,3>> m1_l2_m2_betaJ_gaunt_coefs;

                for (int m1 = -l1; m1 <= l1; m1++){
                    std::array<std::vector<lm2_gaund_coef_t>, nu_> l2_m2_betaJ_gaunt_coefs;

                    for(auto& l2_rad_int: rad_ints[idxrf]){
                        int l2 = l2_rad_int.l2;

                        double_complex prefac = std::pow(double_complex(0.0, -1.0), l2);
                        for (int m2 = -l2; m2 <= l2; m2++){
                            int lm2 = Utils::lm_by_l_m(l2, m2);
                            double_complex full_prefact = prefac * const_prefac;
                            l2_m2_betaJ_gaunt_coefs[0].push_back( {lm2, -full_prefact * SHT::gaunt_rlm(l1, 1, l2,  m1, 1, m2  )} );
                            l2_m2_betaJ_gaunt_coefs[1].push_back( {lm2, -full_prefact * SHT::gaunt_rlm(l1, 1, l2,  m1, -1, m2 )} );
                            l2_m2_betaJ_gaunt_coefs[2].push_back( {lm2,  full_prefact * SHT::gaunt_rlm(l1, 1, l2,  m1, 0, m2  )} );
                        }
                    }
                    m1_l2_m2_betaJ_gaunt_coefs.push_back(std::move(l2_m2_betaJ_gaunt_coefs));
                }

                rbidx_m1_l2_m2_betaJ_gaunt_coefs.push_back(std::move(m1_l2_m2_betaJ_gaunt_coefs));
            }

            // iterate over gk vectors
            #pragma omp parallel for
            for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm.rank()); igkloc++)
            {
                int igk = gkvec_.gvec_offset(comm.rank()) + igkloc;
                auto gk_cart = gkvec_.gkvec_cart(igk);

                // vs = {r, theta, phi}
                auto vs = SHT::spherical_coordinates(gk_cart);

                // compute real spherical harmonics for G+k vector
                std::vector<double> gkvec_rlm(Utils::lmmax(lmax_jl));
                SHT::spherical_harmonics(lmax_jl, vs[1], vs[2], &gkvec_rlm[0]);

                // iterate over radial basis functions
                for (int idxrf = 0; idxrf < nrb; idxrf++){
                    int l1 = atom_type.indexr(idxrf).l;

                    auto& l2_rad_ints_rf = rad_ints[idxrf];

                    // there are maximum 2 radial integrals
                    double rad_ints_rf[] = {0.0, 0.0};

                    // store them in a static array
                    for(size_t l2_rad_int_idx=0; l2_rad_int_idx < l2_rad_ints_rf.size(); l2_rad_int_idx++){
                        rad_ints_rf[l2_rad_int_idx] = stress_radial_integrals.value_at(l2_rad_ints_rf[l2_rad_int_idx].rad_int, vs[0]);
                    }

                    // get start index for basis functions
                    int xi = atom_type.indexb().index_by_idxrf(idxrf);

                    // iterate over m-components of current radial baiss function
                    for (int im1 = 0; im1 < (int)rbidx_m1_l2_m2_betaJ_gaunt_coefs[idxrf].size(); im1++, xi++){

                        // iteration over column tensor index 'v'
                        for(int v = 0; v < nv_; v++){

                            // sum l2 m2 gaunt coefs and other stuff for given component and basis index xi
                            // store the result in component
                            double_complex component = 0.0;

                            // get reference to the stored before gaunt coefficients
                            auto& l2_m2_betaJ_gaunt_coefs = rbidx_m1_l2_m2_betaJ_gaunt_coefs[idxrf][im1][v];

                            // l2-m2 iterator for gaunt coefficients
                            int l2m2_idx = 0;

                            // summing loop, may be can be collapsed and vectorized
                            for(size_t l2_rad_int_idx = 0; l2_rad_int_idx < l2_rad_ints_rf.size(); l2_rad_int_idx++){
                                int l2 = l2_rad_ints_rf[l2_rad_int_idx].l2;

                                for (int m2 = -l2; m2 <= l2; m2++, l2m2_idx++){
                                    auto lm2_gc = l2_m2_betaJ_gaunt_coefs[l2m2_idx];
                                    int lm2 = lm2_gc.lm2;

                                    component += rad_ints_rf[l2_rad_int_idx] * gkvec_rlm[ lm2 ] * lm2_gc.prefac_gaunt_coef;
                                    //beta_gk_t_(igkloc, atom_type.offset_lo() + xi, u)
                                }
                            }

                            // iteratioon ove row tensor index 'u', add nondiag tensor components
                            for(int u = 0; u < nu_; u++ ){
                                pw_coeffs_t_[ind(u,v)](igkloc, atom_type.offset_lo() + xi) = gk_cart[u] * component;
                            }
                        }

                        for(int v = 0; v < nu_; v++ ){
                            // add diagonal components
                            pw_coeffs_t_[ind(v,v)](igkloc, atom_type.offset_lo() + xi) -= 0.5 * beta__.pw_coeffs_t(0)(igkloc, atom_type.offset_lo() + xi);
                        }
                    }
                }
            }
        }
        if (ctx_.processing_unit() == GPU) {
            for (int x = 0; x < num_; x++) {
                pw_coeffs_t_[x].copy<memory_t::host, memory_t::device>();
            }
        }
    }
};

}


#endif /* __BETA_PROJECTORS_STRESS_H__ */
