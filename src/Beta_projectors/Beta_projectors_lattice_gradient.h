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

class Beta_projectors_lattice_gradient: public Beta_projectors_array<9>
{
    protected:
        const Simulation_context* ctx_;

        mdarray<double_complex, 3> beta_gk_t_;

        // dimensions of a 3d matrix (f.e. stress tensor)
        const int nu_ = 3;
        const int nv_ = 3;

    public:

        struct beta_besselJ_gaunt_coefs_t
        {
            int l1;
            int m1;
            int l2;
            int m2;
            std::array<double_complex, 3> prefac_gaunt_coefs;
        };

        Beta_projectors_lattice_gradient(Beta_projectors* bp__, const Simulation_context* ctx__)
        : Beta_projectors_array<9>(bp__),
          ctx_(ctx__)
        {
            init_beta_gk_t();
            init_beta_gk();
        }

        int ind(int i, int j)
        {
            return i * nu_ + j;
            //return (i + 1) * i / 2 + j ;
        }

        void init_beta_gk_t()
        {
            int num_beta_t = bp_->num_beta_by_atom_types();

            //const std::vector<lpair>& lpairs = rad_int.radial_integrals_lpairs();
            /* allocate array */
            beta_gk_t_ = mdarray<double_complex, 3>(bp_->num_gkvec_loc(), num_beta_t, nu_);
            beta_gk_t_.zero();

            double const_prefac = fourpi * std::sqrt(fourpi / ( 3.0 * ctx_->unit_cell().omega() ));

            //auto m1m2idx = [](int l1, int m1, int l2, int m2){ return (m1+l1) * (2*l2+1) + m2+l2; };

            // compute
            for (int iat = 0; iat < ctx_->unit_cell().num_atom_types(); iat++){
                auto& atom_type = ctx_->unit_cell().atom_type(iat);
                Stress_radial_integrals stress_radial_integrals(ctx_, &ctx_->unit_cell());
                stress_radial_integrals.generate_beta_radial_integrals(iat);

                // + 1 because max BesselJ l is lbeta + 1 in summation
                int lmax_jl = atom_type.indexr().lmax() + 1;

                // get iterators of radial integrals
                auto& rad_ints = stress_radial_integrals.beta_l2_integrals();

                // TODO: make common array of gaunt coefs for all atom types
                // array to store gaunt coeffs
                std::vector<std::vector<beta_besselJ_gaunt_coefs_t>> beta_besselJ_gaunt_coefs;

                int nrb = atom_type.mt_radial_basis_size();

                // generate gaunt coeffs and store
                for (int idxrf = 0; idxrf < nrb; idxrf++){
                    int l1 = atom_type.indexr(idxrf).l;
                    std::vector<beta_besselJ_gaunt_coefs_t> bj_gc;

                    std::cout<<"---------- "<<idxrf<<std::endl;

                    for(auto& l2_rad_int: rad_ints[idxrf]){
                        int l2 = l2_rad_int.l2;

                        double_complex prefac = std::pow(double_complex(0.0, -1.0), l2);

                        std::cout<<"l1l2 = "<<l1<<" "<<l2<<std::endl;

                        for (int m1 = -l1; m1 <= l1; m1++){
                            for (int m2 = -l2; m2 <= l2; m2++){
                                beta_besselJ_gaunt_coefs_t gc{
                                    l1,m1,l2,m2,
                                    {
                                            const_prefac * prefac * SHT::gaunt_rlm(l1, 1, l2, m1,  1, m2),     // for x component
                                            const_prefac * prefac * SHT::gaunt_rlm(l1, 1, l2, m1, -1, m2),     // for y component
                                            const_prefac * prefac * SHT::gaunt_rlm(l1, 1, l2, m1,  0, m2)      // for z component
                                    }
                                };
                                std::cout<<"m1m2 = "<<m1<<" "<<m2<<std::endl;
                                for(int comp: {0,1,2}) std::cout<<gc.prefac_gaunt_coefs[comp]<<" ";
                                std::cout<<std::endl;

                                bj_gc.push_back(std::move(gc));
                            }
                        }
                    }

                    beta_besselJ_gaunt_coefs.push_back(std::move(bj_gc));
                }

                // iterate over gk vectors
                #pragma omp parallel for
                for (int igkloc = 0; igkloc < bp_->num_gkvec_loc(); igkloc++)
                {
                    int igk   = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
                    double gk = bp_->gk_vectors().gvec_len(igk);

                    // vs = {r, theta, phi}
                    auto vs = SHT::spherical_coordinates(bp_->gk_vectors().gkvec_cart(igk));

                    // compute real spherical harmonics for G+k vector
                    std::vector<double> gkvec_rlm(Utils::lmmax(lmax_jl));
                    SHT::spherical_harmonics(lmax_jl, vs[1], vs[2], &gkvec_rlm[0]);

                    // SHOULD WORK
                    for (int idxrf = 0; idxrf < nrb; idxrf++){
                        int l1 = atom_type.indexr(idxrf).l;

                        // m1-l2-m2 iterator for gaunt coefficients
                        int m1l2m2_it = 0;

                        // get gaunt coefs between beta l1 and besselJ l2
                        auto& bj_gc = beta_besselJ_gaunt_coefs[idxrf];

                        for(auto& l2_rad_int: rad_ints[idxrf]){
                            int l2 = l2_rad_int.l2;

                            // multiply radial integral and constants
                            double radint = stress_radial_integrals.integral_at(l2_rad_int.rad_int,gk);

                            // get start index for basis functions
                            int xi = atom_type.indexb().index_by_idxrf(idxrf);

                            // iterate over m-components of l1 and l2
                            for (int m1 = -l1; m1 <= l1; m1++, xi++){
                                for (int m2 = -l2; m2 <= l2; m2++, m1l2m2_it++){
                                    for(int comp=0; comp< nu_; comp++){
                                        beta_gk_t_(igkloc, atom_type.offset_lo() + xi, comp) += radint *
                                                gkvec_rlm[ Utils::lm_by_l_m(l2, m2) ] * bj_gc[ m1l2m2_it ].prefac_gaunt_coefs[comp];
                                    }
                                }
                            }
                        }

                        // debug
//                        int xi = atom_type.indexb().index_by_idxrf(idxrf);
//                        auto Gcart = bp_->gk_vectors().gvec_cart(igk);

//#pragma omp critical
//                        {
//                            for (int m1 = -l1; m1 <= l1; m1++, xi++){
//                                std::cout<<"beta_gt_t "<<l1<<" "<<xi<<" | "<< Gcart<<" | ";
//                                for(int comp=0; comp< nu_; comp++){
//                                    std::cout<<beta_gk_t_(igkloc, atom_type.offset_lo() + xi, comp)<<" ";
//                                }
//                                std::cout<<std::endl;
//                            }
//                            std::cout<<"===================="<<std::endl;
//                        }
                    }
                }
            }
        }

        void init_beta_gk()
        {
            auto& unit_cell = ctx_->unit_cell();
            int num_gkvec_loc = bp_->num_gkvec_loc();

            // TODO maybe remove it
            for(int i = 0; i < this->num_; i++)
            {
                components_gk_a_[i].zero();
            }

            auto cell_matrix = unit_cell.lattice_vectors();
            auto inv_cell_matrix = unit_cell.reciprocal_lattice_vectors();

            double omega = unit_cell.omega();

            double_complex i_over_omega = double_complex(0.0,1.0) / omega;

            #pragma omp parallel for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {

                // prepare G+k phases
                auto vk = bp_->gk_vectors().vk();
                double phase = twopi * (vk * unit_cell.atom(ia).position());
                double_complex phase_k = std::exp(double_complex(0.0, phase));

                std::vector<double_complex> phase_gk(num_gkvec_loc);

                for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
                    int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igk_loc;
                    auto G = bp_->gk_vectors().gvec(igk);
                    phase_gk[igk_loc] = std::conj(ctx_->gvec_phase_factor(G, ia) * phase_k);
                }

                // cartesian atomic coordinate
                auto Rcart = cell_matrix * unit_cell.atom(ia).position();
                auto vk_cart = inv_cell_matrix * vk;

                std::cout<<"vk = "<<vk_cart<<std::endl;

#pragma omp critical
                {
                std::cout<<"k R = "<<vk_cart<<"     "<<Rcart<<std::endl;
                }
                // TODO: need to optimize order of loops
                // calc beta lattice gradient
                for (int xi = 0; xi < unit_cell.atom(ia).mt_lo_basis_size(); xi++) {
                    for (int igkloc = 0; igkloc < bp_->num_gkvec_loc(); igkloc++) {
                        int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
                        auto Gcart = bp_->gk_vectors().gvec_cart(igk);

                        // iteration over tensor components
                        for(int u = 0; u < nu_; u++){
                            for(int v = 0; v < nv_; v++){
                                // complicate formula
                                components_gk_a_[ind(u,v)](igkloc, unit_cell.atom(ia).offset_lo() + xi) =
                                        // first
                                        beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, v) *
                                        phase_gk[igkloc] * Gcart[u] * i_over_omega -
                                        // second
                                        bp_->beta_gk_a()(igkloc, unit_cell.atom(ia).offset_lo() + xi) *
                                        double_complex(0.0 , 1.0) * vk_cart[u] * Rcart[v] / omega;


                            }
                            // third
                            components_gk_a_[ind(u,u)](igkloc, unit_cell.atom(ia).offset_lo() + xi) +=
                                    bp_->beta_gk_a()(igkloc, unit_cell.atom(ia).offset_lo() + xi) / omega * 0.5;
                        }

#pragma omp critical
                        {
                            std::cout<<"test non-local stress:"<<std::endl;

                                std::cout<<"G beta_gt= "<<Gcart<<"  |   ";
                                for(int j=0; j<3; j++){
                                    std::cout<<beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, j) <<"  ";
                                }
                                std::cout<<std::endl;


                            for(int i=0; i<3; i++){
                                for(int j=0; j<3; j++){
                                    std::cout<<beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, j) * Gcart[i] <<"  ";
                                }
                                std::cout<<std::endl;
                            }

                        }
                    }
                }
            }
        }



};

}


#endif /* __BETA_PROJECTORS_STRESS_H__ */
