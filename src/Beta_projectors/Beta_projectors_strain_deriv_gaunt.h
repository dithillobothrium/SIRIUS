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
        static const size_t nu_ = 3;
        static const size_t nv_ = 3;

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

        static size_t ind(size_t i, size_t j)
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

                    //                    std::cout<<"---------- "<<idxrf<<std::endl;

                    //                        std::cout<<"l1l2 = "<<l1<<" "<<l2<<std::endl;

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

                                //                                std::cout<<"m1m2 = "<<m1<<" "<<m2<<std::endl;
                                //                                for(int comp: {0,1,2}) std::cout<<gc.prefac_gaunt_coefs[comp]<<" ";
                                //                                std::cout<<std::endl;
                            }
                        }
                        m1_l2_m2_betaJ_gaunt_coefs.push_back(std::move(l2_m2_betaJ_gaunt_coefs));
                    }

                    rbidx_m1_l2_m2_betaJ_gaunt_coefs.push_back(std::move(m1_l2_m2_betaJ_gaunt_coefs));
                }

                // iterate over gk vectors
                //#pragma omp parallel for
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
                        for (int im1 = 0; im1 < rbidx_m1_l2_m2_betaJ_gaunt_coefs[idxrf].size(); im1++, xi++){

                            // iteration over column tensor index 'v'
                            for(size_t v = 0; v < nv_; v++){

                                // sum l2 m2 gaunt coefs and other stuff for given component and basis index xi
                                // store the result in component
                                double_complex component = 0.0;

                                // get reference to the stored before gaunt coefficients
                                auto& l2_m2_betaJ_gaunt_coefs = rbidx_m1_l2_m2_betaJ_gaunt_coefs[idxrf][im1][v];

                                // l2-m2 iterator for gaunt coefficients
                                int l2m2_idx = 0;

                                // summing loop, may be can be collapsed and vectorized
                                for(size_t l2_rad_int_idx=0; l2_rad_int_idx < l2_rad_ints_rf.size(); l2_rad_int_idx++){
                                    int l2 = l2_rad_ints_rf[l2_rad_int_idx].l2;

                                    for (int m2 = -l2; m2 <= l2; m2++, l2m2_idx++){
                                        auto lm2_gc = l2_m2_betaJ_gaunt_coefs[l2m2_idx];
                                        int lm2 = lm2_gc.lm2;

                                        component += rad_ints_rf[l2_rad_int_idx] * gkvec_rlm[ lm2 ] * lm2_gc.prefac_gaunt_coef;
                                        //beta_gk_t_(igkloc, atom_type.offset_lo() + xi, u)
                                    }
                                }

                                // iteratioon ove row tensor index 'u', add nondiag tensor components
                                for(size_t u = 0; u < nu_; u++ ){
                                    pw_coeffs_t_[ind(u,v)](igkloc, atom_type.offset_lo() + xi) = gk_cart[u] * component;
                                }

                                // add diagonal components
                                pw_coeffs_t_[ind(v,v)](igkloc, atom_type.offset_lo() + xi) -= 0.5 * beta__.pw_coeffs_t(0)(igkloc, atom_type.offset_lo() + xi);
                            }



                            //std::cout<<"\n"<<bj_gc.size()<<std::endl;
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

//        void init_beta_gk()
//        {
//            auto& unit_cell = ctx_->unit_cell();
//            int num_gkvec_loc = bp_->num_gkvec_loc();
//
//            // TODO maybe remove it
//            for(size_t i = 0; i < this->num_; i++)
//            {
//                components_gk_a_[i].zero();
//            }
//
//            //#pragma omp parallel for
//            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
//
//                // prepare G+k phases
//                auto vk = bp_->gk_vectors().vk();
//                double phase = twopi * (vk * unit_cell.atom(ia).position());
//                double_complex phase_k = std::exp(double_complex(0.0, phase));
//
//                std::vector<double_complex> phase_gk(num_gkvec_loc);
//
//                for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
//                    int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igk_loc;
//                    auto G = bp_->gk_vectors().gvec(igk);
//                    phase_gk[igk_loc] = std::conj(ctx_->gvec_phase_factor(G, ia) * phase_k);
//                }
//
//
//                // TODO: need to optimize order of loops
//                // calc beta lattice gradient
//                for (int xi = 0; xi < unit_cell.atom(ia).mt_lo_basis_size(); xi++) {
//                    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++) {
//                        int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
//                        // TODO:store vectors before
//                        auto Gcart = bp_->gk_vectors().gkvec_cart(igk);
//
//                        // iteration over tensor components
//                        for(size_t u = 0; u < nu_; u++){
//                            for(size_t v = 0; v < nv_; v++){
//                                // complicate formula
//                                components_gk_a_[ind(u,v)](igkloc, unit_cell.atom(ia).offset_lo() + xi) =
//                                        // first
//                                        beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, v) *
//                                        phase_gk[igkloc] * Gcart[u] * double_complex(0.0 , 1.0);
//                            }
//                        }
//
//                        // third
//                        for(size_t u = 0; u < nu_; u++){
//                            components_gk_a_[ind(u,v)](igkloc, unit_cell.atom(ia).offset_lo() + xi) -=
//                                    bp_->beta_gk_a()(igkloc, unit_cell.atom(ia).offset_lo() + xi) * 0.5;
//                        }
////                        #pragma omp critical
////                        {
////                            std::cout<<"test non-local stress:"<<std::endl;
////
////                                std::cout<<"G beta_gt= "<<Gcart<<"  |   ";
////                                for(int j=0; j<3; j++){
////                                    std::cout<<beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, j) <<"  ";
////                                }
////                                std::cout<<std::endl;
////
////
////                            for(int i=0; i<3; i++){
////                                for(int j=0; j<3; j++){
////                                    std::cout<<components_gk_a_[ind(i,j)](igkloc, unit_cell.atom(ia).offset_lo() + xi) <<"  ";
////                                }
////                                std::cout<<std::endl;
////                            }
////
////                        }
//                    }
//                }
//            }
//        }
};

}

/*


 Cu_paw QE 4x4x4 kp
 -9.1692907433697818E-002    -1.3408784222729286E-004    -1.3394692060626073E-004
  -1.3389837891349815E-004    -9.1530136341425650E-002     1.9127056857143054E-003
  -1.3375766277778228E-004     1.9127054344352384E-003    -9.1530257493276951E-002

 Cu_paw Sirius 4x4x4 kp
-0.0423254 -5.44514e-05 -5.44521e-05
-5.46125e-05 -0.042157 0.000940562
-5.46132e-05 0.000940562 -0.042157

LiF_paw QE
  -4.5659140602775271E-004     4.2385491663234168E-005     4.2385550851812388E-005
   4.2385491663234168E-005    -5.9273534534111654E-004     1.3298952395974793E-004
   4.2385550851812388E-005     1.3298952395974793E-004    -5.9273538256867698E-004

LiF_paw Sirius
0.000226209 -2.11201e-05 -2.11201e-05
-2.13106e-05 0.000294652 -6.64715e-05
-2.13106e-05 -6.64715e-05 0.000294652

-0.000226209 2.11201e-05 2.11201e-05
2.13106e-05 -0.000294652 6.64715e-05
2.13106e-05 6.64715e-05 -0.000294652

0.00155738 4.23852e-05 4.23852e-05
3.02141e-05 0.00151196 1.88412e-05
3.02141e-05 1.88412e-05 0.00151196


//////////

  QE
   3.1302203360987209E-003     4.7953985328350751E-005     7.0629952604793107E-005
   4.7953985328350751E-005     3.0749076361499608E-003     1.0909589297567711E-004
   7.0629952604793107E-005     1.0909589297567711E-004     3.0313974757552168E-003

   S
   0.00177094 6.04164e-06 1.51468e-05
   2.39659e-05 0.0017496 6.90991e-05
   3.28976e-05 6.83131e-05 0.00172689

-0.0113956 3.48557e-05 5.54032e-05
3.04608e-05 -0.011442 0.000101329
5.10887e-05 0.000101173 -0.0114627



   3.1302203360987204E-003     4.7980593047019002E-005     7.0673691978238414E-005
   4.7953985328350812E-005     3.0749076361499617E-003     1.0914875484888534E-004
   7.0629952604793093E-005     1.0909589297567726E-004     3.0313974757552151E-003

   0.00156335 2.40018e-05 3.536e-05
2.40382e-05 0.00153565 5.46457e-05
3.54212e-05 5.46816e-05 0.00151385


   3.6654353383783972E-003     2.8401307640959963E-006     4.3743532633919776E-006
   2.8691262927431523E-006     3.6201845901544759E-003     1.6488786421643784E-006
   4.3853192999605828E-006     1.6676819410345756E-006     3.6410013971595731E-003


0.00183164 1.43495e-06 2.19392e-06
1.42045e-06 0.00180901 8.37579e-07
2.18843e-06 8.28175e-07 0.00181942

0.00183164 1.435e-06 2.19386e-06
1.4205e-06 0.00180901 8.3754e-07
2.18838e-06 8.28136e-07 0.00181942

-0.0378749 0.000273295 -7.24611e-05
0.000271807 -0.0374295 -1.80703e-05
-7.32491e-05 -5.51587e-06 -0.0398204
 */


#endif /* __BETA_PROJECTORS_STRESS_H__ */
