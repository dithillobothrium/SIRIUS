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
        static const size_t nu_ = 3;
        static const size_t nv_ = 3;

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
            init_beta_gk_t2();
            init_beta_gk2();
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

        void init_beta_gk_t()
        {
            int num_beta_t = bp_->num_beta_by_atom_types();

            //const std::vector<lpair>& lpairs = rad_int.radial_integrals_lpairs();
            /* allocate array */
            beta_gk_t_ = mdarray<double_complex, 3>(bp_->num_gkvec_loc(), num_beta_t, nu_);
            beta_gk_t_.zero();

            double const_prefac = fourpi * std::sqrt(fourpi / ( 3.0 * ctx_->unit_cell().omega() ));


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

//                    std::cout<<"---------- "<<idxrf<<std::endl;

                    for(auto& l2_rad_int: rad_ints[idxrf]){
                        int l2 = l2_rad_int.l2;

                        double_complex prefac = std::pow(double_complex(0.0, -1.0), l2);

//                        std::cout<<"l1l2 = "<<l1<<" "<<l2<<std::endl;

                        for (int m1 = -l1; m1 <= l1; m1++){
                            for (int m2 = -l2; m2 <= l2; m2++){
                                double_complex full_prefact = prefac * const_prefac;
                                beta_besselJ_gaunt_coefs_t gc{
                                    l1,m1,l2,m2,
                                    {
                                             -full_prefact * SHT::gaunt_rlm(l1, 1, l2,  m1, 1, m2  ),     // for x component
                                             -full_prefact * SHT::gaunt_rlm(l1, 1, l2,  m1, -1, m2 ),     // for y component
                                              full_prefact * SHT::gaunt_rlm(l1, 1, l2,  m1, 0, m2  )      // for z component
                                    }
                                };
//                                std::cout<<"m1m2 = "<<m1<<" "<<m2<<std::endl;
//                                for(int comp: {0,1,2}) std::cout<<gc.prefac_gaunt_coefs[comp]<<" ";
//                                std::cout<<std::endl;

                                bj_gc.push_back(std::move(gc));
                            }
                        }
                    }

                    beta_besselJ_gaunt_coefs.push_back(std::move(bj_gc));
                }

                // iterate over gk vectors
                //#pragma omp parallel for
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
                                    for(size_t comp = 0; comp < nu_; comp++){
                                        beta_gk_t_(igkloc, atom_type.offset_lo() + xi, comp) += radint *
                                                gkvec_rlm[ Utils::lm_by_l_m(l2, m2) ] * bj_gc[ m1l2m2_it ].prefac_gaunt_coefs[comp];
                                        //std::cout<<m1l2m2_it<<" ";
                                    }
                                }
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

        void init_beta_gk()
        {
            auto& unit_cell = ctx_->unit_cell();
            int num_gkvec_loc = bp_->num_gkvec_loc();

            // TODO maybe remove it
            for(size_t i = 0; i < this->num_; i++)
            {
                components_gk_a_[i].zero();
            }

            //#pragma omp parallel for
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


                // TODO: need to optimize order of loops
                // calc beta lattice gradient
                for (int xi = 0; xi < unit_cell.atom(ia).mt_lo_basis_size(); xi++) {
                    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++) {
                        int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
                        // TODO:store vectors before
                        auto Gcart = bp_->gk_vectors().gkvec_cart(igk);

                        // iteration over tensor components
                        for(size_t u = 0; u < nu_; u++){
                            for(size_t v = 0; v < nv_; v++){
                                // complicate formula
                                components_gk_a_[ind(u,v)](igkloc, unit_cell.atom(ia).offset_lo() + xi) =
                                        // first
                                        beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, v) *
                                        phase_gk[igkloc] * Gcart[u] * double_complex(0.0 , 1.0);
                            }
                        }

                        // third
                        for(size_t u = 0; u < nu_; u++){
                            components_gk_a_[ind(u,u)](igkloc, unit_cell.atom(ia).offset_lo() + xi) -=
                                    bp_->beta_gk_a()(igkloc, unit_cell.atom(ia).offset_lo() + xi) * 0.5;
                        }
//                        #pragma omp critical
//                        {
//                            std::cout<<"test non-local stress:"<<std::endl;
//
//                                std::cout<<"G beta_gt= "<<Gcart<<"  |   ";
//                                for(int j=0; j<3; j++){
//                                    std::cout<<beta_gk_t_(igkloc, unit_cell.atom(ia).type().offset_lo() + xi, j) <<"  ";
//                                }
//                                std::cout<<std::endl;
//
//
//                            for(int i=0; i<3; i++){
//                                for(int j=0; j<3; j++){
//                                    std::cout<<components_gk_a_[ind(i,j)](igkloc, unit_cell.atom(ia).offset_lo() + xi) <<"  ";
//                                }
//                                std::cout<<std::endl;
//                            }
//
//                        }
                    }
                }
            }
        }

        void init_beta_gk_t2()
        {
            int num_beta_t = bp_->num_beta_by_atom_types();
            auto& unit_cell = ctx_->unit_cell();

            /* allocate array */
            beta_gk_t_ = mdarray<double_complex, 3>(bp_->num_gkvec_loc(), num_beta_t, this->num_);
            //beta_gk_t_.zero();

            double fourpi_omega = fourpi / std::sqrt(unit_cell.omega());

            /* compute <G+k|beta> */
            #pragma omp parallel for
            for (int igkloc = 0; igkloc < bp_->num_gkvec_loc(); igkloc++) {
                int igk   = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igkloc;
                auto gk_cart = bp_->gk_vectors().gkvec_cart(igk);
                double gk_length = gk_cart.length();

                /* vs = {r, theta, phi} */
                auto vs = SHT::spherical_coordinates(gk_cart);

                /* compute real spherical harmonics for G+k vector */
                std::vector<double> gkvec_rlm_deriv_theta(Utils::lmmax(bp_->lmax_beta()));
                std::vector<double> gkvec_rlm_deriv_phi(Utils::lmmax(bp_->lmax_beta()));

                SHT::spherical_harmonics_deriv_theta(bp_->lmax_beta(), vs[1], vs[2], &gkvec_rlm_deriv_theta[0]);
                SHT::spherical_harmonics_deriv_phi(bp_->lmax_beta(), vs[1], vs[2], &gkvec_rlm_deriv_phi[0]);

                double cos_th = std::cos(vs[1]);
                double cos_ph = std::cos(vs[2]);
                double sin_th = std::sin(vs[1]);
                double sin_ph = std::sin(vs[2]);

                if(igk != 0 && std::abs(vs[1]) > 10e-9){
                    for (int iat = 0; iat < unit_cell.num_atom_types(); iat++) {
                        auto& atom_type = unit_cell.atom_type(iat);

                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            double_complex prefac_djldq = std::pow(double_complex(0, -1), l) * fourpi_omega *
                                    ctx_->radial_integrals().beta_djldq_radial_integral(idxrf, iat, gk_length) /gk_length;

                            double_complex prefac = std::pow(double_complex(0, -1), l) * fourpi_omega *
                                    ctx_->radial_integrals().beta_radial_integral(idxrf, iat, gk_length) / gk_length;

                            vector3d<double_complex> rlm_grad (
                                    prefac * (-gkvec_rlm_deriv_theta[lm] * cos_th * cos_ph + gkvec_rlm_deriv_phi[lm] * sin_ph / sin_th ) ,
                                    prefac * (-gkvec_rlm_deriv_theta[lm] * cos_th * sin_ph - gkvec_rlm_deriv_phi[lm] * cos_ph / sin_th ) ,
                                    prefac * gkvec_rlm_deriv_theta[lm] * sin_th
                                    );

                            for(size_t u = 0; u < nu_; u++){
                                for(size_t v = 0; v < nv_; v++){
                                    beta_gk_t_(igkloc, atom_type.offset_lo() + xi, ind(u,v)) =
                                            prefac_djldq * gk_cart[u] * gk_cart[v] + gk_cart[u] * rlm_grad[v];
                                }
                            }
                        }
                    }
                }

                if(igk != 0 && std::abs(vs[1]) < 10e-9){
                    for (int iat = 0; iat < unit_cell.num_atom_types(); iat++) {
                        auto& atom_type = unit_cell.atom_type(iat);


                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            double_complex prefac_djldq = std::pow(double_complex(0, -1), l) * fourpi_omega *
                                    ctx_->radial_integrals().beta_djldq_radial_integral(idxrf, iat, gk_length) /gk_length;

                            for(size_t u = 0; u < nu_; u++){
                                for(size_t v = 0; v < nv_; v++){
                                    beta_gk_t_(igkloc, atom_type.offset_lo() + xi, ind(u,v)) =
                                            prefac_djldq * gk_cart[u] * gk_cart[v];
                                }
                            }
                        }
                    }
                }

                if(igk == 0 ){
                    for (int iat = 0; iat < unit_cell.num_atom_types(); iat++) {
                        auto& atom_type = unit_cell.atom_type(iat);

                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            for(size_t u = 0; u < nu_; u++){
                                for(size_t v = 0; v < nv_; v++){
                                    beta_gk_t_(igkloc, atom_type.offset_lo() + xi, ind(u,v)) = double_complex(0.0, 0.0);
                                }
                            }
                        }
                    }
                }
            }
        }

        void init_beta_gk2()
        {
            auto& unit_cell = ctx_->unit_cell();
            int num_gkvec_loc = bp_->num_gkvec_loc();

//            auto cell_matrix = unit_cell.lattice_vectors();
//            auto inv_cell_matrix = unit_cell.reciprocal_lattice_vectors();

            #pragma omp for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
                auto vk = bp_->gk_vectors().vk();
                double phase = twopi * ( vk * unit_cell.atom(ia).position());
                double_complex phase_k = std::exp(double_complex(0.0, phase));

                std::vector<double_complex> phase_gk(num_gkvec_loc);
                for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
                    int igk = bp_->gk_vectors().gvec_offset(bp_->comm().rank()) + igk_loc;
                    auto G = bp_->gk_vectors().gvec(igk);
                    phase_gk[igk_loc] = std::conj(ctx_->gvec_phase_factor(G, ia) * phase_k);
                }

                // cartesian atomic coordinate and k-vector
//                auto r_cart = cell_matrix * unit_cell.atom(ia).position();
//                auto vk_cart = inv_cell_matrix * vk;

                for (int xi = 0; xi < unit_cell.atom(ia).mt_lo_basis_size(); xi++) {
                    for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
                        for(size_t u = 0; u < nu_; u++){
                            for(size_t v = 0; v < nv_; v++){
                                components_gk_a_[ind(u,v)](igk_loc, unit_cell.atom(ia).offset_lo() + xi) =
                                        beta_gk_t_(igk_loc, unit_cell.atom(ia).type().offset_lo() + xi, ind(u,v)) * phase_gk[igk_loc];
                            }
                        }

                        for(size_t u = 0; u < nu_; u++){
                            components_gk_a_[ind(u,u)](igk_loc, unit_cell.atom(ia).offset_lo() + xi) -=
                                    0.5 * bp_->beta_gk()(igk_loc, unit_cell.atom(ia).offset_lo() + xi);
                        }
                    }
                }
            }
        }
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
