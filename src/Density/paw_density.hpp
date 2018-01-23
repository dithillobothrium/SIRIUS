/*
 * paw_density.cpp
 *
 *  Created on: Oct 24, 2016
 *      Author: isivkov
 */

/**
 *
 * Initialize PAW variables All-electron (AW) and pseudo (PS) density arrays.
 * In case of spin orbit a small component of AE density is also initialized
 */

inline void Density::init_paw()
{
    paw_density_data_.clear();

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        int ia_paw = unit_cell_.spl_num_paw_atoms(i);
        int ia = unit_cell_.paw_atom_index(ia_paw);
        auto& atom = unit_cell_.atom(ia);
        auto& atom_type = atom.type();

        int l_max = 2 * atom_type.indexr().lmax_lo();
        int lm_max_rho = Utils::lmmax(l_max);

        paw_density_data_t pdd;

        pdd.atom_ = &atom;

        pdd.ia = ia;

        /* allocate density arrays , 1,2 or 4 components*/
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            pdd.ae_density_.push_back(Spheric_function<spectral, double>(lm_max_rho, pdd.atom_->radial_grid()));
            pdd.ps_density_.push_back(Spheric_function<spectral, double>(lm_max_rho, pdd.atom_->radial_grid()));

            pdd.ae_density_tp_.push_back(Spheric_function<spatial, double>(sht_->num_points(), pdd.atom_->radial_grid()));
            pdd.ps_density_tp_.push_back(Spheric_function<spatial, double>(sht_->num_points(), pdd.atom_->radial_grid()));

            /* Spin-Orbit */
            if (atom.type().pp_desc().spin_orbit_coupling) {
                /* add 4-comp density from small component*/
                pdd.ae_rel_small_density_.push_back(Spheric_function<spectral, double>(lm_max_rho, pdd.atom_->radial_grid()));
            }
        }

        paw_density_data_.push_back(std::move(pdd));
    }
}

/**
 * Initializes Density Matrix as in Quantum Espresso
 */
inline void Density::init_density_matrix_for_paw()
{
    density_matrix_.zero();

    for (int ipaw = 0; ipaw < unit_cell_.num_paw_atoms(); ipaw++ ) {
        int ia = unit_cell_.paw_atom_index(ipaw);

        auto& atom = unit_cell_.atom(ia);
        auto& atom_type = atom.type();

        int nbf = atom_type.mt_basis_size();

        auto& occupations = atom_type.pp_desc().occupations;

        // magnetization vector
        vector3d<double> magn = atom.vector_field();

        for (int xi = 0; xi < nbf; xi++)
        {
            basis_function_index_descriptor const& basis_func_index_dsc = atom_type.indexb()[xi];

            int rad_func_index = basis_func_index_dsc.idxrf;

            double occ = occupations[rad_func_index];

            int l = basis_func_index_dsc.l;

            switch (ctx_.num_mag_dims()){
                case 0:{
                    density_matrix_(xi,xi,0,ia) = occ / (double)( 2 * l + 1 );
                    break;
                }

                case 3:
                case 1:{
                    double nm = ( std::abs(magn[2]) < 1. ) ? magn[2] :  std::copysign(1, magn[2] ) ;

                    density_matrix_(xi,xi,0,ia) = 0.5 * (1.0 + nm ) * occ / (double)( 2 * l + 1 );
                    density_matrix_(xi,xi,1,ia) = 0.5 * (1.0 - nm ) * occ / (double)( 2 * l + 1 );
                    break;
                }
            }

        }
    }
}

/**
 * Generates PAW all-electron(AE) and pseudo(PS) density in \f$lm\f$ components using density matrix \f$\rho_{ij}\f$,
 * Gaunt coefficients and radial AE wave functions \f$\phi_i\f$.
 * The variable \f$\phi_{l_i}(r) \phi_{l_j}(r)\f$ is precalculated and stored during reading from pseudopotential file
 *
 * In the case of non-collinear calculation density is calculated as 4-components variable  \f$ \rho^{\alpha} = (\rho, m_x, m_y, m_z),\ \alpha=0,1,2,3\f$
 * While density matrix is stored as \f$(\rho^{00}_{ij}, \rho^{11}_{ij}, \rho^{10}_{ij}, \rho^{01}_{ij})\f$
 *
 * - AE density
 * \f[
 *      n^{\alpha}(\bf r ) \rightarrow n^{alpha}_{lm}(r) = \sum_{ij} \rho^{\alpha}_{ij} Gaunt(l_i, l, l_j\ |\ m_i, m, m_j) \phi_{l_i}(r) \phi_{l_j}(r)
 * \f]
 *
 * - PS density together with compensation charge
 *
 * \f[
 *      \tilde{n}^{\alpha}(\bf r ) + \hat{n}^{\alpha}(\bf r)  \rightarrow \tilde{n}^{\alpha}_{lm}(r) = \sum_{ij} \rho^{\alpha}_{ij} Gaunt(l_i, l, l_j\ |\ m_i, m, m_j)  ( \tilde\phi_{l_i}(r) \tilde\phi_{l_j}(r)  + Q^l_{l_i,l_j}(r) )
 * \f]
 *
 * - Densities are converted from \f$lm\f$ representation to \f$\theta-\phi\f$ representation to use later for XC potential calculation
 *
 * - In the case of spin orbit a contribution from relativistic small components is calculated in the same way
 *
 * \f[
 *      \eta^{\alpha}(\bf r ) \rightarrow \eta^{\alpha}_{lm}(r) = \sum_{ij} \rho^{\alpha}_{ij} Gaunt(l_i, l, l_j\ |\ m_i, m, m_j) \chi_{l_i}(r) \chi_{l_j}(r)
 * \f]
 *
 * Also, in the case of spin orbit and non-collinearity (not they are alway together) we need to compute another
 * contribution to magnetization from small component in \f$\theta-\phi\f$ representation
 * (which is non-zero in the non-collinear magnetization, while spin-orbit can excist in the collinear case in theory)
 *
 *  Therefore, write total expression for AE density with spin orbit
 * \f[
 *      n_{tot}^{a}(\bf{r} ) = n^{a}(\bf{r} ) + \eta^{a}(\bf r ) - 2 \bf r_a \sum_{b}  \bf r_b \eta^{b}(\bf r) \\
 *      n_{tot}^{0}(\bf{r} ) = n^{a}(\bf{r} ) + \eta^{a}(\bf r )
 * \f]
 *
 * \f$a,b=1,2,3\ or\ x,y,z\f$
 */
inline void Density::generate_paw_atom_density(paw_density_data_t& pdd)
{
    int ia = pdd.ia;

    auto& atom_type = pdd.atom_->type();
    auto& pp_desc = atom_type.pp_desc();

    auto l_by_lm = Utils::l_by_lm(2 * atom_type.indexr().lmax_lo());

    // get gaunt coefficients
    Gaunt_coefficients<double> GC(atom_type.indexr().lmax_lo(),
                                  2 * atom_type.indexr().lmax_lo(),
                                  atom_type.indexr().lmax_lo(),
                                  SHT::gaunt_rlm);

    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        pdd.ae_density_[i].zero();
        pdd.ps_density_[i].zero();

        /* Spin-Orbit */
        if (atom_type.pp_desc().spin_orbit_coupling) {
            pdd.ae_rel_small_density_[i].zero();
        }
    }

    /* get radial grid to divide density over r^2 */
    auto &grid = atom_type.radial_grid();
    std::vector<double> inv_r2_grid();

    /* iterate over local basis functions (or over lm1 and lm2) */
    for (int ib2 = 0; ib2 < atom_type.indexb().size(); ib2++){
        for(int ib1 = 0; ib1 <= ib2; ib1++){

            // get lm quantum numbers (lm index) of the basis functions
            int lm2 = atom_type.indexb(ib2).lm;
            int lm1 = atom_type.indexb(ib1).lm;

            //get radial basis functions indices
            int irb2 = atom_type.indexb(ib2).idxrf;
            int irb1 = atom_type.indexb(ib1).idxrf;

            // index to iterate Qij,
            int iqij = irb2 * (irb2 + 1) / 2 + irb1;

            // get num of non-zero GC
            int num_non_zero_gk = GC.num_gaunt(lm1,lm2);

            double diag_coef = ib1 == ib2 ? 1. : 2. ;

            /* store density matrix in aux form */
            double dm[4]={0,0,0,0};
            switch (ctx_.num_mag_dims()) {
                case 3: {
                    dm[2] =  2 * std::real(density_matrix_(ib1, ib2, 2, ia));
                    dm[3] = -2 * std::imag(density_matrix_(ib1, ib2, 2, ia));
                }
                case 1: {
                    dm[0] = std::real(density_matrix_(ib1, ib2, 0, ia) + density_matrix_(ib1, ib2, 1, ia));
                    dm[1] = std::real(density_matrix_(ib1, ib2, 0, ia) - density_matrix_(ib1, ib2, 1, ia));
                    break;
                }
                case 0: {
                    dm[0] = density_matrix_(ib1, ib2, 0, ia).real();
                    break;
                }
                default:{
                    TERMINATE("generate_paw_atom_density FATAL ERROR!");
                    break;
                }
            }

            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
                /* add nonzero coefficients */
                for(int inz = 0; inz < num_non_zero_gk; inz++){
                    auto& lm3coef = GC.gaunt(lm1,lm2,inz);
                    /* iterate over radial points */
                    for(int irad = 0; irad < (int)grid.num_points(); irad++){
                        /* we need to divide density over r^2 since wave functions are stored multiplied by r */
                        double prefac = dm[imagn] * diag_coef * grid.x_inv(irad) * grid.x_inv(irad) * lm3coef.coef;
                        /* calculate unified density/magnetization
                         * dm_ij * GauntCoef * ( phi_i phi_j  +  Q_ij) */
                        pdd.ae_density_[imagn](lm3coef.lm3, irad) += prefac * pp_desc.all_elec_wfc_matrix(irad, iqij);
                        pdd.ps_density_[imagn](lm3coef.lm3, irad) += prefac * (pp_desc.pseudo_wfc_matrix(irad, iqij) +
                                pp_desc.q_radial_functions_l(irad, iqij, l_by_lm[lm3coef.lm3]));
                    }
                    /* in case of spin-orbit add small component to 4-component density */
                    if (atom_type.pp_desc().spin_orbit_coupling) {
                        /* iterate over radial points */
                        for(int irad = 0; irad < (int)grid.num_points(); irad++){
                            pdd.ae_rel_small_density_[imagn](lm3coef.lm3, irad) += dm[imagn] * diag_coef * grid.x_inv(irad) * grid.x_inv(irad) *
                                    lm3coef.coef * pp_desc.all_elec_rel_small_wfc_matrix(irad, iqij);
                        }
                    }
                }
            }
        }
    }

    for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
        pdd.ae_density_tp_[imagn] = transform(sht_.get(), pdd.ae_density_[imagn]);
        pdd.ps_density_tp_[imagn] = transform(sht_.get(), pdd.ps_density_[imagn]);
    }

    /* in case of spin-orbit if we have 3 spin components, we also need to add component to magnetization in spatial representation */
    /* lm representation will be used only in non-magnetic case */
    if (atom_type.pp_desc().spin_orbit_coupling && ctx_.num_mag_dims() == 3) {
        /* over magnetic components */
        for (int imagn = 0; imagn < 3 ; imagn++) {
            auto ae_rel_small_magn_comp_tp = transform(sht_.get(), pdd.ae_rel_small_density_[imagn + 1]);
            /* over theta phi */
            for (int itp = 0; itp < sht_->num_points(); itp++) {
                /* Cartesian coords of the point */
                auto coord = sht_->coord(itp);
                /* over radial part */
                for (int jmagn = 0; jmagn < 3; jmagn++) {
                    for (int irad = 0; irad < (int)grid.num_points(); irad++) {
                        pdd.ae_density_tp_[jmagn + 1](itp, irad) -= 2.0 * ae_rel_small_magn_comp_tp(itp, irad) * coord[jmagn] * coord[imagn];
                    }
                }
            }
        }
    }
}

inline void Density::generate_paw_loc_density()
{
    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    PROFILE("sirius::Density::generate_paw_loc_density");

    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        generate_paw_atom_density(paw_density_data_[i]);
    }
}

