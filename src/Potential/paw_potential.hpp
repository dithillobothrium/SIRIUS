/*
 *  Created on: May 3, 2016
 *      Author: isivkov
 */

inline void Potential::init_PAW()
{
    paw_potential_data_.clear();
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

        paw_potential_data_t ppd;

        ppd.atom_ = &atom;

        ppd.ia = ia;

        ppd.ia_paw = ia_paw;

        /* allocate potential arrays */
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            ppd.ae_potential_.push_back(Spheric_function<spectral, double>(lm_max_rho, ppd.atom_->radial_grid()));
            ppd.ps_potential_.push_back(Spheric_function<spectral, double>(lm_max_rho, ppd.atom_->radial_grid()));
        }

        ppd.core_energy_ = atom_type.pp_desc().core_energy;

        paw_potential_data_.push_back(std::move(ppd));
    }

    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia = unit_cell_.paw_atom_index(i);
        int bs = unit_cell_.atom(ia).mt_basis_size();
        max_paw_basis_size_ = std::max(max_paw_basis_size_, bs);
    }
    
    /* initialize dij matrix */
    paw_dij_ = mdarray<double, 4>(max_paw_basis_size_, max_paw_basis_size_, ctx_.num_mag_dims() + 1, unit_cell_.num_paw_atoms(),
                                  memory_t::host, "paw_dij_");

    /* allocate PAW energy array */
    paw_hartree_energies_.resize(unit_cell_.num_paw_atoms());
    paw_xc_energies_.resize(unit_cell_.num_paw_atoms());
    paw_core_energies_.resize(unit_cell_.num_paw_atoms());
    paw_one_elec_energies_.resize(unit_cell_.num_paw_atoms());
}

inline void Potential::generate_PAW_effective_potential(Density const& density)
{
    PROFILE("sirius::Potential::generate_PAW_effective_potential");

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    /* zero PAW arrays */
    std::fill(paw_one_elec_energies_.begin(), paw_one_elec_energies_.end(), 0.0);
    std::fill(paw_hartree_energies_.begin(), paw_hartree_energies_.end(), 0.0);
    std::fill(paw_xc_energies_.begin(), paw_xc_energies_.end(), 0.0);

    /* zero Dij */
    paw_dij_.zero();

    /* calculate xc and hartree for atoms without OpenMP since it is used inside*/
    for(int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        calc_PAW_local_potential(paw_potential_data_[i],
                                 density.ae_paw_atom_density(i),
                                 density.ps_paw_atom_density(i));
    }


    /* calculate PAW Dij matrix */
    #pragma omp parallel for
    for(int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        calc_PAW_local_Dij(paw_potential_data_[i], paw_dij_);

        calc_PAW_one_elec_energy(paw_potential_data_[i], density.density_matrix(), paw_dij_);
    }

    // collect Dij and add to atom d_mtrx
    comm_.allreduce(&paw_dij_(0, 0, 0, 0), static_cast<int>(paw_dij_.size()));

    if (ctx_.control().print_checksum_ && comm_.rank() == 0) {
        auto cs = paw_dij_.checksum();
        print_checksum("paw_dij", cs);
    }

    // add paw Dij to uspp Dij
    add_paw_Dij_to_atom_Dmtrx();

    // calc total energy
    double energies[] = {0.0, 0.0, 0.0, 0.0};

    for(int ia = 0; ia < unit_cell_.spl_num_paw_atoms().local_size(); ia++) {
        energies[0] += paw_potential_data_[ia].hartree_energy_;
        energies[1] += paw_potential_data_[ia].xc_energy_;
        energies[2] += paw_potential_data_[ia].one_elec_energy_;
        energies[3] += paw_potential_data_[ia].core_energy_;  // it is light operation
    }

    comm_.allreduce(&energies[0], 4);

    paw_hartree_total_energy_ = energies[0];
    paw_xc_total_energy_ = energies[1];
    paw_one_elec_energy_ = energies[2];
    paw_total_core_energy_ = energies[3];
}

inline double Potential::xc_mt_PAW_nonmagnetic(Spheric_function<spectral, double>& full_potential,
                                               Spheric_function<spectral, double> const& full_density,
                                               std::vector<double> const& rho_core)
{
    int lmmax = static_cast<int>(full_density.size(0));

    Radial_grid<double> const& rgrid = full_density.radial_grid();

    /* new array to store core and valence densities */
    Spheric_function<spectral, double> full_rho_lm_sf_new(lmmax, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += full_density;

    double invY00 = 1. / y00 ;

    /* adding core part */
    for(int ir = 0; ir < rgrid.num_points(); ir++ )
    {
        full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
    }

    Spheric_function<spatial,double> full_rho_tp_sf = transform(sht_.get(), full_rho_lm_sf_new);

    // create potential in theta phi
    Spheric_function<spatial,double> vxc_tp_sf(sht_->num_points(), rgrid);

    // create energy in theta phi
    Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

    xc_mt_nonmagnetic(rgrid, xc_func_, full_rho_lm_sf_new, full_rho_tp_sf, vxc_tp_sf, exc_tp_sf);

    full_potential += transform(sht_.get(), vxc_tp_sf);

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<spectral,double> exc_lm_sf = transform(sht_.get(), exc_tp_sf );

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}

inline double Potential::xc_mt_PAW_collinear(std::vector<Spheric_function<spectral, double>>& potential,
                                             std::vector<Spheric_function<spectral, double>> const& density,
                                             std::vector<double> const& rho_core)
{
    int lmsize_rho = static_cast<int>(density[0].size(0));

    Radial_grid<double> const& rgrid = density[0].radial_grid();

    /* new array to store core and valence densities */
    Spheric_function<spectral, double> full_rho_lm_sf_new(lmsize_rho, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += density[0];

    double invY00 = 1 / y00;

    /* adding core part */
    for (int ir = 0; ir < rgrid.num_points(); ir++ ) {
        full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
    }

    // calculate spin up spin down density components in lm components
    // up = 1/2 ( rho + magn );  down = 1/2 ( rho - magn )
    Spheric_function<spectral, double> rho_u_lm_sf = 0.5 * (full_rho_lm_sf_new + density[1]);
    Spheric_function<spectral, double> rho_d_lm_sf = 0.5 * (full_rho_lm_sf_new - density[1]);

    // transform density to theta phi components
    Spheric_function<spatial, double> rho_u_tp_sf = transform(sht_.get(), rho_u_lm_sf );
    Spheric_function<spatial, double> rho_d_tp_sf = transform(sht_.get(), rho_d_lm_sf );

    // create potential in theta phi
    Spheric_function<spatial, double> vxc_u_tp_sf(sht_->num_points(), rgrid);
    Spheric_function<spatial, double> vxc_d_tp_sf(sht_->num_points(), rgrid);

    // create energy in theta phi
    Spheric_function<spatial, double> exc_tp_sf(sht_->num_points(), rgrid);

    // calculate XC
    xc_mt_magnetic(rgrid, xc_func_,
                   rho_u_lm_sf, rho_u_tp_sf,
                   rho_d_lm_sf, rho_d_tp_sf,
                   vxc_u_tp_sf, vxc_d_tp_sf,
                   exc_tp_sf);

    // transform back in lm
    potential[0] += transform(sht_.get(), 0.5 * (vxc_u_tp_sf + vxc_d_tp_sf) );
    potential[1] += transform(sht_.get(), 0.5 * (vxc_u_tp_sf - vxc_d_tp_sf));

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<spectral,double> exc_lm_sf = transform(sht_.get(), exc_tp_sf );

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}


inline double Potential::xc_mt_PAW_noncollinear(std::vector<Spheric_function<spectral, double>>& potential,
                            std::vector<Spheric_function<spectral, double>> const& density,
                            std::vector<double> const& rho_core)
{
    if (density.size() != 4 || potential.size() != 4){
       TERMINATE("xc_mt_PAW_noncollinear FATAL ERROR!")
    }

    Radial_grid<double> const& rgrid = density[0].radial_grid();

    /* transform density to theta phi components */
    std::vector<Spheric_function<spatial, double>> rho_tp(density.size());

    for (size_t i = 0; i < density.size(); i++){
        rho_tp[i] = transform(sht_.get(), density[i]);
    }

    /* transform 4D magnetization to spin-up, spin-down form (correct for LSDA)  rho ± |magn| */
    Spheric_function<spatial, double> rho_u_tp(sht_->num_points(), rgrid);
    Spheric_function<spatial, double> rho_d_tp(sht_->num_points(), rgrid);

    //#pragma omp parallel for
    for (int ir = 0; ir < rgrid.num_points(); ir++ ) {
        for (int itp = 0; itp < sht_->num_points(); itp++ ) {
            vector3d<double> magn({rho_tp[1](itp, ir), rho_tp[2](itp, ir), rho_tp[3](itp, ir)});
            double norm = magn.length();

            rho_u_tp(itp, ir) = 0.5 * (rho_tp[0](itp, ir) + rho_core[ir] + norm);
            rho_d_tp(itp, ir) = 0.5 * (rho_tp[0](itp, ir) + rho_core[ir] - norm);
        }
    }

    /* in lm representation */
    Spheric_function<spectral, double> rho_u_lm = transform(sht_.get(), rho_u_tp);
    Spheric_function<spectral, double> rho_d_lm = transform(sht_.get(), rho_d_tp);

    // allocate potential in theta phi
    Spheric_function<spatial, double> vxc_u_tp(sht_->num_points(), rgrid);
    Spheric_function<spatial, double> vxc_d_tp(sht_->num_points(), rgrid);

    // allocate energy in theta phi
    Spheric_function<spatial, double> exc_tp(sht_->num_points(), rgrid);

    // calculate XC
    xc_mt_magnetic(rgrid, xc_func_,
                   rho_u_lm, rho_u_tp,
                   rho_d_lm, rho_d_tp,
                   vxc_u_tp, vxc_d_tp,
                   exc_tp);

    /* allocate 4D potential in theta phi components */
    std::vector<Spheric_function<spatial, double>> vxc_tp;

    /* allocate the rest */
    for (size_t i = 0; i < potential.size(); i++){
        vxc_tp.push_back(Spheric_function<spatial, double>(sht_->num_points(), rgrid));
    }

    /* transform back potential from up/down to 4D form*/
    //#pragma omp parallel for
    for (int ir = 0; ir < rgrid.num_points(); ir++ ) {
        for (int itp = 0; itp < sht_->num_points(); itp++ ) {
            /* get total potential and field abs value*/
            double pot   = 0.5 * (vxc_u_tp(itp, ir) + vxc_d_tp(itp, ir));
            double field = 0.5 * (vxc_u_tp(itp, ir) - vxc_d_tp(itp, ir));

            /* get unit magnetization vector*/
            vector3d<double> magn({rho_tp[1](itp, ir), rho_tp[2](itp, ir), rho_tp[3](itp, ir)});
            double norm = magn.length();
            magn = magn * (norm > 0.0 ? field / norm : 0.0) ;

            /* add total potential and effective field values at current point */
            vxc_tp[0](itp, ir) = pot;
            for (int i: {0,1,2}) {
                vxc_tp[i+1](itp, ir) = magn[i];
            }
        }
    }

    // transform back in lm
    for(size_t i = 0; i < density.size(); i++){
        potential[i] += transform(sht_.get(), vxc_tp[i]);
    }

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<spectral,double> exc_lm = transform(sht_.get(), exc_tp );

    return inner(exc_lm, rho_u_lm + rho_d_lm);
}


inline double Potential::calc_PAW_hartree_potential(Atom& atom,
                                                    Spheric_function<spectral, double> const& full_density,
                                                    Spheric_function<spectral, double>& full_potential)
{
    int lmsize_rho = static_cast<int>(full_density.size(0));

    const Radial_grid<double>& grid = full_density.radial_grid();

    // array passed to poisson solver
    Spheric_function<spectral, double> atom_pot_sf(lmsize_rho, grid);
    atom_pot_sf.zero();

    poisson_vmt<true>(atom, full_density, atom_pot_sf);

    /* add hartree contribution */
    full_potential += atom_pot_sf;

    //---------------------
    //--- calc energy ---
    //---------------------
    auto l_by_lm = Utils::l_by_lm( Utils::lmax_by_lmmax(lmsize_rho) );

    double hartree_energy=0.0;

    #pragma omp parallel reduction(+:hartree_energy)
    {
        /* create array for integration */
        std::vector<double> intdata(grid.num_points(),0);

        #pragma omp for
        for (int lm=0; lm < lmsize_rho; lm++){
            /* fill data to integrate */
            for (int irad = 0; irad < grid.num_points(); irad++){
                intdata[irad] = full_density(lm,irad) * full_potential(lm, irad) * grid[irad] * grid[irad];
            }

            /* create spline from the data */
            Spline<double> h_spl(grid,intdata);
            hartree_energy += 0.5 * h_spl.integrate(0);
        }
    }

    return hartree_energy;
}

inline void Potential::calc_PAW_local_potential(paw_potential_data_t &ppd,
                                                std::vector<Spheric_function<spectral, double>> const& ae_density,
                                                std::vector<Spheric_function<spectral, double>> const& ps_density)
{
    auto& pp_desc = ppd.atom_->type().pp_desc();

    //-----------------------------------------
    //---- Calculation of Hartree potential ---
    //-----------------------------------------

    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        ppd.ae_potential_[i].zero();
        ppd.ps_potential_[i].zero();
    }


    double ae_hartree_energy = calc_PAW_hartree_potential(*ppd.atom_,
                                                          ae_density[0],
                                                          ppd.ae_potential_[0]);

    double ps_hartree_energy = calc_PAW_hartree_potential(*ppd.atom_,
                                                          ps_density[0],
                                                          ppd.ps_potential_[0]);

    ppd.hartree_energy_ = ae_hartree_energy - ps_hartree_energy;

    //-----------------------------------------
    //---- Calculation of XC potential ---
    //-----------------------------------------
    double ae_xc_energy = 0.0;
    double ps_xc_energy = 0.0;

    switch (ctx_.num_mag_dims()){
        case 0:{
            ae_xc_energy = xc_mt_PAW_nonmagnetic(ppd.ae_potential_[0], ae_density[0], pp_desc.all_elec_core_charge);
            ps_xc_energy = xc_mt_PAW_nonmagnetic(ppd.ps_potential_[0], ps_density[0], pp_desc.core_charge_density);
        }break;

        case 1:{
            ae_xc_energy = xc_mt_PAW_collinear(ppd.ae_potential_, ae_density, pp_desc.all_elec_core_charge);
            ps_xc_energy = xc_mt_PAW_collinear(ppd.ps_potential_, ps_density, pp_desc.core_charge_density);
        }break;

        case 3:{
            ae_xc_energy = xc_mt_PAW_noncollinear(ppd.ae_potential_, ae_density, pp_desc.all_elec_core_charge);
            ps_xc_energy = xc_mt_PAW_noncollinear(ppd.ps_potential_, ps_density, pp_desc.core_charge_density);
        }break;

        default:{
            TERMINATE("PAW local potential error! Wrong number of spins!")
        }break;
    }

    /* save xc energy in pdd structure */
    ppd.xc_energy_ = ae_xc_energy - ps_xc_energy;
}

inline void Potential::calc_PAW_local_Dij(paw_potential_data_t &pdd, mdarray<double,4>& paw_dij)
{
    int paw_ind = pdd.ia_paw;

    auto& atom_type = pdd.atom_->type();

    auto& pp_desc = atom_type.pp_desc();

    /* get lm size for density */
    int lmax = atom_type.indexr().lmax_lo();
    int lmsize_rho = Utils::lmmax(2 * lmax);

    auto l_by_lm = Utils::l_by_lm(2 * lmax);

    Gaunt_coefficients<double> GC(lmax, 2 * lmax, lmax, SHT::gaunt_rlm);

    /* store integrals here */
    mdarray<double, 3> integrals(lmsize_rho, pp_desc.num_beta_radial_functions * (pp_desc.num_beta_radial_functions + 1) / 2,
                                 ctx_.num_mag_dims() + 1);

    for(int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++ ){
        auto &ae_atom_pot = pdd.ae_potential_[imagn];
        auto &ps_atom_pot = pdd.ps_potential_[imagn];

        for (int irb2 = 0; irb2 < pp_desc.num_beta_radial_functions; irb2++){
            for (int irb1 = 0; irb1 <= irb2; irb1++){
                int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

                Radial_grid<double> newgrid = atom_type.radial_grid().segment(pp_desc.cutoff_radius_index);

                // create array for integration
                std::vector<double> intdata(newgrid.num_points(),0);

                for (int lm3 = 0; lm3 < lmsize_rho; lm3++) {
                    // fill array
                    for (int irad = 0; irad < newgrid.num_points(); irad++) {
                        double ae_part = pp_desc.all_elec_wfc(irad,irb1) * pp_desc.all_elec_wfc(irad,irb2);
                        double ps_part = pp_desc.pseudo_wfc(irad,irb1) * pp_desc.pseudo_wfc(irad,irb2)  + pp_desc.q_radial_functions_l(irad,iqij,l_by_lm[lm3]);

                        intdata[irad] = ae_atom_pot(lm3,irad) * ae_part - ps_atom_pot(lm3,irad) * ps_part;
                    }

                    // create spline from data arrays
                    Spline<double> dij_spl(newgrid,intdata);

                    // integrate
                    integrals(lm3, iqij, imagn) = dij_spl.integrate(0);
                }

            }
        }
    }

    //---- calc Dij ----
    for (int ib2 = 0; ib2 < atom_type.mt_lo_basis_size(); ib2++) {
        for (int ib1 = 0; ib1 <= ib2; ib1++) {
            //int idij = (ib2 * (ib2 + 1)) / 2 + ib1;

            // get lm quantum numbers (lm index) of the basis functions
            int lm1 = atom_type.indexb(ib1).lm;
            int lm2 = atom_type.indexb(ib2).lm;

            //get radial basis functions indices
            int irb1 = atom_type.indexb(ib1).idxrf;
            int irb2 = atom_type.indexb(ib2).idxrf;

            // common index
            int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

            // get num of non-zero GC
            int num_non_zero_gk = GC.num_gaunt(lm1,lm2);

            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++ ){
                /* add nonzero coefficients */
                for (int inz = 0; inz < num_non_zero_gk; inz++) {
                    auto& lm3coef = GC.gaunt(lm1, lm2, inz);

                    /* add to atom Dij an integral of dij array */
                    paw_dij(ib1, ib2, imagn, paw_ind) += lm3coef.coef * integrals(lm3coef.lm3, iqij, imagn);
                }

                if (ib1 != ib2) {
                    paw_dij(ib2, ib1, imagn, paw_ind) = paw_dij(ib1, ib2, imagn, paw_ind);
                }
            }
        }
    }
}

inline double Potential::calc_PAW_one_elec_energy(paw_potential_data_t& pdd,
                                                  const mdarray<double_complex, 4>& density_matrix,
                                                  const mdarray<double, 4>& paw_dij)
{
    int ia = pdd.ia;
    int paw_ind = pdd.ia_paw;

    double_complex energy = 0.0;

    for (int ib2 = 0; ib2 < pdd.atom_->mt_lo_basis_size(); ib2++ ) {
        for (int ib1 = 0; ib1 < pdd.atom_->mt_lo_basis_size(); ib1++ ) {
            double dm[4]={0,0,0,0};
            switch (ctx_.num_mag_dims()) {
                case 3: {
                    dm[2] =  2 * std::real(density_matrix(ib1, ib2, 2, ia));
                    dm[3] = -2 * std::imag(density_matrix(ib1, ib2, 2, ia));
                }
                case 1: {
                    dm[0] = std::real(density_matrix(ib1, ib2, 0, ia) + density_matrix(ib1, ib2, 1, ia));
                    dm[1] = std::real(density_matrix(ib1, ib2, 0, ia) - density_matrix(ib1, ib2, 1, ia));
                    break;
                }
                case 0: {
                    dm[0] = density_matrix(ib1, ib2, 0, ia).real();
                    break;
                }
                default:{
                    TERMINATE("calc_PAW_one_elec_energy FATAL ERROR!");
                    break;
                }
            }
            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++ ){
                energy += dm[imagn] * paw_dij(ib1, ib2, imagn, paw_ind);
            }
        }
    }

    if (std::abs(energy.imag()) > 1e-10) {
        std::stringstream s;
        s << "PAW energy is not real: "<< energy;
        TERMINATE(s.str());
    }

    pdd.one_elec_energy_ = energy.real();

    return energy.real();
}

inline void Potential::add_paw_Dij_to_atom_Dmtrx()
{
    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia = unit_cell_.paw_atom_index(i);
        auto& atom = unit_cell_.atom(ia);

        for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++ ){
            for (int ib2 = 0; ib2 < atom.mt_lo_basis_size(); ib2++) {
                for (int ib1 = 0; ib1 < atom.mt_lo_basis_size(); ib1++) {
                     atom.d_mtrx(ib1, ib2, imagn) += paw_dij_(ib1, ib2, imagn, i);
                }
            }
        }
    }
}

