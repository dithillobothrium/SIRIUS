inline void Potential::xc_mt_nonmagnetic(Radial_grid<double> const& rgrid,
                                         std::vector<XC_functional>& xc_func,
                                         Spheric_function<spectral, double> const& rho_lm, 
                                         Spheric_function<spatial, double>& rho_tp, 
                                         Spheric_function<spatial, double>& vxc_tp, 
                                         Spheric_function<spatial, double>& exc_tp)
{
    PROFILE("sirius::Potential::xc_mt_nonmagnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) if (ixc.is_gga()) is_gga = true;

    Spheric_function_gradient<spatial, double> grad_rho_tp(sht_->num_points(), rgrid);
    Spheric_function<spatial, double> lapl_rho_tp;
    Spheric_function<spatial, double> grad_rho_grad_rho_tp;

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_lm = gradient(rho_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            grad_rho_tp[x] = transform(sht_.get(), grad_rho_lm[x]);
        }

        /* compute density gradient product */
        grad_rho_grad_rho_tp = grad_rho_tp * grad_rho_tp;
        
        /* compute Laplacian in Rlm spherical harmonics */
        auto lapl_rho_lm = laplacian(rho_lm);

        /* backward transform Laplacian from Rlm to (theta, phi) */
        lapl_rho_tp = transform(sht_.get(), lapl_rho_lm);
    }

    exc_tp.zero();
    vxc_tp.zero();

    Spheric_function<spatial, double> vsigma_tp;
    if (is_gga) {
        vsigma_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_tp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func) {
        /* if this is an LDA functional */
        if (ixc.is_lda()) {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vxc_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++) {
                    ixc.get_lda(sht_->num_points(), &rho_tp(0, ir), &vxc_t[0], &exc_t[0]);
                    for (int itp = 0; itp < sht_->num_points(); itp++) {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc */
                        vxc_tp(itp, ir) += vxc_t[itp];
                    }
                }
            }
        }
        if (ixc.is_gga()) {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vrho_t(sht_->num_points());
                std::vector<double> vsigma_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++)
                {
                    ixc.get_gga(sht_->num_points(), &rho_tp(0, ir), &grad_rho_grad_rho_tp(0, ir), &vrho_t[0], &vsigma_t[0], &exc_t[0]);
                    for (int itp = 0; itp < sht_->num_points(); itp++)
                    {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc available contributions */
                        vxc_tp(itp, ir) += (vrho_t[itp] - 2 * vsigma_t[itp] * lapl_rho_tp(itp, ir));

                        /* save the sigma derivative */
                        vsigma_tp(itp, ir) += vsigma_t[itp]; 
                    }
                }
            }
        }
    }

    if (is_gga)
    {
        /* forward transform vsigma to Rlm */
        auto vsigma_lm = transform(sht_.get(), vsigma_tp);

        /* compute gradient of vsgima in spherical harmonics */
        auto grad_vsigma_lm = gradient(vsigma_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        Spheric_function_gradient<spatial, double> grad_vsigma_tp(sht_->num_points(), rgrid);
        for (int x = 0; x < 3; x++) {
            grad_vsigma_tp[x] = transform(sht_.get(), grad_vsigma_lm[x]);
        }

        /* compute scalar product of two gradients */
        auto grad_vsigma_grad_rho_tp = grad_vsigma_tp * grad_rho_tp;

        /* add remaining term to Vxc */
        for (int ir = 0; ir < rgrid.num_points(); ir++)
        {
            for (int itp = 0; itp < sht_->num_points(); itp++)
            {
                vxc_tp(itp, ir) -= 2 * grad_vsigma_grad_rho_tp(itp, ir);
            }
        }
    }
}

inline void Potential::xc_mt_magnetic(Radial_grid<double> const& rgrid,
                                      std::vector<XC_functional>& xc_func,
                                      Spheric_function<spectral, double>& rho_up_lm, 
                                      Spheric_function<spatial, double>& rho_up_tp, 
                                      Spheric_function<spectral, double>& rho_dn_lm, 
                                      Spheric_function<spatial, double>& rho_dn_tp, 
                                      Spheric_function<spatial, double>& vxc_up_tp, 
                                      Spheric_function<spatial, double>& vxc_dn_tp, 
                                      Spheric_function<spatial, double>& exc_tp)
{
    PROFILE("sirius::Potential::xc_mt_magnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) {
        if (ixc.is_gga()) {
            is_gga = true;
        }
    }

    Spheric_function_gradient<spatial, double> grad_rho_up_tp(sht_->num_points(), rgrid);
    Spheric_function_gradient<spatial, double> grad_rho_dn_tp(sht_->num_points(), rgrid);

    Spheric_function<spatial, double> lapl_rho_up_tp(sht_->num_points(), rgrid);
    Spheric_function<spatial, double> lapl_rho_dn_tp(sht_->num_points(), rgrid);

    Spheric_function<spatial, double> grad_rho_up_grad_rho_up_tp;
    Spheric_function<spatial, double> grad_rho_dn_grad_rho_dn_tp;
    Spheric_function<spatial, double> grad_rho_up_grad_rho_dn_tp;

    //assert(rho_up_lm.radial_grid().hash() == rho_dn_lm.radial_grid().hash());

    vxc_up_tp.zero();
    vxc_dn_tp.zero();
    exc_tp.zero();

    if (is_gga)
    {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_up_lm = gradient(rho_up_lm);
        auto grad_rho_dn_lm = gradient(rho_dn_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++)
        {
            grad_rho_up_tp[x] = transform(sht_.get(), grad_rho_up_lm[x]);
            grad_rho_dn_tp[x] = transform(sht_.get(), grad_rho_dn_lm[x]);
        }

        /* compute density gradient products */
        grad_rho_up_grad_rho_up_tp = grad_rho_up_tp * grad_rho_up_tp;
        grad_rho_up_grad_rho_dn_tp = grad_rho_up_tp * grad_rho_dn_tp;
        grad_rho_dn_grad_rho_dn_tp = grad_rho_dn_tp * grad_rho_dn_tp;
        
        /* compute Laplacians in Rlm spherical harmonics */
        auto lapl_rho_up_lm = laplacian(rho_up_lm);
        auto lapl_rho_dn_lm = laplacian(rho_dn_lm);

        /* backward transform Laplacians from Rlm to (theta, phi) */
        lapl_rho_up_tp = transform(sht_.get(), lapl_rho_up_lm);
        lapl_rho_dn_tp = transform(sht_.get(), lapl_rho_dn_lm);
    }

    Spheric_function<spatial, double> vsigma_uu_tp;
    Spheric_function<spatial, double> vsigma_ud_tp;
    Spheric_function<spatial, double> vsigma_dd_tp;
    if (is_gga)
    {
        vsigma_uu_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_uu_tp.zero();

        vsigma_ud_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_ud_tp.zero();

        vsigma_dd_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_dd_tp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func)
    {
        /* if this is an LDA functional */
        if (ixc.is_lda()) {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vxc_up_t(sht_->num_points());
                std::vector<double> vxc_dn_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++) {
                    ixc.get_lda(sht_->num_points(), &rho_up_tp(0, ir), &rho_dn_tp(0, ir), &vxc_up_t[0], &vxc_dn_t[0], &exc_t[0]);
                    for (int itp = 0; itp < sht_->num_points(); itp++) {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc */
                        vxc_up_tp(itp, ir) += vxc_up_t[itp];
                        vxc_dn_tp(itp, ir) += vxc_dn_t[itp];
                    }
                }
            }
        }
        if (ixc.is_gga()) {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vrho_up_t(sht_->num_points());
                std::vector<double> vrho_dn_t(sht_->num_points());
                std::vector<double> vsigma_uu_t(sht_->num_points());
                std::vector<double> vsigma_ud_t(sht_->num_points());
                std::vector<double> vsigma_dd_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++) {
                    ixc.get_gga(sht_->num_points(), 
                                &rho_up_tp(0, ir), 
                                &rho_dn_tp(0, ir), 
                                &grad_rho_up_grad_rho_up_tp(0, ir), 
                                &grad_rho_up_grad_rho_dn_tp(0, ir), 
                                &grad_rho_dn_grad_rho_dn_tp(0, ir),
                                &vrho_up_t[0], 
                                &vrho_dn_t[0],
                                &vsigma_uu_t[0], 
                                &vsigma_ud_t[0],
                                &vsigma_dd_t[0],
                                &exc_t[0]);

                    for (int itp = 0; itp < sht_->num_points(); itp++) {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc available contributions */
                        vxc_up_tp(itp, ir) += (vrho_up_t[itp] - 2 * vsigma_uu_t[itp] * lapl_rho_up_tp(itp, ir) - vsigma_ud_t[itp] * lapl_rho_dn_tp(itp, ir));
                        vxc_dn_tp(itp, ir) += (vrho_dn_t[itp] - 2 * vsigma_dd_t[itp] * lapl_rho_dn_tp(itp, ir) - vsigma_ud_t[itp] * lapl_rho_up_tp(itp, ir));

                        /* save the sigma derivatives */
                        vsigma_uu_tp(itp, ir) += vsigma_uu_t[itp]; 
                        vsigma_ud_tp(itp, ir) += vsigma_ud_t[itp]; 
                        vsigma_dd_tp(itp, ir) += vsigma_dd_t[itp]; 
                    }
                }
            }
        }
    }

    if (is_gga) {
        /* forward transform vsigma to Rlm */
        auto vsigma_uu_lm = transform(sht_.get(), vsigma_uu_tp);
        auto vsigma_ud_lm = transform(sht_.get(), vsigma_ud_tp);
        auto vsigma_dd_lm = transform(sht_.get(), vsigma_dd_tp);

        /* compute gradient of vsgima in spherical harmonics */
        auto grad_vsigma_uu_lm = gradient(vsigma_uu_lm);
        auto grad_vsigma_ud_lm = gradient(vsigma_ud_lm);
        auto grad_vsigma_dd_lm = gradient(vsigma_dd_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        Spheric_function_gradient<spatial, double> grad_vsigma_uu_tp(sht_->num_points(), rgrid);
        Spheric_function_gradient<spatial, double> grad_vsigma_ud_tp(sht_->num_points(), rgrid);
        Spheric_function_gradient<spatial, double> grad_vsigma_dd_tp(sht_->num_points(), rgrid);
        for (int x = 0; x < 3; x++)
        {
            grad_vsigma_uu_tp[x] = transform(sht_.get(), grad_vsigma_uu_lm[x]);
            grad_vsigma_ud_tp[x] = transform(sht_.get(), grad_vsigma_ud_lm[x]);
            grad_vsigma_dd_tp[x] = transform(sht_.get(), grad_vsigma_dd_lm[x]);
        }

        /* compute scalar product of two gradients */
        auto grad_vsigma_uu_grad_rho_up_tp = grad_vsigma_uu_tp * grad_rho_up_tp;
        auto grad_vsigma_dd_grad_rho_dn_tp = grad_vsigma_dd_tp * grad_rho_dn_tp;
        auto grad_vsigma_ud_grad_rho_up_tp = grad_vsigma_ud_tp * grad_rho_up_tp;
        auto grad_vsigma_ud_grad_rho_dn_tp = grad_vsigma_ud_tp * grad_rho_dn_tp;

        /* add remaining terms to Vxc */
        for (int ir = 0; ir < rgrid.num_points(); ir++)
        {
            for (int itp = 0; itp < sht_->num_points(); itp++)
            {
                vxc_up_tp(itp, ir) -= (2 * grad_vsigma_uu_grad_rho_up_tp(itp, ir) + grad_vsigma_ud_grad_rho_dn_tp(itp, ir));
                vxc_dn_tp(itp, ir) -= (2 * grad_vsigma_dd_grad_rho_dn_tp(itp, ir) + grad_vsigma_ud_grad_rho_up_tp(itp, ir));
            }
        }
    }
}

inline void Potential::xc_mt(Periodic_function<double>* rho, 
                             Periodic_function<double>* magnetization[3],
                             std::vector<XC_functional>& xc_func,
                             Periodic_function<double>* vxc, 
                             Periodic_function<double>* bxc[3], 
                             Periodic_function<double>* exc)
{
    PROFILE("sirius::Potential::xc_mt");

    for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = unit_cell_.spl_num_atoms(ialoc);
        auto& rgrid = unit_cell_.atom(ia).radial_grid();
        int nmtp = unit_cell_.atom(ia).num_mt_points();

        /* backward transform density from Rlm to (theta, phi) */
        auto rho_tp = transform(sht_.get(), rho->f_mt(ialoc));

        /* backward transform magnetization from Rlm to (theta, phi) */
        std::vector< Spheric_function<spatial, double> > vecmagtp(ctx_.num_mag_dims());
        for (int j = 0; j < ctx_.num_mag_dims(); j++)
            vecmagtp[j] = transform(sht_.get(), magnetization[j]->f_mt(ialoc));
       
        /* "up" component of the density */
        Spheric_function<spectral, double> rho_up_lm;
        Spheric_function<spatial, double> rho_up_tp(sht_->num_points(), rgrid);

        /* "dn" component of the density */
        Spheric_function<spectral, double> rho_dn_lm;
        Spheric_function<spatial, double> rho_dn_tp(sht_->num_points(), rgrid);

        /* check if density has negative values */
        double rhomin = 0.0;
        for (int ir = 0; ir < nmtp; ir++)
        {
            for (int itp = 0; itp < sht_->num_points(); itp++) rhomin = std::min(rhomin, rho_tp(itp, ir));
        }

        if (rhomin < 0.0)
        {
            std::stringstream s;
            s << "Charge density for atom " << ia << " has negative values" << std::endl
              << "most negatve value : " << rhomin << std::endl
              << "current Rlm expansion of the charge density may be not sufficient, try to increase lmax_rho";
            WARNING(s);
        }

        if (ctx_.num_spins() == 1)
        {
            for (int ir = 0; ir < nmtp; ir++)
            {
                /* fix negative density */
                for (int itp = 0; itp < sht_->num_points(); itp++) 
                {
                    if (rho_tp(itp, ir) < 0.0) rho_tp(itp, ir) = 0.0;
                }
            }
        }
        else
        {
            for (int ir = 0; ir < nmtp; ir++)
            {
                for (int itp = 0; itp < sht_->num_points(); itp++)
                {
                    /* compute magnitude of the magnetization vector */
                    double mag = 0.0;
                    for (int j = 0; j < ctx_.num_mag_dims(); j++) mag += pow(vecmagtp[j](itp, ir), 2);
                    mag = std::sqrt(mag);

                    /* in magnetic case fix both density and magnetization */
                    for (int itp = 0; itp < sht_->num_points(); itp++) 
                    {
                        if (rho_tp(itp, ir) < 0.0)
                        {
                            rho_tp(itp, ir) = 0.0;
                            mag = 0.0;
                        }
                        /* fix numerical noise at high values of magnetization */
                        mag = std::min(mag, rho_tp(itp, ir));
                    
                        /* compute "up" and "dn" components */
                        rho_up_tp(itp, ir) = 0.5 * (rho_tp(itp, ir) + mag);
                        rho_dn_tp(itp, ir) = 0.5 * (rho_tp(itp, ir) - mag);
                    }
                }
            }

            /* transform from (theta, phi) to Rlm */
            rho_up_lm = transform(sht_.get(), rho_up_tp);
            rho_dn_lm = transform(sht_.get(), rho_dn_tp);
        }

        Spheric_function<spatial, double> exc_tp(sht_->num_points(), rgrid);
        Spheric_function<spatial, double> vxc_tp(sht_->num_points(), rgrid);

        if (ctx_.num_spins() == 1)
        {
            xc_mt_nonmagnetic(rgrid, xc_func, rho->f_mt(ialoc), rho_tp, vxc_tp, exc_tp);
        }
        else
        {
            Spheric_function<spatial, double> vxc_up_tp(sht_->num_points(), rgrid);
            Spheric_function<spatial, double> vxc_dn_tp(sht_->num_points(), rgrid);

            xc_mt_magnetic(rgrid, xc_func, rho_up_lm, rho_up_tp, rho_dn_lm, rho_dn_tp, vxc_up_tp, vxc_dn_tp, exc_tp);

            for (int ir = 0; ir < nmtp; ir++)
            {
                for (int itp = 0; itp < sht_->num_points(); itp++)
                {
                    /* align magnetic filed parallel to magnetization */
                    /* use vecmagtp as temporary vector */
                    double mag =  rho_up_tp(itp, ir) - rho_dn_tp(itp, ir);
                    if (mag > 1e-8)
                    {
                        /* |Bxc| = 0.5 * (V_up - V_dn) */
                        double b = 0.5 * (vxc_up_tp(itp, ir) - vxc_dn_tp(itp, ir));
                        for (int j = 0; j < ctx_.num_mag_dims(); j++)
                            vecmagtp[j](itp, ir) = b * vecmagtp[j](itp, ir) / mag;
                    }
                    else
                    {
                        for (int j = 0; j < ctx_.num_mag_dims(); j++) vecmagtp[j](itp, ir) = 0.0;
                    }
                    /* Vxc = 0.5 * (V_up + V_dn) */
                    vxc_tp(itp, ir) = 0.5 * (vxc_up_tp(itp, ir) + vxc_dn_tp(itp, ir));
                }       
            }
            /* convert magnetic field back to Rlm */
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                auto bxcrlm = transform(sht_.get(), vecmagtp[j]);
                for (int ir = 0; ir < nmtp; ir++) {
                    for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                        bxc[j]->f_mt<index_domain_t::local>(lm, ir, ialoc) = bxcrlm(lm, ir);
                    }
                }
            }
        }

        /* forward transform from (theta, phi) to Rlm */
        auto vxcrlm = transform(sht_.get(), vxc_tp);
        auto excrlm = transform(sht_.get(), exc_tp);
        for (int ir = 0; ir < nmtp; ir++) {
            for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                vxc->f_mt<index_domain_t::local>(lm, ir, ialoc) = vxcrlm(lm, ir);
                exc->f_mt<index_domain_t::local>(lm, ir, ialoc) = excrlm(lm, ir);
            }
        }
    }
}

inline void Potential::xc_it_nonmagnetic(Periodic_function<double>* rho__, 
                                         std::vector<XC_functional>& xc_func__,
                                         Periodic_function<double>* vxc__, 
                                         Periodic_function<double>* exc__)
{
    PROFILE("sirius::Potential::xc_it_nonmagnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func__) {
        if (ixc.is_gga()) {
            is_gga = true;
        }
    }

    int num_points = ctx_.fft().local_size();

    /* we can use this comm for parallelization */
    auto& comm = ctx_.gvec().comm_ortho_fft();

    /* split real-space points between available ranks */
    splindex<block> spl_np(num_points, comm.size(), comm.rank());

    /* check for negative values */
    double rhomin = 0.0;
    for (int irloc = 0; irloc < spl_np.local_size(); irloc++) {
        int ir = spl_np[irloc];
        rhomin = std::min(rhomin, rho__->f_rg(ir));
        if (rho__->f_rg(ir) < 0.0) {
            rho__->f_rg(ir) = 0.0;
        }
    }
    if (rhomin < 0.0) {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        WARNING(s);
    }
    
    Smooth_periodic_function_gradient<double> grad_rho;
    Smooth_periodic_function<double> lapl_rho;
    Smooth_periodic_function<double> grad_rho_grad_rho;
    
    if (is_gga) {
        /* use fft_transfrom of the base class (Smooth_periodic_function) */
        rho__->Smooth_periodic_function<double>::fft_transform(-1);

        /* generate pw coeffs of the gradient and laplacian */
        grad_rho = gradient(*rho__);
        lapl_rho = laplacian(*rho__);

        /* gradient in real space */
        for (int x: {0, 1, 2}) {
            grad_rho[x].fft_transform(1);
        }

        /* product of gradients */
        grad_rho_grad_rho = grad_rho * grad_rho;
        
        /* Laplacian in real space */
        lapl_rho.fft_transform(1);
    }

    mdarray<double, 1> exc_tmp(spl_np.local_size());
    exc_tmp.zero();

    mdarray<double, 1> vxc_tmp(spl_np.local_size());
    vxc_tmp.zero();

    mdarray<double, 1> vsigma_tmp;
    if (is_gga)
    {
        vsigma_tmp = mdarray<double, 1>(spl_np.local_size());
        vsigma_tmp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func__) {
        #pragma omp parallel
        {
            /* split local size between threads */
            splindex<block> spl_np_t(spl_np.local_size(), omp_get_num_threads(), omp_get_thread_num());

            std::vector<double> exc_t(spl_np_t.local_size());

            /* if this is an LDA functional */
            if (ixc.is_lda()) {
                std::vector<double> vxc_t(spl_np_t.local_size());

                ixc.get_lda(spl_np_t.local_size(),
                            &rho__->f_rg(spl_np.global_offset() + spl_np_t.global_offset()),
                            &vxc_t[0],
                            &exc_t[0]);

                for (int i = 0; i < spl_np_t.local_size(); i++) {
                    /* add Exc contribution */
                    exc_tmp(spl_np_t[i]) += exc_t[i];

                    /* directly add to Vxc */
                    vxc_tmp(spl_np_t[i]) += vxc_t[i];
                }
            }
            if (ixc.is_gga()) {
                std::vector<double> vrho_t(spl_np_t.local_size());
                std::vector<double> vsigma_t(spl_np_t.local_size());
                
                ixc.get_gga(spl_np_t.local_size(), 
                            &rho__->f_rg(spl_np.global_offset() + spl_np_t.global_offset()), 
                            &grad_rho_grad_rho.f_rg(spl_np.global_offset() + spl_np_t.global_offset()),
                            &vrho_t[0], 
                            &vsigma_t[0], 
                            &exc_t[0]);


                for (int i = 0; i < spl_np_t.local_size(); i++) {
                    /* add Exc contribution */
                    exc_tmp(spl_np_t[i]) += exc_t[i];

                    /* directly add to Vxc available contributions */
                    vxc_tmp(spl_np_t[i]) += (vrho_t[i] - 2 * vsigma_t[i] * lapl_rho.f_rg(spl_np.global_offset() + spl_np_t[i]));

                    /* save the sigma derivative */
                    vsigma_tmp(spl_np_t[i]) += vsigma_t[i]; 
                }
            }
        }
    }

    if (is_gga) {
        Smooth_periodic_function<double> vsigma(ctx_.fft(), ctx_.gvec());

        /* gather vsigma */
        comm.allgather(&vsigma_tmp[0], &vsigma.f_rg(0), spl_np.global_offset(), spl_np.local_size()); 

        /* forward transform vsigma to plane-wave domain */
        vsigma.fft_transform(-1);

        /* gradient of vsigma in plane-wave domain */
        auto grad_vsigma = gradient(vsigma);

        /* backward transform gradient from pw to real space */
        for (int x: {0, 1, 2}) {
            grad_vsigma[x].fft_transform(1);
        }

        /* compute scalar product of two gradients */
        auto grad_vsigma_grad_rho = grad_vsigma * grad_rho;

        /* add remaining term to Vxc */
        for (int irloc = 0; irloc < spl_np.local_size(); irloc++) {
            vxc_tmp(irloc) -= 2 * grad_vsigma_grad_rho.f_rg(spl_np[irloc]);
        }
    }
    comm.allgather(&vxc_tmp[0], &vxc__->f_rg(0), spl_np.global_offset(), spl_np.local_size()); 
    comm.allgather(&exc_tmp[0], &exc__->f_rg(0), spl_np.global_offset(), spl_np.local_size()); 

    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(vxc_tmp): %18.10f", vxc_tmp.checksum());
    DUMP("checksum(exc_tmp): %18.10f", exc_tmp.checksum());
    #endif
    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(vxc_tmp): %16llX", vxc_tmp.hash());
    DUMP("hash(exc_tmp): %16llX", exc_tmp.hash());
    #endif
}

inline void Potential::xc_it_magnetic(Periodic_function<double>* rho, 
                                      Periodic_function<double>* magnetization[3], 
                                      std::vector<XC_functional>& xc_func,
                                      Periodic_function<double>* vxc, 
                                      Periodic_function<double>* bxc[3], 
                                      Periodic_function<double>* exc)
{
    PROFILE("sirius::Potential::xc_it_magnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) {
        if (ixc.is_gga()) {
            is_gga = true;
        }
    }

    int num_points = ctx_.fft().local_size();
    
    Smooth_periodic_function<double> rho_up(ctx_.fft(), ctx_.gvec());
    Smooth_periodic_function<double> rho_dn(ctx_.fft(), ctx_.gvec());

    /* compute "up" and "dn" components and also check for negative values of density */
    double rhomin = 0.0;
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        double mag = 0.0;
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            mag += std::pow(magnetization[j]->f_rg(ir), 2);
        }
        mag = std::sqrt(mag);

        /* remove numerical noise at high values of magnetization */
        mag = std::min(mag, rho->f_rg(ir));

        rhomin = std::min(rhomin, rho->f_rg(ir));
        if (rho->f_rg(ir) < 0.0) {
            rho->f_rg(ir) = 0.0;
            mag = 0.0;
        }
        
        rho_up.f_rg(ir) = 0.5 * (rho->f_rg(ir) + mag);
        rho_dn.f_rg(ir) = 0.5 * (rho->f_rg(ir) - mag);
    }

    if (rhomin < 0.0) {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        WARNING(s);
    }

    Smooth_periodic_function_gradient<double> grad_rho_up;
    Smooth_periodic_function_gradient<double> grad_rho_dn;
    Smooth_periodic_function<double> lapl_rho_up;
    Smooth_periodic_function<double> lapl_rho_dn;
    Smooth_periodic_function<double> grad_rho_up_grad_rho_up;
    Smooth_periodic_function<double> grad_rho_up_grad_rho_dn;
    Smooth_periodic_function<double> grad_rho_dn_grad_rho_dn;
    
    if (is_gga) {
        /* get plane-wave coefficients of densities */
        rho_up.fft_transform(-1);
        rho_dn.fft_transform(-1);

        /* generate pw coeffs of the gradient and laplacian */
        auto grad_rho_up = gradient(rho_up);
        auto grad_rho_dn = gradient(rho_dn);
        auto lapl_rho_up = laplacian(rho_up);
        auto lapl_rho_dn = laplacian(rho_dn);

        /* gradient in real space */
        for (int x: {0, 1, 2}) {
            grad_rho_up[x].fft_transform(1);
            grad_rho_dn[x].fft_transform(1);
        }

        /* product of gradients */
        grad_rho_up_grad_rho_up = grad_rho_up * grad_rho_up;
        grad_rho_up_grad_rho_dn = grad_rho_up * grad_rho_dn;
        grad_rho_dn_grad_rho_dn = grad_rho_dn * grad_rho_dn;
        
        /* Laplacian in real space */
        lapl_rho_up.fft_transform(1);
        lapl_rho_dn.fft_transform(1);
    }

    mdarray<double, 1> exc_tmp(num_points);
    exc_tmp.zero();

    mdarray<double, 1> vxc_up_tmp(num_points);
    vxc_up_tmp.zero();

    mdarray<double, 1> vxc_dn_tmp(num_points);
    vxc_dn_tmp.zero();

    mdarray<double, 1> vsigma_uu_tmp;
    mdarray<double, 1> vsigma_ud_tmp;
    mdarray<double, 1> vsigma_dd_tmp;

    if (is_gga) {
        vsigma_uu_tmp = mdarray<double, 1>(num_points);
        vsigma_uu_tmp.zero();
        
        vsigma_ud_tmp = mdarray<double, 1>(num_points);
        vsigma_ud_tmp.zero();
        
        vsigma_dd_tmp = mdarray<double, 1>(num_points);
        vsigma_dd_tmp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func) {
        #pragma omp parallel
        {
            /* split local size between threads */
            splindex<block> spl_t(num_points, omp_get_num_threads(), omp_get_thread_num());

            std::vector<double> exc_t(spl_t.local_size());

            /* if this is an LDA functional */
            if (ixc.is_lda()) {
                std::vector<double> vxc_up_t(spl_t.local_size());
                std::vector<double> vxc_dn_t(spl_t.local_size());

                ixc.get_lda(spl_t.local_size(), 
                            &rho_up.f_rg(spl_t.global_offset()), 
                            &rho_dn.f_rg(spl_t.global_offset()), 
                            &vxc_up_t[0], 
                            &vxc_dn_t[0], 
                            &exc_t[0]);

                for (int i = 0; i < spl_t.local_size(); i++) {
                    /* add Exc contribution */
                    exc_tmp(spl_t[i]) += exc_t[i];

                    /* directly add to Vxc */
                    vxc_up_tmp(spl_t[i]) += vxc_up_t[i];
                    vxc_dn_tmp(spl_t[i]) += vxc_dn_t[i];
                }
            }
            if (ixc.is_gga()) {
                std::vector<double> vrho_up_t(spl_t.local_size());
                std::vector<double> vrho_dn_t(spl_t.local_size());
                std::vector<double> vsigma_uu_t(spl_t.local_size());
                std::vector<double> vsigma_ud_t(spl_t.local_size());
                std::vector<double> vsigma_dd_t(spl_t.local_size());

                ixc.get_gga(spl_t.local_size(), 
                            &rho_up.f_rg(spl_t.global_offset()), 
                            &rho_dn.f_rg(spl_t.global_offset()), 
                            &grad_rho_up_grad_rho_up.f_rg(spl_t.global_offset()), 
                            &grad_rho_up_grad_rho_dn.f_rg(spl_t.global_offset()), 
                            &grad_rho_dn_grad_rho_dn.f_rg(spl_t.global_offset()), 
                            &vrho_up_t[0], 
                            &vrho_dn_t[0], 
                            &vsigma_uu_t[0], 
                            &vsigma_ud_t[0], 
                            &vsigma_dd_t[0], 
                            &exc_t[0]);

                for (int i = 0; i < spl_t.local_size(); i++) {
                    /* add Exc contribution */
                    exc_tmp(spl_t[i]) += exc_t[i];

                    /* directly add to Vxc available contributions */
                    vxc_up_tmp(spl_t[i]) += (vrho_up_t[i] - 2 * vsigma_uu_t[i] * lapl_rho_up.f_rg(spl_t[i]) - vsigma_ud_t[i] * lapl_rho_dn.f_rg(spl_t[i]));
                    vxc_dn_tmp(spl_t[i]) += (vrho_dn_t[i] - 2 * vsigma_dd_t[i] * lapl_rho_dn.f_rg(spl_t[i]) - vsigma_ud_t[i] * lapl_rho_up.f_rg(spl_t[i]));

                    /* save the sigma derivative */
                    vsigma_uu_tmp(spl_t[i]) += vsigma_uu_t[i]; 
                    vsigma_ud_tmp(spl_t[i]) += vsigma_ud_t[i]; 
                    vsigma_dd_tmp(spl_t[i]) += vsigma_dd_t[i]; 
                }
            }
        }
    }

    if (is_gga) {
        /* gather vsigma */
        Smooth_periodic_function<double> vsigma_uu(ctx_.fft(), ctx_.gvec());
        Smooth_periodic_function<double> vsigma_ud(ctx_.fft(), ctx_.gvec());
        Smooth_periodic_function<double> vsigma_dd(ctx_.fft(), ctx_.gvec());
        for (int ir = 0; ir < num_points; ir++) {
            vsigma_uu.f_rg(ir) = vsigma_uu_tmp[ir];
            vsigma_ud.f_rg(ir) = vsigma_ud_tmp[ir];
            vsigma_dd.f_rg(ir) = vsigma_dd_tmp[ir];
        }

        /* forward transform vsigma to plane-wave domain */
        vsigma_uu.fft_transform(-1);
        vsigma_ud.fft_transform(-1);
        vsigma_dd.fft_transform(-1);

        /* gradient of vsigma in plane-wave domain */
        auto grad_vsigma_uu = gradient(vsigma_uu);
        auto grad_vsigma_ud = gradient(vsigma_ud);
        auto grad_vsigma_dd = gradient(vsigma_dd);

        /* backward transform gradient from pw to real space */
        for (int x: {0, 1, 2}) {
            grad_vsigma_uu[x].fft_transform(1);
            grad_vsigma_ud[x].fft_transform(1);
            grad_vsigma_dd[x].fft_transform(1);
        }

        /* compute scalar product of two gradients */
        auto grad_vsigma_uu_grad_rho_up = grad_vsigma_uu * grad_rho_up;
        auto grad_vsigma_dd_grad_rho_dn = grad_vsigma_dd * grad_rho_dn;
        auto grad_vsigma_ud_grad_rho_up = grad_vsigma_ud * grad_rho_up;
        auto grad_vsigma_ud_grad_rho_dn = grad_vsigma_ud * grad_rho_dn;

        /* add remaining term to Vxc */
        for (int ir = 0; ir < num_points; ir++) {
            vxc_up_tmp(ir) -= (2 * grad_vsigma_uu_grad_rho_up.f_rg(ir) + grad_vsigma_ud_grad_rho_dn.f_rg(ir)); 
            vxc_dn_tmp(ir) -= (2 * grad_vsigma_dd_grad_rho_dn.f_rg(ir) + grad_vsigma_ud_grad_rho_up.f_rg(ir)); 
        }
    }

    for (int irloc = 0; irloc < num_points; irloc++) {
        exc->f_rg(irloc) = exc_tmp(irloc);
        vxc->f_rg(irloc) = 0.5 * (vxc_up_tmp(irloc) + vxc_dn_tmp(irloc));
        double m = rho_up.f_rg(irloc) - rho_dn.f_rg(irloc);

        if (m > 1e-8) {
            double b = 0.5 * (vxc_up_tmp(irloc) - vxc_dn_tmp(irloc));
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
               bxc[j]->f_rg(irloc) = b * magnetization[j]->f_rg(irloc) / m;
            }
        } else {
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                bxc[j]->f_rg(irloc) = 0.0;
            }
        }
    }
}

inline void Potential::xc(Periodic_function<double>* rho, 
                          Periodic_function<double>* magnetization[3], 
                          Periodic_function<double>* vxc, 
                          Periodic_function<double>* bxc[3], 
                          Periodic_function<double>* exc)
{
    PROFILE("sirius::Potential::xc");

    if (ctx_.xc_functionals().size() == 0) {
        vxc->zero();
        exc->zero();
        for (int i = 0; i < ctx_.num_mag_dims(); i++) {
            bxc[i]->zero();
        }
        return;
    }

    if (ctx_.full_potential()) {
        xc_mt(rho, magnetization, xc_func_, vxc, bxc, exc);
    }

    if (ctx_.num_spins() == 1) {
        xc_it_nonmagnetic(rho, xc_func_, vxc, exc);
    } else {
        xc_it_magnetic(rho, magnetization, xc_func_, vxc, bxc, exc);
    }
}
