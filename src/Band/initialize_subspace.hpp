inline void Band::initialize_subspace(K_point_set& kset__, Potential& potential__) const
{
    PROFILE("sirius::Band::initialize_subspace");

    int N{0};
    /* interpolate I_{\alpha,n}(q) = <j_{l_n}(q*x) | wf_{n,l_n}(x) > with splines */
    std::vector<std::vector<Spline<double>>> rad_int(unit_cell_.num_atom_types());

    int nq = static_cast<int>(ctx_.gk_cutoff() * 10);
    /* this is the regular grid in reciprocal space in the range [0, |G+k|_max ] */
    Radial_grid_lin<double> qgrid(nq, 0, ctx_.gk_cutoff());

    std::vector<int> pref = {1, 2, 6, 24, 120};
    if (ctx_.iterative_solver_input().init_subspace_ == "lcao") {
        /* spherical Bessel functions jl(qx) for atom types */
        // mdarray<Spherical_Bessel_functions, 2> jl(nq, unit_cell_.num_atom_types());

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            ///* create jl(qx) */
            //#pragma omp parallel for
            // for (int iq = 0; iq < nq; iq++) {
            //    jl(iq, iat) = Spherical_Bessel_functions(atom_type.indexr().lmax(), atom_type.radial_grid(),
            //    qgrid[iq]);
            //}

            // rad_int[iat].resize(atom_type.pp_desc().atomic_pseudo_wfs_.size());
            rad_int[iat].resize(atom_type.indexr().lmax() + 1);
            /* loop over all pseudo wave-functions */
            // for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
            for (int l = 0; l <= atom_type.indexr().lmax(); l++) {
                // rad_int[iat][i] = Spline<double>(qgrid);
                rad_int[iat][l] = Spline<double>(qgrid);

		///* interpolate atomic_pseudo_wfs(r) */
		// Spline<double> wf(atom_type.radial_grid());
		// for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
		//    //wf[ir] = atom_type.pp_desc().atomic_pseudo_wfs_[i].second[ir];
		//    double x = atom_type.radial_grid(ir);
		//    wf[ir] = std::exp(-atom_type.zn() * x) * std::pow(x, l);
		//}
		// wf.interpolate();
		// double norm = inner(wf, wf, 2);
		//
		////int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
                #pragma omp parallel for
                for (int iq = 0; iq < nq; iq++) {
                    double q = qgrid[iq];
                    // rad_int[iat][i][iq] = sirius::inner(jl(iq, iat)[l], wf, 1);
                    // rad_int[iat][l][iq] = inner(jl(iq, iat)[l], wf, 2) / std::sqrt(norm);
                    double q2 = std::pow(q, 2);
                    /* integral of Exp[-2x]x^l with spherical bessel functions jl(qx) and standard x^2 weight */
                    rad_int[iat][l][iq] =
                        std::pow(2, 2 + l) * std::pow(q, l) * std::pow(1.0 / (4 + q2), 2 + l) * pref[l];
                }

                // rad_int[iat][i].interpolate();
                rad_int[iat][l].interpolate();
            }
        }

        /* get the total number of atomic-centered orbitals */
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            int n           = Utils::lmmax(atom_type.indexr().lmax());
            // int n{0};
            // for (auto& wf: atom_type.pp_desc().atomic_pseudo_wfs_) {
            //    n += (2 * wf.first + 1);
            //}
            N += atom_type.num_atoms() * n;
        }

        if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 2) {
            printf("number of atomic orbitals: %i\n", N);
        }
    }

    local_op_->prepare(ctx_.gvec_coarse(), ctx_.num_mag_dims(), potential__);

    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];

        if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
            initialize_subspace<double>(kp, N, rad_int);
        } else {
            initialize_subspace<double_complex>(kp, N, rad_int);
        }
    }
    local_op_->dismiss();

    /* reset the energies for the iterative solver to do at least two steps */
    for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
        for (int i = 0; i < ctx_.num_bands(); i++) {
            kset__[ik]->band_energy(i)    = 0;
            kset__[ik]->band_occupancy(i) = ctx_.max_occupancy();
        }
    }
}

template <typename T>
inline void
Band::initialize_subspace(K_point* kp__, int num_ao__, std::vector<std::vector<Spline<double>>> const& rad_int__) const
{
    PROFILE("sirius::Band::initialize_subspace|kp");

    /* number of basis functions */
    int num_phi = std::max(num_ao__, ctx_.num_fv_states());

    int num_sc = (ctx_.num_mag_dims() == 3) ? 2 : 1;

    int num_spin_steps = (ctx_.num_mag_dims() == 3) ? 1 : ctx_.num_spins();

    int num_phi_tot = (ctx_.num_mag_dims() == 3) ? num_phi * 2 : num_phi;

    /* initial basis functions */
    Wave_functions phi(ctx_.processing_unit(), kp__->gkvec(), num_phi_tot, num_sc);
    for (int ispn = 0; ispn < num_sc; ispn++) {
        phi.component(ispn).pw_coeffs().prime().zero();
    }

    sddk::timer t1("sirius::Band::initialize_subspace|kp|wf");

    if (num_ao__ > 0) {
        mdarray<double, 2> rlm_gk(kp__->num_gkvec_loc(), Utils::lmmax(unit_cell_.lmax()));
        mdarray<std::pair<int, double>, 1> idx_gk(kp__->num_gkvec_loc());
        #pragma omp parallel for schedule(static)
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            int igk = kp__->idxgk(igk_loc);
            /* vs = {r, theta, phi} */
            auto vs = SHT::spherical_coordinates(kp__->gkvec().gkvec_cart(igk));
            /* compute real spherical harmonics for G+k vector */
            std::vector<double> rlm(Utils::lmmax(unit_cell_.lmax()));
            SHT::spherical_harmonics(unit_cell_.lmax(), vs[1], vs[2], &rlm[0]);
            for (int lm = 0; lm < Utils::lmmax(unit_cell_.lmax()); lm++) {
                rlm_gk(igk_loc, lm) = rlm[lm];
            }
            int i           = static_cast<int>((vs[0] / ctx_.gk_cutoff()) * (rad_int__[0][0].num_points() - 1));
            double dgk      = vs[0] - rad_int__[0][0].radial_grid()[i];
            idx_gk(igk_loc) = std::pair<int, double>(i, dgk);
        }

        std::vector<int> idxao;
        int n{0};
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_type = unit_cell_.atom(ia).type();
            idxao.push_back(n);
            // for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
            for (int l = 0; l <= atom_type.indexr().lmax(); l++) {
                // int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
                n += (2 * l + 1);
            }
        }

        mdarray<double, 3> ri(kp__->num_gkvec_loc(), unit_cell_.lmax() + 1, unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            for (int l = 0; l <= unit_cell_.atom_type(iat).indexr().lmax(); l++) {
                #pragma omp parallel for
                for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
                    ri(igk_loc, l, iat) = rad_int__[iat][l](idx_gk[igk_loc].first, idx_gk[igk_loc].second);
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            double phase           = twopi * (kp__->gkvec().vk() * unit_cell_.atom(ia).position());
            double_complex phase_k = std::exp(double_complex(0.0, phase));

            std::vector<double_complex> phase_gk(kp__->num_gkvec_loc());
            for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
                int igk           = kp__->idxgk(igk_loc);
                auto G            = kp__->gkvec().gvec(igk);
                phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
            }
            auto& atom_type = unit_cell_.atom(ia).type();
            int n{0};
            // for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
            for (int l = 0; l <= atom_type.indexr().lmax(); l++) {
                // int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
                double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                for (int m = -l; m <= l; m++) {
                    int lm = Utils::lm_by_l_m(l, m);
                    for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
                        // phi.pw_coeffs().prime(igk_loc, n++) = z * phase_factor * gkvec_rlm[lm] *
                        // rad_int__[atom_type.id()][i](vs[0]);
                        phi.component(0).pw_coeffs().prime(igk_loc, idxao[ia] + n) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * ri(igk_loc, l, atom_type.id());
                    }
                    n++;
                }
            }
        }
    }

    /* fill remaining wave-functions with pseudo-random guess */
    assert(kp__->num_gkvec() > num_phi + 10);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_phi - num_ao__; i++) {
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->idxgk(igk_loc);
            if (igk == i + 1) {
                phi.component(0).pw_coeffs().prime(igk_loc, num_ao__ + i) = 1.0;
            }
            if (igk == i + 2) {
                phi.component(0).pw_coeffs().prime(igk_loc, num_ao__ + i) = 0.5;
            }
            if (igk == i + 3) {
                phi.component(0).pw_coeffs().prime(igk_loc, num_ao__ + i) = 0.25;
            }
        }
        // for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
        //    /* global index of G+k vector */
        //    int igk = kp__->idxgk(igk_loc);
        //    /* G-vector */
        //    auto G = kp__->gkvec().gvec(igk);
        //    /* index of G-vector */
        //    int ig = ctx_.gvec().index_by_gvec(G);

        //    if (ig == -1) {
        //        ig = ctx_.gvec().index_by_gvec(G * (-1));
        //    }

        //    if (ig >= 0 && ctx_.gvec().shell(ig) == i + 1) {
        //        phi.component(0).pw_coeffs().prime(igk_loc, num_ao__ + i) = 1.0;
        //    }
        //}
    }

    std::vector<double> tmp(4096);
    for (int i = 0; i < 4096; i++) {
        tmp[i] = type_wrapper<double>::random();
    }
    int igk0 = (kp__->comm().rank() == 0) ? 1 : 0;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_phi; i++) {
        for (int igk_loc = igk0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->idxgk(igk_loc);
            phi.component(0).pw_coeffs().prime(igk_loc, i) += tmp[igk & 0xFFF] * 1e-5;
        }
    }

    if (ctx_.num_mag_dims() == 3) {
        phi.component(1).copy_from(phi.component(0), 0, num_phi, num_phi, CPU);
    }
    t1.stop();

    /* short notation for number of target wave-functions */
    int num_bands = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : ctx_.num_fv_states();

    ctx_.fft_coarse().prepare(kp__->gkvec().partition());
    local_op_->prepare(kp__->gkvec());

    D_operator<T> d_op(ctx_, kp__->beta_projectors());
    Q_operator<T> q_op(ctx_, kp__->beta_projectors());

    /* allocate wave-functions */
    Wave_functions hphi(ctx_.processing_unit(), kp__->gkvec(), num_phi_tot, num_sc);
    Wave_functions ophi(ctx_.processing_unit(), kp__->gkvec(), num_phi_tot, num_sc);
    /* temporary wave-functions required as a storage during orthogonalization */
    wave_functions wf_tmp(ctx_.processing_unit(), kp__->gkvec(), num_phi_tot);

    int bs        = ctx_.cyclic_block_size();
    auto mem_type = (std_evp_solver().type() == ev_magma) ? memory_t::host_pinned : memory_t::host;
    dmatrix<T> hmlt(num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<T> evec(num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<T> hmlt_old;

    std::vector<double> eval(num_bands);

    kp__->beta_projectors().prepare();

    if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        if (!keep_wf_on_gpu) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                kp__->spinor_wave_functions(ispn).pw_coeffs().allocate_on_device();
            }
        }
        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.component(ispn).allocate_on_device();
            phi.component(ispn).copy_to_device(0, num_phi_tot);
            hphi.component(ispn).allocate_on_device();
            ophi.component(ispn).allocate_on_device();
        }
        wf_tmp.allocate_on_device();
        evec.allocate(memory_t::device);
        hmlt.allocate(memory_t::device);
    }
#endif

    if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            auto cs = phi.component(ispn).checksum(0, num_phi_tot);
            DUMP("checksum(phi%i): %18.10f %18.10f", ispn, cs.real(), cs.imag());
        }
    }

    for (int ispn_step = 0; ispn_step < num_spin_steps; ispn_step++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o<T>(kp__, ispn_step, 0, num_phi_tot, phi, hphi, ophi, d_op, q_op);

        /* do some checks */
        if (ctx_.control().verification_ >= 1) {
            set_subspace_mtrx<T>(num_sc, 0, num_phi_tot, phi, ophi, hmlt, hmlt_old);
            // hmlt.serialize("overlap", num_phi);
            double max_diff = check_hermitian(hmlt, num_phi_tot);
            if (max_diff > 1e-12) {
                std::stringstream s;
                s << "overlap matrix is not hermitian, max_err = " << max_diff;
                TERMINATE(s);
            }
            std::vector<double> eo(num_phi_tot);
            if (std_evp_solver().solve(num_phi_tot, num_phi_tot, hmlt.template at<CPU>(), hmlt.ld(), eo.data(),
                                       evec.template at<CPU>(), evec.ld(), hmlt.num_rows_local(),
                                       hmlt.num_cols_local())) {
                std::stringstream s;
                s << "error in diagonalziation";
                TERMINATE(s);
            }
            if (kp__->comm().rank() == 0) {
                printf("[verification] minimum eigen-value of the overlap matrix: %18.12f\n", eo[0]);
            }
            if (eo[0] < 0) {
                TERMINATE("overlap matrix is not positively defined");
            }
        }

        orthogonalize<T>(ctx_.processing_unit(), num_sc, 0, num_phi_tot, phi, hphi, ophi, hmlt, wf_tmp);

        /* setup eigen-value problem */
        set_subspace_mtrx<T>(num_sc, 0, num_phi_tot, phi, hphi, hmlt, hmlt_old);

        // hmlt.serialize("hmlt", num_phi_tot);

        /* solve generalized eigen-value problem with the size N */
        if (std_evp_solver().solve(num_phi_tot, num_bands, hmlt.template at<CPU>(), hmlt.ld(), eval.data(),
                                   evec.template at<CPU>(), evec.ld(), hmlt.num_rows_local(), hmlt.num_cols_local())) {
            std::stringstream s;
            s << "error in diagonalization";
            TERMINATE(s);
        }

        if (ctx_.control().print_checksum_) {
            auto cs = evec.checksum();
            kp__->comm().allreduce(&cs, 1);
            DUMP("checksum(evec): %18.10f", std::abs(cs));
            double cs1{0};
            for (int i = 0; i < num_bands; i++) {
                cs1 += eval[i];
            }
            DUMP("checksum(eval): %18.10f", cs1);
        }

        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands; i++) {
                DUMP("eval[%i]=%20.16f", i, eval[i]);
            }
        }

        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        if (ctx_.num_mag_dims() == 3) {
            transform<T>(ctx_.processing_unit(), 1.0, {&phi}, 0, num_phi_tot, evec, 0, 0, 0.0,
                         {&kp__->spinor_wave_functions()}, 0, num_bands);
        } else {
            transform<T>(ctx_.processing_unit(), phi.component(0), 0, num_phi, evec, 0, 0,
                         kp__->spinor_wave_functions(ispn_step), 0, num_bands);
        }

        for (int j = 0; j < num_bands; j++) {
            kp__->band_energy(j + ispn_step * ctx_.num_fv_states()) = eval[j];
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            auto cs = kp__->spinor_wave_functions(ispn).checksum(0, num_bands);
            DUMP("checksum(spinor_wave_functions_%i): %18.10f %18.10f", ispn, cs.real(), cs.imag());
        }
    }

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__->spinor_wave_functions(ispn).pw_coeffs().copy_to_host(0, num_bands);
            if (!keep_wf_on_gpu) {
                kp__->spinor_wave_functions(ispn).pw_coeffs().deallocate_on_device();
            }
        }
    }
#endif

    kp__->beta_projectors().dismiss();
    ctx_.fft_coarse().dismiss();
}
