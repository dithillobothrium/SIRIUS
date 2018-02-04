void generate_atomic_orbitals(K_point& kp, Q_operator<double>& q_op)
{
    TERMINATE("Not implemented for gamma point only");
}

void generate_atomic_orbitals(K_point& kp, Q_operator<double_complex>& q_op)
{
    int lmax{0};
    // return immediately if the wave functions are already allocated
    if (kp.hubbard_wave_functions_calculated()) {
        return;
    }
    // printf("test\n");
    kp.allocate_hubbard_wave_functions(this->number_of_hubbard_orbitals());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        lmax = std::max(lmax, unit_cell_.atom_type(iat).lmax_ps_atomic_wf());
    }
    // we need the complex spherical harmonics for the spin orbit case
    // mdarray<double_complex, 2> ylm_gk;
    // if (ctx_.so_correction())
    //   ylm_gk = mdarray<double_complex, 2>(this->num_gkvec_loc(), Utils::lmmax(lmax));

    mdarray<double, 2> rlm_gk(kp.num_gkvec_loc(), Utils::lmmax(lmax));
    mdarray<std::pair<int, double>, 1> idx_gk(kp.num_gkvec_loc());

    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
        int igk = kp.idxgk(igk_loc);
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(kp.gkvec().gkvec_cart(igk));
        /* compute real spherical harmonics for G+k vector */
        std::vector<double> rlm(Utils::lmmax(lmax));
        SHT::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);

        for (int lm = 0; lm < Utils::lmmax(lmax); lm++) {
            rlm_gk(igk_loc, lm) = rlm[lm];
        }

        idx_gk(igk_loc) = ctx_.atomic_wf_ri().iqdq(vs[0]);
    }

    // temporary wave functions
    Wave_functions sphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), ctx_.num_spins());

    #pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        const auto& atom             = unit_cell_.atom(ia);
        const double phase           = twopi * geometry3d::dot(kp.gkvec().vk(), unit_cell_.atom(ia).position());
        const double_complex phase_k = double_complex(cos(phase), sin(phase));

        std::vector<double_complex> phase_gk(kp.num_gkvec_loc());
        for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
            int igk           = kp.idxgk(igk_loc);
            auto G            = kp.gkvec().gvec(igk);
            phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
        }
        const auto& atom_type = atom.type();
        int n{0};
        if (atom_type.hubbard_correction()) {
            const int l      = atom_type.hubbard_l();
            const double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
            if (atom_type.spin_orbit_coupling()) {
                int orb[2];
                int s = 0;
                for (auto i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                    if (atom_type.ps_atomic_wf(i).first == atom_type.hubbard_l()) {
                        orb[s] = i;
                        s++;
                    }
                }
                for (int m = -l; m <= l; m++) {
                    int lm = Utils::lm_by_l_m(l, m);
                    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                        const double_complex temp = (ctx_.atomic_wf_ri().values(orb[0], atom_type.id())(
                                                     idx_gk[igk_loc].first, idx_gk[igk_loc].second) +
                                                     ctx_.atomic_wf_ri().values(orb[1], atom_type.id())(
                                                     idx_gk[igk_loc].first, idx_gk[igk_loc].second)) * 0.5;
                        sphi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + n) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * temp;

                        sphi.pw_coeffs(1).prime(igk_loc, this->offset[ia] + n + 2 * l + 1) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * temp;
                    }
                    n++;
                }
            } else {
                // find the right hubbard orbital
                int orb = -1;
                for (auto i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                    if (atom_type.ps_atomic_wf(i).first == atom_type.hubbard_l()) {
                        orb = i;
                        break;
                    }
                }

                for (int m = -l; m <= l; m++) {
                    int lm = Utils::lm_by_l_m(l, m);
                    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                        for (int s = 0; s < ctx_.num_spins(); s++) {
                            sphi.pw_coeffs(s).prime(igk_loc, this->offset[ia] + n + s * (2 * l + 1)) =
                                z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) *
                                ctx_.atomic_wf_ri().values(orb, atom_type.id())(idx_gk[igk_loc].first, idx_gk[igk_loc].second);
                        }
                    }
                }
            }
        }
    }

    // check if we have a norm conserving pseudo potential only
    bool augment = false;
    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
        augment = ctx_.unit_cell().atom_type(ia).augment();
    }

    for (int s = 0; s < ctx_.num_spins(); s++) {
        // I need to consider the case where all atoms are norm
        // conserving. In that case the S operator is diagonal in orbital space
        kp.hubbard_wave_functions().copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), sphi, s, 0, s, 0);
    }

    if (!ctx_.full_potential() && augment) {
        /* need to apply the matrix here on the orbitals (ultra soft pseudo potential) */
        for (int i = 0; i < ctx_.beta_projector_chunks().num_chunks(); i++) {
            /* generate beta-projectors for a block of atoms */
            kp.beta_projectors().generate(i);
            /* non-collinear case */
            if (ctx_.num_mag_dims() == 3) {
                for (int ispn = 0; ispn < 2; ispn++) {

                    auto beta_phi = kp.beta_projectors().inner<double_complex>(i, sphi, ispn, 0,
                                                                               this->number_of_hubbard_orbitals());

                    /* apply Q operator (diagonal in spin) */
                    q_op.apply(i, ispn, kp.hubbard_wave_functions(), 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                               beta_phi);
                    /* apply non-diagonal spin blocks */
                    if (ctx_.so_correction()) {
                        q_op.apply(i, ispn ^ 3, kp.hubbard_wave_functions(), 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                                   beta_phi);
                    }

                }
            } else { /* non-magnetic or collinear case */
                auto beta_phi = kp.beta_projectors().inner<double_complex>(i, kp.hubbard_wave_functions(), 0, 0,
                                                                           this->number_of_hubbard_orbitals());

                q_op.apply(i, 0, kp.hubbard_wave_functions(), 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(), beta_phi);
            }
        }
        kp.beta_projectors().dismiss();
    }

    // do we orthogonalize the all thing

    if (this->orthogonalize_hubbard_orbitals_ || this->normalize_hubbard_orbitals_only()) {
        dmatrix<double_complex> S(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals());
        S.zero();

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            linalg<CPU>::gemm(2, 0,
                              this->number_of_hubbard_orbitals(),
                              this->number_of_hubbard_orbitals(),
                              sphi.pw_coeffs(ispn).num_rows_loc(),
                              linalg_const<double_complex>::one(),
                              sphi.pw_coeffs(ispn).prime().at<CPU>(0, 0),
                              sphi.pw_coeffs(ispn).prime().ld(),
                              kp.hubbard_wave_functions().pw_coeffs(ispn).prime().at<CPU>(0, 0),
                              kp.hubbard_wave_functions().pw_coeffs(ispn).prime().ld(),
                              linalg_const<double_complex>::one(),
                              S.at<CPU>(0, 0),
                              S.ld());
        }

        kp.comm().allreduce<double_complex, mpi_op_t::sum>(S.at<CPU>(), static_cast<int>(S.size()));

        // diagonalize the all stuff

        if (this->orthogonalize_hubbard_orbitals_) {
            dmatrix<double_complex> Z(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals());

            auto ev_solver = Eigensolver_factory<double_complex>(ev_solver_t::lapack);

            std::vector<double> eigenvalues(this->number_of_hubbard_orbitals(), 0.0);

            ev_solver->solve(number_of_hubbard_orbitals(), S, &eigenvalues[0], Z);

            // build the O^{-1/2} operator
            for (int i = 0; i < static_cast<int>(eigenvalues.size()); i++) {
                eigenvalues[i] = 1.0 / std::sqrt(eigenvalues[i]);
            }

            // First compute S_{nm} = E_m Z_{nm}
            S.zero();
            for (int l = 0; l < this->number_of_hubbard_orbitals(); l++) {
                for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
                    for (int n = 0; n < this->number_of_hubbard_orbitals(); n++) {
                        S(n, m) += eigenvalues[l] * Z(n, l) * std::conj(Z(m, l));
                    }
                }
            }
        } else {
            for (int l = 0; l < this->number_of_hubbard_orbitals(); l++) {
                for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
                    if (l == m) {
                        S(l, m) = 1.0 / sqrt(S(l, l));
                    } else {
                        S(l, m) = 0.0;
                    }
                }
            }
        }

        // now apply the overlap matrix
        for (int s = 0; s < ctx_.num_spins(); s++) {
            sphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), kp.hubbard_wave_functions(),
                           s, 0, s, 0);
            linalg<CPU>::gemm(0, 2, sphi.pw_coeffs(s).num_rows_loc(), this->number_of_hubbard_orbitals(),
                              this->number_of_hubbard_orbitals(), sphi.pw_coeffs(s).prime().at<CPU>(0, 0),
                              sphi.pw_coeffs(s).prime().ld(), S.at<CPU>(0, 0), S.ld(),
                              kp.hubbard_wave_functions().pw_coeffs(s).prime().at<CPU>(0, 0),
                              kp.hubbard_wave_functions().pw_coeffs(s).prime().ld());
        }
    }
}
