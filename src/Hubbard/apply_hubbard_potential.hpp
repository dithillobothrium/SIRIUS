// this function computes the hubbard contribution to the hamiltonian
// and add it to ophi.

// the S matrix is already applied to phi_i

void apply_hubbard_potential(const K_point& kp,
                             const int idx__,
                             const int n__,
                             Wave_functions& phi,
                             Wave_functions& ophi)
{

    mdarray<double_complex, 2> dm(this->number_of_hubbard_orbitals(), n__); // independent of the k point

    // First calculate the projections
    // dm(i, n, sigma)  = <phi_i^\sigma | psi_{nk}>

    dm.zero();
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        linalg<CPU>::gemm(2, 0,
                          this->number_of_hubbard_orbitals(),
                          n__,
                          kp.hubbard_wave_functions().pw_coeffs(ispn).num_rows_loc(),
                          linalg_const<double_complex>::one(),
                          kp.hubbard_wave_functions().pw_coeffs(ispn).prime().at<CPU>(0, 0),
                          kp.hubbard_wave_functions().pw_coeffs(ispn).prime().ld(),
                          phi.pw_coeffs(ispn).prime().at<CPU>(0, idx__),
                          phi.pw_coeffs(ispn).prime().ld(),
                          linalg_const<double_complex>::one(),
                          dm.at<CPU>(0, 0),
                          dm.ld());
    }

    // Need a reduction over the pool
    kp.comm().allreduce<double_complex, mpi_op_t::sum>(dm.at<CPU>(), static_cast<int>(dm.size()));

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
        const auto& atom = ctx_.unit_cell().atom(ia);
        if (atom.type().hubbard_correction()) {

            // we apply the hubbard correction. For now I have no papers
            // giving me the formula for the SO case so I rely on QE for it
            // but I do not like it at all

            for (int nbnd = 0; nbnd < n__; nbnd++) {
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                        double_complex temp = linalg_const<double_complex>::zero();

                        // computes \f[
                        // \sum_{\sigma,m} V^{\sigma\sigma'}_{m,m'} \left<\phi_m^\sigma|\Psi_{nk}\right>
                        // \f]
                        for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                            const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);
                            for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                temp += this->hubbard_potential_(m1, m2, ind, ia, 0) *
                                        dm(this->offset[ia] + s2 * (2 * atom.type().hubbard_l() + 1) + m2, nbnd);
                            }
                        }

                        for (int s = 0; s < ctx_.num_spins(); s++) {
                            for (int l = 0; l < kp.hubbard_wave_functions().pw_coeffs(s).num_rows_loc(); l++) {
                                ophi.pw_coeffs(s).prime(l, idx__ + nbnd) += temp *
                                    kp.hubbard_wave_functions().pw_coeffs(s).prime(l, this->offset[ia] + s1 * (2 * atom.type().hubbard_l() + 1) + m1);
                            }
                        }
                    }
                }
            }
        }
    }
}
