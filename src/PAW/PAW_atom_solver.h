/*
 * Paw_atom_solver.h
 *
 *  Created on: Oct 21, 2016
 *      Author: isivkov
 */

#ifndef SRC_PAW_PAW_ATOM_SOLVER_H_
#define SRC_PAW_PAW_ATOM_SOLVER_H_

#include "../atom.h"
#include "../atom_type.h"
#include "../descriptors.h"
#include "../potential.h"
#include "../density.h"

namespace sirius
{

class PAW_atom_solver
{
private:

    Simulation_context *ctx_;

    // atom info
    Atom *atom_;
    Atom_type *atom_type_;
    pseudopotential_descriptor *pp_desc_;

    int basis_size_;
    int num_mt_points_;
    int l_max_;
    int lm_max_rho_;

    /// ae and ps local densities
    mdarray<double, 2> ae_density_;
    mdarray<double, 2> ps_density_;

    mdarray<double, 3> ae_magnetization_;
    mdarray<double, 3> ps_magnetization_;

    // potential data
    Potential *potential_;

    mdarray<double,3>  ae_potential_;
    mdarray<double,3>  ps_potential_;

    mdarray<double,3> dij_mtrx_;

    double  paw_hartree_energy_;
    double  paw_xc_energy_;
    double  paw_core_energy_;
    double  paw_one_elec_energy_;

public:

    PAW_atom_solver(Simulation_context* ctx, Atom* atom, pseudopotential_descriptor *pp_desc) :
        ctx_(ctx), atom_(atom), pp_desc_(pp_desc)
    {
        atom_type_ = &atom_->type();

        basis_size_ = atom_type_->mt_lo_basis_size();
        num_mt_points_ = atom_type_->num_mt_points();

        l_max_ = atom_type_->indexr().lmax_lo();
        lm_max_rho_ = (2 * l_max_ + 1) * (2 * l_max_ + 1);
    }

    void init_density();

    void init_potential();

    void generate_potential();

    double calc_xc_mt_nonmagnetic(const Radial_grid& rgrid,
                                  mdarray<double, 3> &out_atom_pot,
                                  mdarray<double, 2> &full_rho_lm,
                                  const std::vector<double> &rho_core);

    double calc_xc_mt_collinear(const Radial_grid& rgrid,
                                mdarray<double,3> &out_atom_pot,
                                mdarray<double,2> &full_rho_lm,
                                mdarray<double,3> &magnetization_lm,
                                const std::vector<double> &rho_core);

    double calc_hartree_potential(Atom& atom, const Radial_grid& grid,
                                  mdarray<double, 2> &full_density,
                                  mdarray<double, 3> &out_atom_pot);


    void calc_Dij(mdarray<double_complex,4>& paw_dij_in_out);

    inline double  PAW_hartree_energies(){ return paw_hartree_energy_; }

    inline double  PAW_xc_energies(){ return paw_xc_energy_; }

    inline double  PAW_core_energies(){ return paw_core_energy_; }

    inline double  PAW_one_elec_energies(){ return paw_one_elec_energy_; }
};

}


#endif /* SRC_PAW_PAW_ATOM_SOLVER_H_ */
