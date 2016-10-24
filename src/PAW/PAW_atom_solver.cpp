/*
 * Paw_atom_solver.cpp
 *
 *  Created on: Oct 21, 2016
 *      Author: isivkov
 */


#include "PAW_atom_solver.h"

namespace sirius
{

void PAW_atom_solver::init_density()
{
    // allocate density arrays
    ae_density_ = mdarray<double, 2>(lm_max_rho_, num_mt_points_);
    ps_density_ = mdarray<double, 2>(lm_max_rho_, num_mt_points_);

    ae_density_.zero();
    ps_density_.zero();

    // magnetization arrays
    ae_magnetization_ = mdarray<double, 3>(lm_max_rho_, num_mt_points_, 3);
    ps_magnetization_ = mdarray<double, 3>(lm_max_rho_, num_mt_points_, 3);

    ae_magnetization_.zero();
    ps_magnetization_.zero();
}



void PAW_atom_solver::init_potential()
{

    // allocate potential
    mdarray<double, 3> ae_atom_potential(lm_max_rho_, num_mt_points_, ctx_.num_mag_dims()+1);
    mdarray<double, 3> ps_atom_potential(lm_max_rho_, num_mt_points_, ctx_.num_mag_dims()+1);

    // initialize dij matrix
    paw_dij_ = mdarray<double_complex,3>(basis_size_, basis_size_, ctx_.num_mag_dims()+1);

    core_energy_ = 0.5 * atom_type_->pp_desc().core_energy; // Hartree to A.U.
}





void PAW_atom_solver::generate_potential()
{
    int atom_index = unit_cell_.spl_num_atoms(spl_atom_index);

    auto& atom = unit_cell_.atom(atom_index);

    auto& atom_type = atom.type();

    auto& pp_desc = atom_type.pp_desc();

    //-----------------------------------------
    //---- Calculation of Hartree potential ---
    //-----------------------------------------

    ae_paw_local_potential_[spl_atom_index].zero();
    ps_paw_local_potential_[spl_atom_index].zero();

    double ae_hartree_energy = calc_PAW_hartree_potential(ae_density_,
                                                          ae_potential_);

    double ps_hartree_energy = calc_PAW_hartree_potential(ps_density_,
                                                          ps_potential_);

    paw_hartree_energy_ = ae_hartree_energy - ps_hartree_energy;

    ////////////////////////////////////////////////////////////////////
    //    stringstream s,sd;
    //    s<<"vh_"<<atom_index<<".dat";
    //
    //    ofstream ofs(s.str());
    //
    //    for(int i=0; i<ae_paw_local_potential_[atom_index].size(); i++)
    //    {
    //        ofs<<ae_paw_local_potential_[atom_index][i]<<" "<<ps_paw_local_potential_[atom_index][i]<<endl;
    //    }
    //
    //    ofs.close();
    ////////////////////////////////////////////////////////////////////

    //-----------------------------------------
    //---- Calculation of XC potential ---
    //-----------------------------------------
    double ae_xc_energy = 0.0;
    double ps_xc_energy = 0.0;

    switch(ctx_.num_mag_comp())
    {
        case 1:
        {
            ae_xc_energy = xc_mt_PAW_nonmagnetic(ae_potential_,
                                                 ae_density_, pp_desc_.all_elec_core_charge);

            ps_xc_energy = xc_mt_PAW_nonmagnetic(ps_potential_,
                                                 ps_density_, pp_desc_.core_charge_density);
        }break;

        case 2:
        {
            //            mdarray<double,3> xc(ae_full_density.size(0),ae_full_density.size(1),2);
            //            mdarray<double,3> xcps(ae_full_density.size(0),ae_full_density.size(1),2);
            //
            //            xc.zero();
            //            xcps.zero();

            ae_xc_energy = xc_mt_PAW_collinear(ae_potential_, ae_density_,
                                               ae_magnetization_, pp_desc_.all_elec_core_charge);

            ps_xc_energy = xc_mt_PAW_collinear(ps_potential_, ps_density_,
                                               ps_magnetization_, pp_desc_.core_charge_density);


            //            for(int i=0; i<xc.size(); i++) ae_paw_local_potential_[spl_atom_index][i] += xc[i];
            //            for(int i=0; i<xc.size(); i++) ps_paw_local_potential_[spl_atom_index][i] += xcps[i];

            ////////////////////////////////////////////////////////////////////
            //            stringstream sx;
            //
            //            sx<<"xc_"<<atom_index<<".dat";
            //
            //            ofstream ofsx(sx.str());
            //
            //
            //            for(int i=0; i<xc.size(); i++)
            //            {
            //                ofsx<<xc[i]<<" "<<xcps[i]<<endl;
            //            }
            //
            //            ofsx.close();
            ////////////////////////////////////////////////////////////////////
        }break;

        case 3:
        {
            xc_mt_PAW_noncollinear();
            TERMINATE("PAW potential ERROR! Non-collinear is not implemented");
        }break;

        default:
        {
            TERMINATE("PAW local potential error! Wrong number of spins!")
        }break;
    }

    ////////////////////////////////////////////////////////////////
    //      for(int i=0;i<ps_paw_local_potential_[atom_index].size(1);i++)
    //          for(int j=0;j<ps_paw_local_potential_[atom_index].size(0);j++)
    //              std::cout<<ps_paw_local_potential_[atom_index](j,i,0) - ps_paw_local_potential_[atom_index](j,i,1) << " ";
    //
    //      std::cout<<std::endl;
    ////////////////////////////////////////////////////////////////

    paw_xc_energy_ = ae_xc_energy - ps_xc_energy;
}




double PAW_atom_solver::calc_xc_nonmagnetic(mdarray<double, 3> &out_atom_pot,
                                            mdarray<double, 2> &full_rho_lm,
                                            const std::vector<double> &rho_core)
{
    const Radial_grid& rgrid = atom_->radial_grid();

    int lmmax = static_cast<int>(full_rho_lm.size(0));

    Spheric_function<spectral,double> out_atom_pot_sf(&out_atom_pot(0,0,0), lmmax, rgrid);

    Spheric_function<spectral,double> full_rho_lm_sf(&full_rho_lm(0,0), lmmax, rgrid);

    Spheric_function<spectral,double> full_rho_lm_sf_new(lmmax, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += full_rho_lm_sf;

    double invY00 = 1. / y00 ;

    for(int ir = 0; ir < rgrid.num_points(); ir++ )
    {
        full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
    }

    Spheric_function<spatial,double> full_rho_tp_sf = transform(sht_, full_rho_lm_sf_new);

    // create potential in theta phi
    Spheric_function<spatial,double> vxc_tp_sf(sht_->num_points(), rgrid);

    // create energy in theta phi
    Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

    potential_->xc_mt_nonmagnetic(rgrid, potential_->get_xc_functionals(), full_rho_lm_sf_new, full_rho_tp_sf, vxc_tp_sf, exc_tp_sf);

    out_atom_pot_sf += transform(sht_, vxc_tp_sf);

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<spectral,double> exc_lm_sf = transform(sht_, exc_tp_sf );

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}




double PAW_atom_solver::calc_xc_collinear(mdarray<double,3> &out_atom_pot,
                                          mdarray<double,2> &full_rho_lm,
                                          mdarray<double,3> &magnetization_lm,
                                          const std::vector<double> &rho_core)
{
    const Radial_grid& rgrid = atom_->radial_grid();

    assert(out_atom_pot.size(2)==2);

    int lmsize_rho = static_cast<int>(full_rho_lm.size(0));

    // make spherical functions for input density
    Spheric_function<spectral,double> full_rho_lm_sf(&full_rho_lm(0,0), lmsize_rho, rgrid);

    Spheric_function<spectral,double> full_rho_lm_sf_new(lmsize_rho, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += full_rho_lm_sf;

    double invY00 = 1. / y00 ;

    for(int ir = 0; ir < rgrid.num_points(); ir++ )
    {
        full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
    }

    // make spherical functions for output potential
    Spheric_function<spectral,double> out_atom_effective_pot_sf(&out_atom_pot(0,0,0),lmsize_rho,rgrid);
    Spheric_function<spectral,double> out_atom_effective_field_sf(&out_atom_pot(0,0,1),lmsize_rho,rgrid);

    // make magnetization from z component in lm components
    Spheric_function<spectral,double> magnetization_Z_lm(&magnetization_lm(0,0,0), lmsize_rho, rgrid );

    // calculate spin up spin down density components in lm components
    // up = 1/2 ( rho + magn );  down = 1/2 ( rho - magn )
    Spheric_function<spectral,double> rho_u_lm_sf =  0.5 * (full_rho_lm_sf_new + magnetization_Z_lm);
    Spheric_function<spectral,double> rho_d_lm_sf =  0.5 * (full_rho_lm_sf_new - magnetization_Z_lm);

    // transform density to theta phi components
    Spheric_function<spatial,double> rho_u_tp_sf = transform(sht_, rho_u_lm_sf );
    Spheric_function<spatial,double> rho_d_tp_sf = transform(sht_, rho_d_lm_sf );

    // create potential in theta phi
    Spheric_function<spatial,double> vxc_u_tp_sf(sht_->num_points(), rgrid);
    Spheric_function<spatial,double> vxc_d_tp_sf(sht_->num_points(), rgrid);

    // create energy in theta phi
    Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

    // calculate XC
    potential_->xc_mt_magnetic(rgrid, potential_->get_xc_functionals(),
                               rho_u_lm_sf, rho_u_tp_sf,
                               rho_d_lm_sf, rho_d_tp_sf,
                               vxc_u_tp_sf, vxc_d_tp_sf,
                               exc_tp_sf);

    // transform back in lm
    out_atom_effective_pot_sf += transform(sht_, 0.5 * (vxc_u_tp_sf + vxc_u_tp_sf) );
    out_atom_effective_field_sf += transform(sht_, 0.5 * (vxc_u_tp_sf - vxc_u_tp_sf));
    //////////////////////////////////////////////////////////////
    //  std::cout<<"\n UP "<<std::endl;
    //  for(int lm=0; lm<out_atom_pot_up_sf.angular_domain_size(); lm++)
    //  {
    //      for(int ir = 0; ir < rgrid.num_points(); ir++ )
    //      {
    //          if(full_rho_lm_sf(lm,ir) > 50)
    //          std::cout<<full_rho_lm_sf(lm,ir)<<" ";
    //      }
    //  }
    //  std::cout<<"\n POT "<<std::endl;
    //  for(int lm=0; lm<out_atom_pot_up_sf.angular_domain_size(); lm++)
    //  {
    //      for(int ir = 0; ir < rgrid.num_points(); ir++ )
    //      {
    //          if(out_atom_pot_dn_sf(lm,ir) > 50)
    //          std::cout<<out_atom_pot_dn_sf(lm,ir)<<" ";
    //      }
    //  }
    //////////////////////////////////////////////////////////////

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<spectral,double> exc_lm_sf = transform(sht_, exc_tp_sf );

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}





double PAW_atom_solver::calc_hartree_potential( mdarray<double, 2> &full_density, mdarray<double, 3> &out_atom_pot)
{
    //---------------------
    //-- calc potential --
    //---------------------
    const Radial_grid& rgrid = atom_->radial_grid();

    int lmsize_rho = static_cast<int>(out_atom_pot.size(0));

    Spheric_function<function_domain_t::spectral,double> dens_sf(&full_density(0, 0), lmsize_rho, grid);

    // array passed to poisson solver
    Spheric_function<spectral,double> atom_pot_sf(lmsize_rho, grid);
    atom_pot_sf.zero();

    // create qmt to store multipoles
    mdarray<double_complex,1> qmt(lmsize_rho);

    // solve poisson eq and fill 0th spin component of hartree array (in nonmagnetic we have only this)
    qmt.zero();
    potential_->poisson_atom_vmt(dens_sf, atom_pot_sf, qmt, *atom_);

    // make spher funcs from arrays
    Spheric_function<spectral,double> out_atom_pot_sf(&out_atom_pot(0, 0, 0), lmsize_rho, grid);
    out_atom_pot_sf += atom_pot_sf;


    //---------------------
    //--- calc energy ---
    //---------------------
    std::vector<int> l_by_lm = Utils::l_by_lm( Utils::lmax_by_lmmax(lmsize_rho) );

    // create array for integration
    std::vector<double> intdata(grid.num_points(),0);

    double hartree_energy=0.0;

    for(int lm=0; lm < lmsize_rho; lm++)
    {
        // fill data to integrate
        for(int irad = 0; irad < grid.num_points(); irad++)
        {
            intdata[irad] = full_density(lm,irad) * out_atom_pot(lm, irad, 0) * grid[irad] * grid[irad];
        }

        // create spline from the data
        Spline<double> h_spl(grid,intdata);

        hartree_energy += 0.5 * h_spl.integrate(0);
    }

    return hartree_energy;
}





void PAW_atom_solver::calc_Dij(mdarray<double_complex,4>& paw_dij_in_out)
{

}


}


