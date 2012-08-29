namespace sirius
{

class Density
{
    //private:
    public:
        mdarray<double,3> charge_density_mt_;
        mdarray<double,1> charge_density_it_;
        mdarray<complex16,1> charge_density_pw_;
        
        mdarray<double,3> magnetization_density_mt_;
        mdarray<double,2> magnetization_density_it_;
        
    
        
    public:
    
        void initialize()
        {
            charge_density_mt_.set_dimensions(global.lmmax_rho(), global.max_num_mt_points(), global.num_atoms());
            charge_density_mt_.allocate();
            charge_density_it_.set_dimensions(global.fft().size());
            charge_density_it_.allocate();
            charge_density_pw_.set_dimensions(global.num_gvec());
            charge_density_pw_.allocate();
        }
        
        void get_density(double* _rhomt, double* _rhoir)
        {
            memcpy(_rhomt, charge_density_mt_.get_ptr(), charge_density_mt_.size() * sizeof(double)); 
            memcpy(_rhoir, charge_density_it_.get_ptr(), charge_density_it_.size() * sizeof(double)); 
        }
    
        void initial_density()
        {
            std::vector<double> enu;
            for (int i = 0; i < global.num_atom_types(); i++)
                global.atom_type(i)->solve_free_atom(1e-8, 1e-5, 1e-4, enu);

            charge_density_mt_.zero();
            double charge_in_mt = 0.0;
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                Spline rho(global.atom(ia)->type()->num_mt_points(), global.atom(ia)->type()->radial_grid());
                for (int i = 0; i < global.atom(ia)->type()->num_mt_points(); i++)
                {
                    rho[i] = global.atom(ia)->type()->free_atom_density(i);
                    charge_density_mt_(0, i, ia) = rho[i] / y00; 
                    charge_density_mt_(1, i, ia) = rho[i] / y00; // TODO: remove later after tests
                }
                rho.interpolate();
                charge_in_mt += fourpi * rho.integrate(global.atom(ia)->type()->num_mt_points() - 1, 2);
            }
            
            for (int i = 0; i < global.fft().size(); i++)
                charge_density_it_(i) = (global.num_electrons() - charge_in_mt) / global.volume_it();
            
            std::cout << "initial density in IT : " << charge_density_it_(0) << std::endl;
             
            global.fft().transform(&charge_density_it_(0), NULL);

            complex16* fft_buf = global.fft().output_buffer_ptr();
            for (int ig = 0; ig < global.num_gvec(); ig++)
                charge_density_pw_(ig) = fft_buf[global.fft_index(ig)];
        }



};

Density density;

};
