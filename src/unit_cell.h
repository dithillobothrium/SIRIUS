// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file unit_cell.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Unit_cell class.
 */

#ifndef __UNIT_CELL_H__
#define __UNIT_CELL_H__

#include <algorithm>
#include "descriptors.h"
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "atom.h"
#include "mpi_grid.hpp"
#include "symmetry.h"
#include "simulation_parameters.h"
#include "json.hpp"

namespace sirius {

using json = nlohmann::json;

class Unit_cell
{
    private:
        
        /// Basic parameters of the simulation.
        Simulation_parameters const& parameters_;
        
        /// Mapping between atom type label and an ordered internal id in the range [0, \f$ N_{types} \f$).
        std::map<std::string, int> atom_type_id_map_;
         
        /// List of atom types.
        std::vector<Atom_type> atom_types_;

        /// List of atom classes.
        std::vector<Atom_symmetry_class> atom_symmetry_classes_;
        
        /// List of atoms.
        std::vector<Atom> atoms_;

        /// Split index of atoms.
        splindex<block> spl_num_atoms_;
        
        /// Global index of atom by index of PAW atom.
        std::vector<int> paw_atom_index_;

        /// Split index of PAW atoms.
        splindex<block> spl_num_paw_atoms_;

        /// Split index of atom symmetry classes.
        splindex<block> spl_num_atom_symmetry_classes_;

        /// Bravais lattice vectors in column order.
        /** The following convention is used to transform fractional coordinates to Cartesian:  
         *  \f[
         *    \vec v_{C} = {\bf L} \vec v_{f}
         *  \f]
         */
        matrix3d<double> lattice_vectors_;
        
        /// Inverse matrix of Bravais lattice vectors.
        /** This matrix is used to find fractional coordinates by Cartesian coordinates:
         *  \f[
         *    \vec v_{f} = {\bf L}^{-1} \vec v_{C}
         *  \f]
         */
        matrix3d<double> inverse_lattice_vectors_;

        /// Reciprocal lattice vectors in column order.
        /** The following convention is used:
         *  \f[
         *    \vec a_{i} \vec b_{j} = 2 \pi \delta_{ij}
         *  \f]
         *  or in matrix notation
         *  \f[
         *    {\bf A} {\bf B}^{T} = 2 \pi {\bf I}
         *  \f]
         */
        matrix3d<double> reciprocal_lattice_vectors_;
        
        /// Volume \f$ \Omega \f$ of the unit cell. Volume of Brillouin zone is then \f$ (2\Pi)^3 / \Omega \f$.
        double omega_{0};
       
        /// Total volume of the muffin-tin spheres.
        double volume_mt_{0};
        
        /// Volume of the interstitial region.
        double volume_it_{0};

        /// Total nuclear charge.
        int total_nuclear_charge_{0};
        
        /// Total number of core electrons.
        double num_core_electrons_{0};
        
        /// Total number of valence electrons.
        double num_valence_electrons_{0};

        /// Total number of electrons.
        double num_electrons_{0};

        /// List of equivalent atoms, provided externally.
        std::vector<int> equivalent_atoms_;
    
        /// Maximum number of muffin-tin points among all atom types.
        int max_num_mt_points_{0};
        
        /// Total number of MT basis functions.
        int mt_basis_size_{0};
        
        /// Maximum number of MT basis functions among all atoms.
        int max_mt_basis_size_{0};

        /// Maximum number of MT radial basis functions among all atoms.
        int max_mt_radial_basis_size_{0};

        /// Total number of augmented wave basis functions in the muffin-tins.
        /** This is equal to the total number of matching coefficients for each plane-wave. */
        int mt_aw_basis_size_{0};

        /// List of augmented wave basis descriptors.
        /** Establishes mapping between global index in the range [0, mt_aw_basis_size_) 
         *  and corresponding atom and local index \f$ \xi \f$ */
        std::vector<mt_basis_descriptor> mt_aw_basis_descriptors_; 

        std::vector<mt_basis_descriptor> mt_lo_basis_descriptors_; 
        
        /// Total number of local orbital basis functions.
        int mt_lo_basis_size_{0};

        /// Maximum AW basis size among all atoms.
        int max_mt_aw_basis_size_{0};
        
        /// Maximum local orbital basis size among all atoms.
        int max_mt_lo_basis_size_{0};

        /// List of nearest neighbours for each atom.
        std::vector<std::vector<nearest_neighbour_descriptor>> nearest_neighbours_;

        /// Minimum muffin-tin radius.
        double min_mt_radius_{0};
        
        /// Maximum muffin-tin radius.
        double max_mt_radius_{0};
        
        /// Maximum orbital quantum number of radial functions between all atom types.
        int lmax_{-1};

        Communicator_bundle comm_bundle_atoms_;
        
        std::unique_ptr<Symmetry> symmetry_;

        Communicator const& comm_;

        /// Automatically determine new muffin-tin radii as a half distance between neighbor atoms.
        /** In order to guarantee a unique solution muffin-tin radii are dermined as a half distance
         *  bethween nearest atoms. Initial values of the muffin-tin radii are ignored. */
        std::vector<double> find_mt_radii();
        
        /// Check if MT spheres overlap
        inline bool check_mt_overlap(int& ia__, int& ja__);

        inline int next_atom_type_id(std::string label__)
        {
            /* check if the label was already added */
            if (atom_type_id_map_.count(label__) != 0) {   
                std::stringstream s;
                s << "atom type with label " << label__ << " is already in list";
                TERMINATE(s);
            }
            /* take text id */
            atom_type_id_map_[label__] = static_cast<int>(atom_types_.size());
            return atom_type_id_map_[label__];
        }

    public:
    
        Unit_cell(Simulation_parameters const& parameters__,
                  Communicator const& comm__)
            : parameters_(parameters__)
            , comm_(comm__)
        {
        }

        /// Initialize the unit cell data
        /** Several things must be done during this phase:
         *    1. Compute number of electrons
         *    2. Compute MT basis function indices
         *    3. [if needed] Scale MT radii
         *    4. Check MT overlap 
         *    5. Create radial grid for each atom type
         *    6. Find symmetry and assign symmetry class to each atom
         *    7. Create split indices for atoms and atom classes */
        inline void initialize();

        /// Add new atom type to the list of atom types and read necessary data from the .json file
        inline void add_atom_type(const std::string label, const std::string file_name)
        {
            PROFILE("sirius::Unit_cell::add_atom_type");

            if (atoms_.size()) {
                TERMINATE("Can't add new atom type if atoms are already added");
            }

            int id = next_atom_type_id(label);
            atom_types_.push_back(std::move(Atom_type(parameters_, id, label, file_name)));
        }
        
        /// Add new atom to the list of atom types.
        inline void add_atom(const std::string label, vector3d<double> position, vector3d<double> vector_field)
        {
            PROFILE("sirius::Unit_cell::add_atom");

            if (atom_type_id_map_.count(label) == 0) {
                std::stringstream s;
                s << "atom type with label " << label << " is not found";
                TERMINATE(s);
            }
            if (atom_id_by_position(position) >= 0) {
                std::stringstream s;
                s << "atom with the same position is already in list" << std::endl
                  << "  position : " << position[0] << " " << position[1] << " " << position[2];
                TERMINATE(s);
            }
            
            atoms_.push_back(std::move(Atom(atom_type(label), position, vector_field)));
            atom_type(label).add_atom_id(static_cast<int>(atoms_.size()) - 1);


        }

        /// Add PAW atoms.
        inline void init_paw()
        {
            for (int ia = 0; ia < num_atoms(); ia++) {
                if (atom(ia).type().pp_desc().is_paw) {
                    paw_atom_index_.push_back(ia);
                }
            }

            spl_num_paw_atoms_ = splindex<block>(num_paw_atoms(), comm_.size(), comm_.rank());
        }

        /// Return number of PAW atoms.
        inline int num_paw_atoms() const
        {
            return static_cast<int>(paw_atom_index_.size());
        }

        /// Get split index of PAW atoms.
        inline splindex<block> spl_num_paw_atoms() const
        {
            return spl_num_paw_atoms_;
        }

        inline int spl_num_paw_atoms(int idx__) const
        {
            return spl_num_paw_atoms_[idx__];
        }

        inline int paw_atom_index(int ipaw__) const
        {
            return paw_atom_index_[ipaw__];
        }

        /// Add new atom without vector field to the list of atom types.
        inline void add_atom(const std::string label, vector3d<double> position)
        {
            PROFILE("sirius::Unit_cell::add_atom");
            add_atom(label, position, {0, 0, 0});
        }
        
        /// Print basic info.
        inline void print_info(int verbosity_);

        inline unit_cell_parameters_descriptor unit_cell_parameters();
        
        /// Get crystal symmetries and equivalent atoms.
        /** Makes a call to spglib providing the basic unit cell information: lattice vectors and atomic types 
         *  and positions. Gets back symmetry operations and a table of equivalent atoms. The table of equivalent 
         *  atoms is then used to make a list of atom symmetry classes and related data. */
        inline void get_symmetry();

        /// Write structure to CIF file.
        inline void write_cif();

        /// Write structure to JSON file.
        inline json serialize();

        inline void set_lattice_vectors(matrix3d<double> lattice_vectors__)
        {
            lattice_vectors_ = lattice_vectors__;
            inverse_lattice_vectors_ = inverse(lattice_vectors_);
            omega_ = std::abs(lattice_vectors_.det());
            reciprocal_lattice_vectors_ = transpose(inverse(lattice_vectors_)) * twopi;
        }
        
        /// Set lattice vectors.
        /** Initializes lattice vectors, inverse lattice vector matrix, reciprocal lattice vectors and the
         *  unit cell volume. */
        inline void set_lattice_vectors(vector3d<double> a0__,
                                        vector3d<double> a1__,
                                        vector3d<double> a2__)
        {
            matrix3d<double> lv;
            for (int x: {0, 1, 2}) {
                lv(x, 0) = a0__[x];
                lv(x, 1) = a1__[x];
                lv(x, 2) = a2__[x];
            }
            set_lattice_vectors(lv);
        }

        /// Find the cluster of nearest neighbours around each atom
        inline void find_nearest_neighbours(double cluster_radius);

        inline bool is_point_in_mt(vector3d<double> vc, int& ja, int& jr, double& dr, double tp[2]) const;
        
        inline void generate_radial_functions();

        inline void generate_radial_integrals();
        
        inline std::string chemical_formula();

        inline int atom_id_by_position(vector3d<double> position__)
        {
            for (int ia = 0; ia < num_atoms(); ia++) {
                auto vd = atom(ia).position() - position__;
                if (vd.length() < 1e-10) {
                    return ia;
                }
            }
            return -1;
        } 

        template <typename T>
        inline vector3d<double> get_cartesian_coordinates(vector3d<T> a__) const
        {
            return lattice_vectors_ * a__;
        }

        inline vector3d<double> get_fractional_coordinates(vector3d<double> a) const
        {
            return inverse_lattice_vectors_ * a;
        }
        
        /// Unit cell volume.
        inline double omega() const
        {
            return omega_;
        }

        /// Number of atom types.
        inline int num_atom_types() const
        {
            assert(atom_types_.size() == atom_type_id_map_.size());
            return static_cast<int>(atom_types_.size());
        }

        /// Return atom type instance by id.
        inline Atom_type& atom_type(int id__)
        {
            assert(id__ >= 0 && id__ < (int)atom_types_.size());
            return atom_types_[id__];
        }

        /// Return const atom type instance by id.
        inline Atom_type const& atom_type(int id__) const
        {
            assert(id__ >= 0 && id__ < (int)atom_types_.size());
            return atom_types_[id__];
        }

        /// Return atom type instance by label.
        inline Atom_type& atom_type(std::string const label__)
        {
            int id = atom_type_id_map_.at(label__);
            return atom_type(id);
        }

        /// Return const atom type instance by label.
        inline Atom_type const& atom_type(std::string const label__) const
        {
            int id = atom_type_id_map_.at(label__);
            return atom_type(id);
        }

        /// Number of atom symmetry classes.
        inline int num_atom_symmetry_classes() const
        {
            return static_cast<int>(atom_symmetry_classes_.size());
        }
       
        /// Return const symmetry class instance by class id.
        inline Atom_symmetry_class const& atom_symmetry_class(int id__) const
        {
            return atom_symmetry_classes_[id__];
        }

        /// Return symmetry class instance by class id.
        inline Atom_symmetry_class& atom_symmetry_class(int id__)
        {
            return atom_symmetry_classes_[id__];
        }

        /// Number of atoms in the unit cell.
        inline int num_atoms() const
        {
            return static_cast<int>(atoms_.size());
        }
        
        /// Return const atom instance by id.
        inline Atom const& atom(int id__) const
        {
            assert(id__ >= 0 && id__ < (int)atoms_.size());
            return atoms_[id__];
        }

        /// Return atom instance by id.
        inline Atom& atom(int id__)
        {
            assert(id__ >= 0 && id__ < (int)atoms_.size());
            return atoms_[id__];
        }

        inline int total_nuclear_charge() const
        {
            return total_nuclear_charge_;
        }
       
        /// Total number of electrons (core + valence).
        inline double num_electrons() const
        {
            return num_electrons_;
        }

        /// Number of valence electrons.
        inline double num_valence_electrons() const
        {
            return num_valence_electrons_;
        }
        
        /// Number of core electrons.
        inline double num_core_electrons() const
        {
            return num_core_electrons_;
        }
        
        /// Maximum number of muffin-tin points among all atom types.
        inline int max_num_mt_points() const
        {
            return max_num_mt_points_;
        }
        
        /// Total number of the augmented wave basis functions over all atoms.
        inline int mt_aw_basis_size() const
        {
            return mt_aw_basis_size_;
        }

        /// Total number of local orbital basis functions over all atoms.
        inline int mt_lo_basis_size() const
        {
            return mt_lo_basis_size_;
        }

        /// Total number of the muffin-tin basis functions.
        /** Total number of MT basis functions equals to the sum of the total number of augmented wave
         *  basis functions and the total number of local orbital basis functions among all atoms. It controls 
         *  the size of the muffin-tin part of the first-variational states and second-variational wave functions. */
        inline int mt_basis_size() const
        {
            return mt_basis_size_;
        }
        
        /// Maximum number of basis functions among all atom types.
        inline int max_mt_basis_size() const
        {
            return max_mt_basis_size_;
        }

        /// Maximum number of radial functions among all atom types.
        inline int max_mt_radial_basis_size() const
        {
            return max_mt_radial_basis_size_;
        }

        /// Minimum muffin-tin radius.
        inline double min_mt_radius() const
        {
            return min_mt_radius_;
        }

        /// Maximum muffin-tin radius.
        inline double max_mt_radius() const
        {
            return max_mt_radius_;
        }

        /// Maximum number of AW basis functions among all atom types.
        inline int max_mt_aw_basis_size() const
        {
            return max_mt_aw_basis_size_;
        }

        inline int max_mt_lo_basis_size() const
        {
            return max_mt_lo_basis_size_;
        }

        void set_equivalent_atoms(int const* equivalent_atoms__)
        {
            equivalent_atoms_.resize(num_atoms());
            memcpy(&equivalent_atoms_[0], equivalent_atoms__, num_atoms() * sizeof(int));
        }
        
        inline splindex<block> const& spl_num_atoms() const
        {
            return spl_num_atoms_;
        }

        inline int spl_num_atoms(int i) const
        {
            return static_cast<int>(spl_num_atoms_[i]);
        }
        
        inline splindex<block> const& spl_num_atom_symmetry_classes() const
        {
            return spl_num_atom_symmetry_classes_;
        }

        inline int spl_num_atom_symmetry_classes(int i) const
        {
            return static_cast<int>(spl_num_atom_symmetry_classes_[i]);
        }

        inline double volume_mt() const
        {
            return volume_mt_;
        }

        inline double volume_it() const
        {
            return volume_it_;
        }

        inline int lmax() const
        {
            return lmax_;
        }

        inline int num_nearest_neighbours(int ia) const
        {
            return static_cast<int>(nearest_neighbours_[ia].size());
        }

        inline nearest_neighbour_descriptor const& nearest_neighbour(int i, int ia) const
        {
            return nearest_neighbours_[ia][i];
        }

        inline mt_basis_descriptor const& mt_aw_basis_descriptor(int idx) const
        {
            return mt_aw_basis_descriptors_[idx];
        }

        inline mt_basis_descriptor const& mt_lo_basis_descriptor(int idx) const
        {
            return mt_lo_basis_descriptors_[idx];
        }

        inline Symmetry const& symmetry() const
        {
            return (*symmetry_);
        }

        inline matrix3d<double> const& lattice_vectors() const
        {
            return lattice_vectors_;
        }

        inline matrix3d<double> const& inverse_lattice_vectors() const
        {
            return inverse_lattice_vectors_;
        }

        inline matrix3d<double> const& reciprocal_lattice_vectors() const
        {
            return reciprocal_lattice_vectors_;
        }
        
        /// Return a single lattice vector.
        inline vector3d<double> lattice_vector(int idx__) const
        {
            return vector3d<double>(lattice_vectors_(0, idx__), lattice_vectors_(1, idx__), lattice_vectors_(2, idx__));
        }

        inline void import(Unit_cell_input const& inp__)
        {
            PROFILE("sirius::Unit_cell::import");
 
            if (inp__.exist_) {
                /* first, load all types */
                for (int iat = 0; iat < (int)inp__.labels_.size(); iat++) {
                    auto label = inp__.labels_[iat];
                    auto fname = inp__.atom_files_.at(label);
                    add_atom_type(label, fname);
                }
                /* then load atoms */
                for (int iat = 0; iat < (int)inp__.labels_.size(); iat++) {
                    auto label = inp__.labels_[iat];
                    auto fname = inp__.atom_files_.at(label);
                    for (size_t ia = 0; ia < inp__.coordinates_[iat].size(); ia++) {
                        auto v = inp__.coordinates_[iat][ia];
                        vector3d<double> p(v[0], v[1], v[2]);
                        vector3d<double> f(v[3], v[4], v[5]);
                        add_atom(label, p, f);
                    }
                }

                set_lattice_vectors(inp__.a0_, inp__.a1_, inp__.a2_);
            }
        }
        
        Simulation_parameters const& parameters() const
        {
            return parameters_;
        }
};
    
inline void Unit_cell::initialize()
{
    PROFILE("sirius::Unit_cell::initialize");

    /* split number of atom between all MPI ranks */
    spl_num_atoms_ = splindex<block>(num_atoms(), comm_.size(), comm_.rank());

    /* initialize atom types */
    int offs_lo{0};
    for (int iat = 0; iat < num_atom_types(); iat++) {
        atom_type(iat).init(offs_lo);
        max_num_mt_points_        = std::max(max_num_mt_points_, atom_type(iat).num_mt_points());
        max_mt_basis_size_        = std::max(max_mt_basis_size_, atom_type(iat).mt_basis_size());
        max_mt_radial_basis_size_ = std::max(max_mt_radial_basis_size_, atom_type(iat).mt_radial_basis_size());
        max_mt_aw_basis_size_     = std::max(max_mt_aw_basis_size_, atom_type(iat).mt_aw_basis_size());
        max_mt_lo_basis_size_     = std::max(max_mt_lo_basis_size_,  atom_type(iat).mt_lo_basis_size());
        lmax_                     = std::max(lmax_, atom_type(iat).indexr().lmax());
        offs_lo += atom_type(iat).mt_lo_basis_size(); 
    }
    
    /* find the charges */
    for (int i = 0; i < num_atoms(); i++) {
        total_nuclear_charge_  += atom(i).zn();
        num_core_electrons_    += atom(i).type().num_core_electrons();
        num_valence_electrons_ += atom(i).type().num_valence_electrons();
    }
    num_electrons_ = num_core_electrons_ + num_valence_electrons_;
    
    /* initialize atoms */
    for (int ia = 0; ia < num_atoms(); ia++) {
        atom(ia).init(mt_aw_basis_size_, mt_lo_basis_size_, mt_basis_size_);
        mt_aw_basis_size_ += atom(ia).mt_aw_basis_size();
        mt_lo_basis_size_ += atom(ia).mt_lo_basis_size();
        mt_basis_size_    += atom(ia).mt_basis_size();
    }

    assert(mt_basis_size_ == mt_aw_basis_size_ + mt_lo_basis_size_);
    
    auto v0 = lattice_vector(0);
    auto v1 = lattice_vector(1);
    auto v2 = lattice_vector(2);

    double r = std::max(v0.length(), std::max(v1.length(), v2.length()));
    find_nearest_neighbours(r);

    if (parameters_.full_potential()) {
        /* find new MT radii and initialize radial grid */
        if (parameters_.auto_rmt()) {
            std::vector<double> Rmt = find_mt_radii();
            for (int iat = 0; iat < num_atom_types(); iat++) {
                atom_type(iat).set_mt_radius(Rmt[iat]);
                atom_type(iat).set_radial_grid();
            }
        }
        
        int ia, ja;
        if (check_mt_overlap(ia, ja)) {
            std::stringstream s;
            s << "overlaping muffin-tin spheres for atoms " << ia << "(" << atom(ia).type().symbol() << ")" << " and " 
              << ja << "(" << atom(ja).type().symbol() << ")" << std::endl 
              << "  radius of atom " << ia << " : " << atom(ia).mt_radius() << std::endl
              << "  radius of atom " << ja << " : " << atom(ja).mt_radius() << std::endl
              << "  distance : " << nearest_neighbours_[ia][1].distance << " " << nearest_neighbours_[ja][1].distance;
            TERMINATE(s);
        }
        
        min_mt_radius_ = 1e100;
        max_mt_radius_ = 0;
        for (int i = 0; i < num_atom_types(); i++) {
             min_mt_radius_ = std::min(min_mt_radius_, atom_type(i).mt_radius());
             max_mt_radius_ = std::max(max_mt_radius_, atom_type(i).mt_radius());
        }
    }
    
    get_symmetry();
    
    spl_num_atom_symmetry_classes_ = splindex<block>(num_atom_symmetry_classes(), comm_.size(), comm_.rank());
    
    volume_mt_ = 0.0;
    if (parameters_.full_potential()) {
        for (int ia = 0; ia < num_atoms(); ia++) {
            volume_mt_ += fourpi * std::pow(atom(ia).mt_radius(), 3) / 3.0; 
        }
    }
    
    volume_it_ = omega() - volume_mt_;

    mt_aw_basis_descriptors_.resize(mt_aw_basis_size_);
    for (int ia = 0, n = 0; ia < num_atoms(); ia++) {
        for (int xi = 0; xi < atom(ia).mt_aw_basis_size(); xi++, n++) {
            mt_aw_basis_descriptors_[n].ia = ia;
            mt_aw_basis_descriptors_[n].xi = xi;
        }
    }

    mt_lo_basis_descriptors_.resize(mt_lo_basis_size_);
    for (int ia = 0, n = 0; ia < num_atoms(); ia++) {
        for (int xi = 0; xi < atom(ia).mt_lo_basis_size(); xi++, n++) {
            mt_lo_basis_descriptors_[n].ia = ia;
            mt_lo_basis_descriptors_[n].xi = xi;
        }
    }

    init_paw();
}

inline void Unit_cell::get_symmetry()
{
    PROFILE("sirius::Unit_cell::get_symmetry");
    
    if (num_atoms() == 0) {
        return;
    }
    
    if (atom_symmetry_classes_.size() != 0) {
        atom_symmetry_classes_.clear();
        for (int ia = 0; ia < num_atoms(); ia++) {
            atom(ia).set_symmetry_class(nullptr);
        }
    }

    if (symmetry_ != nullptr) {
        TERMINATE("Symmetry() object is already allocated");
    }

    mdarray<double, 2> positions(3, num_atoms());
    mdarray<double, 2> spins(3, num_atoms());
    std::vector<int> types(num_atoms());
    for (int ia = 0; ia < num_atoms(); ia++) {
        auto vp = atom(ia).position();
        auto vf = atom(ia).vector_field();
        for (int x: {0, 1, 2}) {
            positions(x, ia) = vp[x];
            spins(x, ia) = vf[x];
        }
        types[ia] = atom(ia).type_id();
    }
    
    symmetry_ = std::unique_ptr<Symmetry>(new Symmetry(lattice_vectors_, num_atoms(), positions, spins, types,
                                                       parameters_.spglib_tolerance()));

    int atom_class_id{-1};
    std::vector<int> asc(num_atoms(), -1);
    for (int i = 0; i < num_atoms(); i++) {
        /* if symmetry class is not assigned to this atom */
        if (asc[i] == -1) {
            /* take next id */
            atom_class_id++;
            atom_symmetry_classes_.push_back(std::move(Atom_symmetry_class(atom_class_id, atoms_[i].type())));
            
            /* scan all atoms */
            for (int j = 0; j < num_atoms(); j++) {
                bool is_equal = (equivalent_atoms_.size()) ? (equivalent_atoms_[j] == equivalent_atoms_[i]) :  
                                (symmetry_->atom_symmetry_class(j) == symmetry_->atom_symmetry_class(i));
                /* assign new class id for all equivalent atoms */
                if (is_equal) {
                    asc[j] = atom_class_id;
                    atom_symmetry_classes_.back().add_atom_id(j);
                }
            }
        }
    }

    for (auto& e: atom_symmetry_classes_) {
        for (int i = 0; i < e.num_atoms(); i++) {
            int ia = e.atom_id(i);
            atoms_[ia].set_symmetry_class(&e);
        }
    }
    
    assert(num_atom_symmetry_classes() != 0);
}

inline std::vector<double> Unit_cell::find_mt_radii()
{
    if (nearest_neighbours_.size() == 0) {
        TERMINATE("array of nearest neighbours is empty");
    }

    std::vector<double> Rmt(num_atom_types(), 1e10);
    
    if (parameters_.auto_rmt() == 1) {
        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();
            if (nearest_neighbours_[ia].size() > 1) {
                int ja = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja).type_id();
                /* don't allow spheres to touch: take a smaller value than half a distance */
                double R = std::min(parameters_.rmt_max(), 0.95 * nearest_neighbours_[ia][1].distance / 2);
                /* take minimal R for the given atom type */
                Rmt[id1] = std::min(R, Rmt[id1]);
                Rmt[id2] = std::min(R, Rmt[id2]);
            } else {
                Rmt[id1] = parameters_.rmt_max();
            }
        }
    }

    if (parameters_.auto_rmt() == 2) {
        std::vector<double> scale(num_atom_types(), 1e10);

        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();
            if (nearest_neighbours_[ia].size() > 1) {
                int ja = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja).type_id();

                double d = nearest_neighbours_[ia][1].distance;
                double s = 0.95 * d / (atom_type(id1).mt_radius() + atom_type(id2).mt_radius());
                scale[id1] = std::min(s, scale[id1]);
                scale[id2] = std::min(s, scale[id2]);
            } else {
                scale[id1] = parameters_.rmt_max() / atom_type(id1).mt_radius();
            }
        }

        for (int iat = 0; iat < num_atom_types(); iat++) {
            Rmt[iat] = std::min(parameters_.rmt_max(),  atom_type(iat).mt_radius() * scale[iat]);
        }
    }
    
    /* Suppose we have 3 different atoms. First we determint Rmt between 1st and 2nd atom, 
     * then we determine Rmt between (let's say) 2nd and 3rd atom and at this point we reduce 
     * the Rmt of the 2nd atom. This means that the 1st atom gets a possibility to expand if 
     * it is far from the 3rd atom. */
    bool inflate = true;
    
    if (inflate) {
        std::vector<bool> scale_Rmt(num_atom_types(), true);
        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();

            if (nearest_neighbours_[ia].size() > 1) {
                int ja = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja).type_id();
                double dist = nearest_neighbours_[ia][1].distance;
            
                if (Rmt[id1] + Rmt[id2] > dist * 0.94) {
                    scale_Rmt[id1] = false;
                    scale_Rmt[id2] = false;
                }
            }
        }

        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();
            if (nearest_neighbours_[ia].size() > 1) {
                int ja = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja).type_id();
                double dist = nearest_neighbours_[ia][1].distance;
                
                if (scale_Rmt[id1]) {
                    Rmt[id1] = std::min(parameters_.rmt_max(), 0.95 * (dist - Rmt[id2]));
                }
            }
        }
    }
    
    for (int i = 0; i < num_atom_types(); i++) {
        if (Rmt[i] < 0.3) {
            TERMINATE("Muffin-tin radius is too small");
        }
    }

    return Rmt;
}

inline bool Unit_cell::check_mt_overlap(int& ia__, int& ja__)
{
    if (num_atoms() != 0 && nearest_neighbours_.size() == 0) {
        TERMINATE("array of nearest neighbours is empty");
    }

    for (int ia = 0; ia < num_atoms(); ia++) {
        /* first atom is always the central one itself */
        if (nearest_neighbours_[ia].size() <= 1) {
            continue;
            //std::stringstream s;
            //s << "array of nearest neighbours for atom " << ia << " is empty";
            //TERMINATE(s);
        }

        int ja = nearest_neighbours_[ia][1].atom_id;
        double dist = nearest_neighbours_[ia][1].distance;
        
        if (atom(ia).mt_radius() + atom(ja).mt_radius() >= dist) {
            ia__ = ia;
            ja__ = ja;
            return true;
        }
    }
    
    return false;
}

inline void Unit_cell::print_info(int verbosity_)
{
    printf("\n");
    printf("Unit cell\n");
    for (int i = 0; i < 80; i++) {
        printf("-");
    }
    printf("\n");
    
    printf("lattice vectors\n");
    for (int i = 0; i < 3; i++) {
        printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors_(0, i), 
                                                             lattice_vectors_(1, i), 
                                                             lattice_vectors_(2, i)); 
    }
    printf("reciprocal lattice vectors\n");
    for (int i = 0; i < 3; i++) {
        printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors_(0, i), 
                                                             reciprocal_lattice_vectors_(1, i), 
                                                             reciprocal_lattice_vectors_(2, i));
    }
    printf("\n");
    printf("unit cell volume : %18.8f [a.u.^3]\n", omega());
    printf("1/sqrt(omega)    : %18.8f\n", 1.0 / sqrt(omega()));
    printf("MT volume        : %f (%5.2f%%)\n", volume_mt(), volume_mt() * 100 / omega());
    printf("IT volume        : %f (%5.2f%%)\n", volume_it(), volume_it() * 100 / omega());
    
    printf("\n"); 
    printf("number of atom types : %i\n", num_atom_types());
    for (int i = 0; i < num_atom_types(); i++) {
        int id = atom_type(i).id();
        printf("type id : %i   symbol : %2s   mt_radius : %10.6f\n", id, atom_type(i).symbol().c_str(), 
                                                                         atom_type(i).mt_radius()); 
    }

    printf("number of atoms : %i\n", num_atoms());
    printf("number of symmetry classes : %i\n", num_atom_symmetry_classes());
    if (!parameters_.full_potential()) {
        printf("number of PAW atoms : %i\n", num_paw_atoms());
    }
    if (verbosity_ >= 2) {
        printf("\n");
        printf("atom id              position            type id    class id\n");
        printf("------------------------------------------------------------\n");
        for (int i = 0; i < num_atoms(); i++) {
            auto pos = atom(i).position();
            printf("%6i      %f %f %f   %6i      %6i\n", i, pos[0], pos[1], pos[2], atom(i).type_id(), atom(i).symmetry_class_id());
        }
   
        printf("\n");
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            printf("class id : %i   atom id : ", ic);
            for (int i = 0; i < atom_symmetry_class(ic).num_atoms(); i++) {
                printf("%i ", atom_symmetry_class(ic).atom_id(i));
            }
            printf("\n");
        }
    }

    if (symmetry_ != nullptr) {
        printf("\n");
        printf("space group number   : %i\n", symmetry_->spacegroup_number());
        printf("international symbol : %s\n", symmetry_->international_symbol().c_str());
        printf("Hall symbol          : %s\n", symmetry_->hall_symbol().c_str());
        printf("number of operations : %i\n", symmetry_->num_mag_sym());
        printf("transformation matrix : \n");
        auto tm = symmetry_->transformation_matrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%12.6f ", tm(i, j));
            }
            printf("\n");
        }
        printf("origin shift : \n");
        auto t = symmetry_->origin_shift();
        printf("%12.6f %12.6f %12.6f\n", t[0], t[1], t[2]);
        
        if (verbosity_ >= 2) {
            printf("symmetry operations  : \n");
            for (int isym = 0; isym < symmetry_->num_mag_sym(); isym++) {
                auto R = symmetry_->magnetic_group_symmetry(isym).spg_op.R;
                auto t = symmetry_->magnetic_group_symmetry(isym).spg_op.t;
                auto S = symmetry_->magnetic_group_symmetry(isym).spin_rotation;

                printf("isym : %i\n", isym);
                printf("R : ");
                for (int i = 0; i < 3; i++) {
                    if (i) {
                        printf("    ");
                    }
                    for (int j = 0; j < 3; j++) {
                        printf("%3i ", R(i, j));
                    }
                    printf("\n");
                }
                printf("T : ");
                for (int j = 0; j < 3; j++) {
                    printf("%8.4f ", t[j]);
                }
                printf("\n");
                printf("S : ");
                for (int i = 0; i < 3; i++) {
                    if (i) {
                        printf("    ");
                    }
                    for (int j = 0; j < 3; j++) {
                        printf("%8.4f ", S(i, j));
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }
    }
}

inline unit_cell_parameters_descriptor Unit_cell::unit_cell_parameters()
{
    unit_cell_parameters_descriptor d;

    vector3d<double> v0(lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0));
    vector3d<double> v1(lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1));
    vector3d<double> v2(lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2));

    d.a = v0.length();
    d.b = v1.length();
    d.c = v2.length();

    d.alpha = acos((v1 * v2) / d.b / d.c) * 180 / pi;
    d.beta  = acos((v0 * v2) / d.a / d.c) * 180 / pi;
    d.gamma = acos((v0 * v1) / d.a / d.b) * 180 / pi;

    return d;
}

inline void Unit_cell::write_cif()
{
    if (comm_.rank() == 0)
    {
        FILE* fout = fopen("unit_cell.cif", "w");

        auto d = unit_cell_parameters();
        
        fprintf(fout, "_cell_length_a %f\n", d.a);
        fprintf(fout, "_cell_length_b %f\n", d.b);
        fprintf(fout, "_cell_length_c %f\n", d.c);
        fprintf(fout, "_cell_angle_alpha %f\n", d.alpha);
        fprintf(fout, "_cell_angle_beta %f\n", d.beta);
        fprintf(fout, "_cell_angle_gamma %f\n", d.gamma);

        //fprintf(fout, "loop_\n");
        //fprintf(fout, "_symmetry_equiv_pos_as_xyz\n");

        fprintf(fout, "loop_\n");
        fprintf(fout, "_atom_site_label\n");
        fprintf(fout, "_atom_type_symbol\n");
        fprintf(fout, "_atom_site_fract_x\n");
        fprintf(fout, "_atom_site_fract_y\n");
        fprintf(fout, "_atom_site_fract_z\n");
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            auto pos = atom(ia).position();
            fprintf(fout,"%i %s %f %f %f\n", ia + 1, atom(ia).type().label().c_str(), pos[0], pos[1], pos[2]);
        }
        fclose(fout);
    }
}

inline json Unit_cell::serialize()
{
    json dict;

    dict["lattice_vectors"] = {{lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0)},
                               {lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1)},
                               {lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2)}};
    dict["atom_types"] = json::array();
    for (int iat = 0; iat < num_atom_types(); iat++) {
        dict["atom_types"].push_back(atom_type(iat).label());
    }
    dict["atom_files"] = json::object();
    for (int iat = 0; iat < num_atom_types(); iat++) {
        dict["atom_files"][atom_type(iat).label()] = atom_type(iat).file_name();
    }
    dict["atoms"] = json::object();
    for (int iat = 0; iat < num_atom_types(); iat++) {
        dict["atoms"][atom_type(iat).label()] = json::array();
        for (int i = 0; i < atom_type(iat).num_atoms(); i++) {
            int ia = atom_type(iat).atom_id(i);
            auto v = atom(ia).position();
            dict["atoms"][atom_type(iat).label()].push_back({v[0], v[1], v[2]});
        }
    }
    return std::move(dict);
}

inline void Unit_cell::find_nearest_neighbours(double cluster_radius)
{
    PROFILE("sirius::Unit_cell::find_nearest_neighbours");

    auto max_frac_coord = find_translations(cluster_radius, lattice_vectors_);
   
    nearest_neighbours_.clear();
    nearest_neighbours_.resize(num_atoms());

    #pragma omp parallel for default(shared)
    for (int ia = 0; ia < num_atoms(); ia++) {
        auto iapos = get_cartesian_coordinates(atom(ia).position());
        
        std::vector<nearest_neighbour_descriptor> nn;

        std::vector< std::pair<double, int> > nn_sort;

        for (int i0 = -max_frac_coord[0]; i0 <= max_frac_coord[0]; i0++) {
            for (int i1 = -max_frac_coord[1]; i1 <= max_frac_coord[1]; i1++) {
                for (int i2 = -max_frac_coord[2]; i2 <= max_frac_coord[2]; i2++) {
                    nearest_neighbour_descriptor nnd;
                    nnd.translation[0] = i0;
                    nnd.translation[1] = i1;
                    nnd.translation[2] = i2;
                    
                    auto vt = get_cartesian_coordinates<int>(nnd.translation);
                    
                    for (int ja = 0; ja < num_atoms(); ja++) {
                        nnd.atom_id = ja;

                        auto japos = get_cartesian_coordinates(atom(ja).position());

                        vector3d<double> v = japos + vt - iapos;

                        nnd.distance = v.length();
                        
                        if (nnd.distance <= cluster_radius) {
                            nn.push_back(nnd);

                            nn_sort.push_back(std::pair<double, int>(nnd.distance, (int)nn.size() - 1));
                        }
                    }

                }
            }
        }
        
        std::sort(nn_sort.begin(), nn_sort.end());
        nearest_neighbours_[ia].resize(nn.size());
        for (int i = 0; i < (int)nn.size(); i++) {
            nearest_neighbours_[ia][i] = nn[nn_sort[i].second];
        }
    }

    //== if (Platform::mpi_rank() == 0) // TODO: move to a separate task
    //== {
    //==     FILE* fout = fopen("nghbr.txt", "w");
    //==     for (int ia = 0; ia < num_atoms(); ia++)
    //==     {
    //==         fprintf(fout, "Central atom: %s (%i)\n", atom(ia).type()->symbol().c_str(), ia);
    //==         for (int i = 0; i < 80; i++) fprintf(fout, "-");
    //==         fprintf(fout, "\n");
    //==         fprintf(fout, "atom (  id)       D [a.u.]    translation  R\n");
    //==         for (int i = 0; i < 80; i++) fprintf(fout, "-");
    //==         fprintf(fout, "\n");
    //==         for (int i = 0; i < (int)nearest_neighbours_[ia].size(); i++)
    //==         {
    //==             int ja = nearest_neighbours_[ia][i].atom_id;
    //==             fprintf(fout, "%4s (%4i)   %12.6f\n", atom(ja)->type()->symbol().c_str(), ja, 
    //==                                                   nearest_neighbours_[ia][i].distance);
    //==         }
    //==         fprintf(fout, "\n");
    //==     }
    //==     fclose(fout);
    //== }
}

inline bool Unit_cell::is_point_in_mt(vector3d<double> vc, int& ja, int& jr, double& dr, double tp[2]) const
{
    /* reduce coordinates to the primitive unit cell */
    auto vr = reduce_coordinates(get_fractional_coordinates(vc));

    for (int ia = 0; ia < num_atoms(); ia++)
    {
        for (int i0 = -1; i0 <= 1; i0++)
        {
            for (int i1 = -1; i1 <= 1; i1++)
            {
                for (int i2 = -1; i2 <= 1; i2++)
                {
                    /* atom position */
                    vector3d<double> posf = vector3d<double>(i0, i1, i2) + atom(ia).position();

                    /* vector connecting center of atom and reduced point */
                    vector3d<double> vf = vr.first - posf;
                    
                    /* convert to spherical coordinates */
                    auto vs = SHT::spherical_coordinates(get_cartesian_coordinates(vf));

                    if (vs[0] < atom(ia).mt_radius())
                    {
                        ja = ia;
                        tp[0] = vs[1]; // theta
                        tp[1] = vs[2]; // phi

                        if (vs[0] < atom(ia).type().radial_grid(0))
                        {
                            jr = 0;
                            dr = 0.0;
                        }
                        else
                        {
                            for (int ir = 0; ir < atom(ia).num_mt_points() - 1; ir++)
                            {
                                if (vs[0] >= atom(ia).type().radial_grid(ir) && vs[0] < atom(ia).type().radial_grid(ir + 1))
                                {
                                    jr = ir;
                                    dr = vs[0] - atom(ia).type().radial_grid(ir);
                                    break;
                                }
                            }
                        }

                        return true;
                    }

                }
            }
        }
    }
    ja = -1;
    jr = -1;
    return false;
}

inline void Unit_cell::generate_radial_functions()
{
    PROFILE("sirius::Unit_cell::generate_radial_functions");
   
    for (int icloc = 0; icloc < (int)spl_num_atom_symmetry_classes().local_size(); icloc++) {
        int ic = spl_num_atom_symmetry_classes(icloc);
        atom_symmetry_class(ic).generate_radial_functions(parameters_.valence_relativity());
    }

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
        int rank = spl_num_atom_symmetry_classes().local_rank(ic);
        atom_symmetry_class(ic).sync_radial_functions(comm_, rank);
    }
    
    if (parameters_.control().verbosity_ >= 1) {
        runtime::pstdout pout(comm_);
        
        for (int icloc = 0; icloc < (int)spl_num_atom_symmetry_classes().local_size(); icloc++) {
            int ic = spl_num_atom_symmetry_classes(icloc);
            atom_symmetry_class(ic).write_enu(pout);
        }

        if (comm_.rank() == 0) {
            printf("\n");
            printf("Linearization energies\n");
        }
    }
    if (parameters_.control().verbosity_ >= 4 && comm_.rank() == 0) {
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            atom_symmetry_class(ic).dump_lo();
        }
    }
}

inline void Unit_cell::generate_radial_integrals()
{
    PROFILE("sirius::Unit_cell::generate_radial_integrals");

    for (int icloc = 0; icloc < spl_num_atom_symmetry_classes().local_size(); icloc++) {
        int ic = spl_num_atom_symmetry_classes(icloc);
        atom_symmetry_class(ic).generate_radial_integrals(parameters_.valence_relativity());
    }

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
        int rank = spl_num_atom_symmetry_classes().local_rank(ic);
        atom_symmetry_class(ic).sync_radial_integrals(comm_, rank);
    }

    for (int ialoc = 0; ialoc < spl_num_atoms_.local_size(); ialoc++) {
        int ia = spl_num_atoms_[ialoc];
        atom(ia).generate_radial_integrals(parameters_.processing_unit(), mpi_comm_self());
    }
    
    for (int ia = 0; ia < num_atoms(); ia++) {
        int rank = spl_num_atoms().local_rank(ia);
        atom(ia).sync_radial_integrals(comm_, rank);
    }
}

inline std::string Unit_cell::chemical_formula()
{
    std::string name;
    for (int iat = 0; iat < num_atom_types(); iat++)
    {
        name += atom_type(iat).symbol();
        int n = 0;
        for (int ia = 0; ia < num_atoms(); ia++)
        {
            if (atom(ia).type_id() == atom_type(iat).id()) n++;
        }
        if (n != 1) 
        {
            std::stringstream s;
            s << n;
            name = (name + s.str());
        }
    }

    return name;
}

} // namespace

#endif // __UNIT_CELL_H__

