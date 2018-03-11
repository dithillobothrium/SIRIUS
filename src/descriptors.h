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

/** \file descriptors.h
 *
 *  \brief Descriptors for various data structures
 */

#ifndef __DESCRIPTORS_H__
#define __DESCRIPTORS_H__

#include "mdarray.hpp"
#include "utils.h"

/// Describes single atomic level.
struct atomic_level_descriptor
{
    /// Principal quantum number.
    int n;

    /// Angular momentum quantum number.
    int l;

    /// Quantum number k.
    int k;

    /// Level occupancy.
    double occupancy;

    /// True if this is a core level.
    bool core;
};

/// Describes radial solution.
struct radial_solution_descriptor
{
    /// Principal quantum number.
    int n;

    /// Angular momentum quantum number.
    int l;

    /// Order of energy derivative.
    int dme;

    /// Energy of the solution.
    double enu;

    /// Automatically determine energy.
    int auto_enu;
};

/// Set of radial solution descriptors, used to construct augmented waves or local orbitals.
typedef std::vector<radial_solution_descriptor> radial_solution_descriptor_set;

/// Descriptor of a local orbital radial function.
struct local_orbital_descriptor
{
    /// Orbital quantum number \f$ \ell \f$.
    int l;

    /// Total angular momentum used in pseudopotential SO code.
    double total_angular_momentum; // TODO: is this neccessary?

    /// Set of radial solution descriptors.
    /** Local orbital is constructed from at least two radial functions in order to make it zero at the
     *  muffin-tin sphere boundary. */
    radial_solution_descriptor_set rsd_set;
};


///// Descriptor of the pseudopotential.
//struct pseudopotential_descriptor
//{
//    /// Occubations of atomic states.
//    /** Length of vector is the same as the number of beta projectors and all_elec_wfc and pseudo_wfc */
//    //std::vector<double> occupations;
//
//    /// total angular momentum j of the (hubbard) wave functions
//    //std::vector<double> total_angular_momentum_wfs;
//
//    /// total angular momentum j of the (hubbard) wave functions
//    //std::vector<double> occupation_wfs;
//};

/// Descriptor of an atom in a list of nearest neigbours for each atom.
/** See sirius::Unit_cell::find_nearest_neighbours() for the details of usage. */
struct nearest_neighbour_descriptor
{
    /// Index of the neighbouring atom.
    int atom_id;

    /// Translation in fractional coordinates.
    geometry3d::vector3d<int> translation;

    /// Distance from the central atom.
    double distance;
};

struct unit_cell_parameters_descriptor
{
    double a;
    double b;
    double c;
    double alpha;
    double beta;
    double gamma;
};

/// Descriptor of the local-orbital part of the LAPW+lo basis.
struct lo_basis_descriptor
{
    /// Index of atom.
    uint16_t ia;

    /// Index of orbital quantum number \f$ \ell \f$.
    uint8_t l;

    /// Combined lm index.
    uint16_t lm;

    /// Order of the local orbital radial function for the given orbital quantum number \f$ \ell \f$.
    /** All radial functions for the given orbital quantum number \f$ \ell \f$ are ordered in the following way:
     *  augmented radial functions come first followed by the local orbital radial function. */
    uint8_t order;

    /// Index of the local orbital radial function.
    uint8_t idxrf;
};

#endif
