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

/** \file smooth_periodic_function.h
 *   
 *  \brief Contains declaration and implementation of sirius::Smooth_periodic_function and 
 *         sirius::Smooth_periodic_function_gradient classes.
 */

#ifndef __SMOOTH_PERIODIC_FUNCTION_H__
#define __SMOOTH_PERIODIC_FUNCTION_H__

namespace sirius {

/// Representation of a smooth (Fourier-transformable) periodic function.
/** The class is designed to handle periodic functions such as density or potential, defined on a regular FFT grid.
 *  The following functionality is expected:
 *    - access to real-space values
 *    - access to plane-wave coefficients
 *    - distribution of plane-wave coefficients over entire communicator
 *    - Fourier transformation using FFT communicator
 *    - gather PW coefficients into global array
 */
template <typename T>
class Smooth_periodic_function
{
    protected:

        /// FFT driver.
        FFT3D* fft_{nullptr};

        /// Distribution of G-vectors.
        Gvec const* gvec_{nullptr};
        
        /// Function on the regular real-space grid.
        mdarray<T, 1> f_rg_;
        
        /// Local set of plane-wave expansion coefficients.
        mdarray<double_complex, 1> f_pw_local_;

        /// Storage of the PW coefficients for the FFT transformation.
        mdarray<double_complex, 1> f_pw_fft_;

        /// Distribution of G-vectors inside FFT slab.
        block_data_descriptor gvec_fft_slab_;
        
        /// Gather plane-wave coefficients for the subsequent FFT call.
        inline void gather_f_pw_fft()
        {
            int rank = fft_->comm().rank() * gvec_->comm_ortho_fft().size() + gvec_->comm_ortho_fft().rank();
            /* collect scattered PW coefficients */
            gvec_->comm_ortho_fft().allgather(f_pw_local_.at<CPU>(),
                                              gvec_->gvec_count(rank),
                                              f_pw_fft_.at<CPU>(),
                                              gvec_fft_slab_.counts.data(), 
                                              gvec_fft_slab_.offsets.data());
        }

    public:
        
        /// Default constructor.
        Smooth_periodic_function() 
        {
        }

        Smooth_periodic_function(FFT3D& fft__, Gvec const& gvec__)
            : fft_(&fft__)
            , gvec_(&gvec__)
        {
            f_rg_       = mdarray<T, 1>(fft_->local_size(), memory_t::host, "Smooth_periodic_function.f_rg_");
            f_pw_fft_   = mdarray<double_complex, 1>(gvec_->partition().gvec_count_fft(), memory_t::host, "Smooth_periodic_function.f_pw_fft_");
            f_pw_local_ = mdarray<double_complex, 1>(gvec_->count(), memory_t::host, "Smooth_periodic_function.f_pw_local_");

            /* check ordering of mpi ranks */
            int rank = fft_->comm().rank() * gvec_->comm_ortho_fft().size() + gvec_->comm_ortho_fft().rank();
            if (rank != gvec_->comm().rank()) {
                TERMINATE("wrong order of MPI ranks");
            }

            gvec_fft_slab_ = block_data_descriptor(gvec_->comm_ortho_fft().size());
            for (int i = 0; i < gvec_->comm_ortho_fft().size(); i++) {
                gvec_fft_slab_.counts[i] = gvec_->gvec_count(fft_->comm().rank() * gvec_->comm_ortho_fft().size() + i);
            }
            gvec_fft_slab_.calc_offsets();
        }

        inline void zero()
        {
            f_rg_.zero();
        }

        inline T& f_rg(int ir__)
        {
            return f_rg_(ir__);
        }

        inline T const& f_rg(int ir__) const
        {
            return f_rg_(ir__);
        }
        
        inline double_complex& f_pw_local(int ig__)
        {
            return f_pw_local_(ig__);
        }

        inline double_complex const& f_pw_local(int ig__) const
        {
            return f_pw_local_(ig__);
        }

        inline double_complex& f_pw_fft(int ig__)
        {
            return f_pw_fft_(ig__);
        }

        inline double_complex f_0() const
        {
            double_complex z;
            if (gvec_->comm().rank() == 0) {
                z = f_pw_local_(0);
            }
            gvec_->comm().bcast(&z, 1, 0);
            return z;
        }

        FFT3D& fft()
        {
            assert(fft_ != nullptr);
            return *fft_;
        }

        FFT3D const& fft() const
        {
            assert(fft_ != nullptr);
            return *fft_;
        }

        Gvec const& gvec() const
        {
            assert(gvec_ != nullptr);
            return *gvec_;
        }

        void fft_transform(int direction__)
        {
            PROFILE("sirius::Smooth_periodic_function::fft_transform");

            assert(gvec_ != nullptr);

            switch (direction__) {
                case 1: {
                    gather_f_pw_fft();
                    fft_->transform<1>(gvec_->partition(), f_pw_fft_.at<CPU>());
                    fft_->output(f_rg_.template at<CPU>());
                    break;
                }
                case -1: {
                    fft_->input(f_rg_.template at<CPU>());
                    fft_->transform<-1>(gvec_->partition(), f_pw_fft_.at<CPU>());
                    int count  = gvec_fft_slab_.counts[gvec_->comm_ortho_fft().rank()];
                    int offset = gvec_fft_slab_.offsets[gvec_->comm_ortho_fft().rank()];
                    std::memcpy(f_pw_local_.at<CPU>(), f_pw_fft_.at<CPU>(offset), count * sizeof(double_complex));
                    break;
                }
                default: {
                    TERMINATE("wrong fft direction");
                }
            }
        }

        inline std::vector<double_complex> gather_f_pw()
        {
            PROFILE("sirius::Smooth_periodic_function::gather_f_pw");

            gather_f_pw_fft();

            std::vector<double_complex> fpw(gvec_->num_gvec());
            fft_->comm().allgather(f_pw_fft_.at<CPU>(), fpw.data(), gvec_->partition().gvec_offset_fft(), gvec_->partition().gvec_count_fft());

            return std::move(fpw);
        }

        inline void scatter_f_pw(std::vector<double_complex> const& f_pw__)
        {
            std::copy(&f_pw__[gvec_->offset()], &f_pw__[gvec_->offset()] + gvec_->count(), &f_pw_local_(0));
        }

        void add(Smooth_periodic_function<T> const& g__, double alpha__ = 1)
        {
            #pragma omp parallel for schedule(static)
            for (int irloc = 0; irloc < this->fft_->local_size(); irloc++) {
                this->f_rg_(irloc) += g__.f_rg(irloc) * alpha__;
            }
        }

        inline T checksum_rg() const
        {
            T cs = this->f_rg_.checksum();
            this->fft_->comm().allreduce(&cs, 1);
            return cs;
        }

        /// Compute inner product <f|g>
        T inner(Smooth_periodic_function<T> const& g__) const
        {
            PROFILE("sirius::Periodic_function::inner");

            assert(this->fft_ == g__.fft_);

            T result_rg{0};

            #pragma omp parallel
            {
                T rt{0};

                #pragma omp for schedule(static)
                for (int irloc = 0; irloc < this->fft_->local_size(); irloc++) {
		  rt += type_wrapper<T>::bypass(std::conj(this->f_rg(irloc))) * g__.f_rg(irloc);
                }

                #pragma omp critical
                result_rg += rt;
            }
            double omega = std::pow(twopi, 3) / std::abs(this->gvec_->lattice_vectors().det());

            result_rg *= (omega / this->fft_->size());

            this->fft_->comm().allreduce(&result_rg, 1);

            return result_rg;
        }

};

/// Gradient of the smooth periodic function.
template<typename T>
class Smooth_periodic_function_gradient
{
    private:

        /// FFT driver.
        FFT3D* fft_{nullptr};

        /// Distribution of G-vectors.
        Gvec const* gvec_{nullptr};
        
        std::array<Smooth_periodic_function<T>, 3> grad_f_;

    public:

        Smooth_periodic_function_gradient()
        {
        }

        Smooth_periodic_function_gradient(FFT3D& fft__, Gvec const& gvec__)
            : fft_(&fft__)
            , gvec_(&gvec__)
        {
            for (int x: {0, 1, 2}) {
                grad_f_[x] = Smooth_periodic_function<T>(fft__, gvec__);
            }
        }

        Smooth_periodic_function<T>& operator[](const int idx__)
        {
            return grad_f_[idx__];
        }

        FFT3D& fft() const
        {
            assert(fft_ != nullptr);
            return *fft_;
        }

        Gvec const& gvec() const
        {
            assert(gvec_ != nullptr);
            return *gvec_;
        }
};

/// Gradient of the function in the plane-wave domain.
inline Smooth_periodic_function_gradient<double> gradient(Smooth_periodic_function<double>& f__)
{
    Smooth_periodic_function_gradient<double> g(f__.fft(), f__.gvec());

    #pragma omp parallel for
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        int ig = f__.gvec().offset() + igloc;
        auto G = f__.gvec().gvec_cart(ig);
        for (int x: {0, 1, 2}) {
            g[x].f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(0, G[x]);
        }
    }
    return std::move(g);
}

/// Laplacian of the function in the plane-wave domain.
inline Smooth_periodic_function<double> laplacian(Smooth_periodic_function<double>& f__)
{
    Smooth_periodic_function<double> g(f__.fft(), f__.gvec());
    
    #pragma omp parallel for
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        int ig = f__.gvec().offset() + igloc;
        auto G = f__.gvec().gvec_cart(ig);
        g.f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(-std::pow(G.length(), 2), 0);
    }

    return std::move(g);
}

template <typename T>
Smooth_periodic_function<T> operator*(Smooth_periodic_function_gradient<T>& grad_f__, 
                                      Smooth_periodic_function_gradient<T>& grad_g__)

{
    assert(&grad_f__.fft() == &grad_g__.fft());
    assert(&grad_f__.gvec() == &grad_g__.gvec());
    
    Smooth_periodic_function<T> result(grad_f__.fft(), grad_f__.gvec());

    #pragma omp parallel for
    for (int ir = 0; ir < grad_f__.fft().local_size(); ir++) {
        double d{0};
        for (int x: {0, 1, 2}) {
            d += grad_f__[x].f_rg(ir) * grad_g__[x].f_rg(ir);
        }
        result.f_rg(ir) = d;
    }

    return std::move(result);
}

} // namespace sirius

#endif // __SMOOTH_PERIODIC_FUNCTION_H__
