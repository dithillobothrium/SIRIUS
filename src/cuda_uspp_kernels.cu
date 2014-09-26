#include <cuda.h>
#include <cublas_v2.h>
#include "cuda_interface.h"

const double twopi = 6.2831853071795864769;

//== __global__ void create_beta_pw_gpu_kernel(int num_gkvec, 
//==                                           int* beta_t_idx, 
//==                                           cuDoubleComplex* beta_pw_type, 
//==                                           double* gkvec, 
//==                                           double* atom_pos,
//==                                           cuDoubleComplex* beta_pw)
//== {
//==     int i = blockIdx.y;
//==     int ia = beta_t_idx[array2D_offset(0, i, 2)];
//==     int offset_t = beta_t_idx[array2D_offset(1, i, 2)];
//== 
//==     int igk = blockDim.x * blockIdx.x + threadIdx.x;
//==     
//==     if (igk < num_gkvec)
//==     {
//==         double p = 0;
//==         for (int x = 0; x < 3; x++) p += atom_pos[array2D_offset(x, ia, 3)] * gkvec[array2D_offset(x, igk, 3)];
//==         p *= twopi;
//==         
//==         double sinp = sin(p);
//==         double cosp = cos(p);
//== 
//==         beta_pw[array2D_offset(igk, i, num_gkvec)] = 
//==             cuCmul(beta_pw_type[array2D_offset(igk, offset_t, num_gkvec)], make_cuDoubleComplex(cosp, -sinp));
//==     }
//== }
//== 
//== extern "C" void create_beta_pw_gpu(int num_gkvec, 
//==                                    int num_beta_atot, 
//==                                    int* beta_t_idx,
//==                                    void* beta_pw_type,
//==                                    double* gkvec,
//==                                    double* atom_pos,
//==                                    void* beta_pw)
//== {
//==     dim3 threadsPerBlock(64);
//==     dim3 numBlocks(num_blocks(num_gkvec, 64), num_beta_atot);
//== 
//==     create_beta_pw_gpu_kernel<<<
//==         numBlocks, 
//==         threadsPerBlock>>>(num_gkvec, 
//==                            beta_t_idx, 
//==                            (cuDoubleComplex*)beta_pw_type,
//==                            gkvec,
//==                            atom_pos,
//==                            (cuDoubleComplex*)beta_pw);
//== }

extern cudaStream_t* streams;

__global__ void create_beta_pw_gpu_kernel_v2
(
    int num_gkvec, 
    int* beta_pw_desc,
    cuDoubleComplex* beta_pw_type, 
    double* gkvec, 
    double* atom_pos,
    cuDoubleComplex* beta_pw
)
{
    int igk = blockDim.x * blockIdx.x + threadIdx.x;
    int ia = blockIdx.y;

    int nbf = beta_pw_desc[array2D_offset(0, ia, 3)];
    int offset_beta_pw = beta_pw_desc[array2D_offset(1, ia, 3)];
    int offset_beta_pw_t = beta_pw_desc[array2D_offset(2, ia, 3)];

    if (igk < num_gkvec)
    {
        double p = 0;
        for (int x = 0; x < 3; x++) p += atom_pos[array2D_offset(x, ia, 3)] * gkvec[array2D_offset(x, igk, 3)];
        p *= twopi;

        double sinp = sin(p);
        double cosp = cos(p);

        for (int xi = 0; xi < nbf; xi++)
        {
            beta_pw[array2D_offset(igk, offset_beta_pw + xi, num_gkvec)] =
                cuCmul(beta_pw_type[array2D_offset(igk, offset_beta_pw_t + xi, num_gkvec)], make_cuDoubleComplex(cosp, -sinp));
        }
    }
}

extern "C" void create_beta_pw_gpu_v2(int num_atoms,
                                      int num_gkvec,
                                      int* beta_pw_desc,
                                      cuDoubleComplex* beta_pw_type,
                                      double* gkvec,
                                      double* atom_pos,
                                      cuDoubleComplex* beta_pw)
{
    CUDA_timer t("create_beta_pw_gpu_v2");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec, grid_t.x), num_atoms);

    create_beta_pw_gpu_kernel_v2 <<<grid_b, grid_t>>> 
    (
        num_gkvec,
        beta_pw_desc,
        beta_pw_type,
        gkvec,
        atom_pos,
        beta_pw
    );
}

//== #define BLOCK_SIZE 32
//== 
//== __global__ void generate_beta_phi_gpu_kernel(int num_gkvec, 
//==                                              int num_beta,
//==                                              int num_phi,
//==                                              int* beta_t_idx, 
//==                                              double* atom_pos, 
//==                                              double* gkvec, 
//==                                              cuDoubleComplex* beta_pw_type,
//==                                              cuDoubleComplex* phi,
//==                                              cuDoubleComplex* beta_phi)
//== {
//==     int idx_beta = blockDim.x * blockIdx.x + threadIdx.x;
//==     int idx_phi = blockDim.y * blockIdx.y + threadIdx.y;
//==     int ia, offset_t;
//==     double x0, y0, z0;
//== 
//==     if (idx_beta < num_beta)
//==     {
//==         ia = beta_t_idx[array2D_offset(0, idx_beta, 2)];
//==         offset_t = beta_t_idx[array2D_offset(1, idx_beta, 2)];
//==         x0 = atom_pos[array2D_offset(0, ia, 3)];
//==         y0 = atom_pos[array2D_offset(1, ia, 3)];
//==         z0 = atom_pos[array2D_offset(2, ia, 3)];
//==     }
//== 
//==     int N = num_blocks(num_gkvec, BLOCK_SIZE);
//== 
//==     cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);
//== 
//==     for (int m = 0; m < N; m++)
//==     {
//==         __shared__ cuDoubleComplex beta_pw_tile[BLOCK_SIZE][BLOCK_SIZE];
//==         __shared__ cuDoubleComplex phi_tile[BLOCK_SIZE][BLOCK_SIZE];
//== 
//==         int bs = (m + 1) * BLOCK_SIZE > num_gkvec ? num_gkvec - m * BLOCK_SIZE : BLOCK_SIZE;
//== 
//==         int igk = m * BLOCK_SIZE + threadIdx.y;
//== 
//==         if (igk < num_gkvec && idx_beta < num_beta)
//==         {
//==             double x1 = gkvec[array2D_offset(igk, 0, num_gkvec)];
//==             double y1 = gkvec[array2D_offset(igk, 1, num_gkvec)];
//==             double z1 = gkvec[array2D_offset(igk, 2, num_gkvec)];
//== 
//==             double p = twopi * (x0 * x1 + y0 * y1 + z0 * z1);
//==             double sinp = sin(p);
//==             double cosp = cos(p);
//== 
//==             beta_pw_tile[threadIdx.x][threadIdx.y] = cuCmul(cuConj(beta_pw_type[array2D_offset(igk, offset_t, num_gkvec)]), 
//==                                                             make_cuDoubleComplex(cosp, sinp));
//== 
//==         }
//==         
//==         igk = m * BLOCK_SIZE + threadIdx.x;
//== 
//==         if (igk < num_gkvec && idx_phi < num_phi)
//==             phi_tile[threadIdx.y][threadIdx.x] = phi[array2D_offset(igk, idx_phi, num_gkvec)];
//== 
//==         __syncthreads();
//== 
//==         for (int i = 0; i < bs; i++) val = cuCadd(val, cuCmul(beta_pw_tile[threadIdx.x][i], phi_tile[threadIdx.y][i]));
//== 
//==         __syncthreads();
//==     }
//== 
//==     if (idx_beta < num_beta && idx_phi < num_phi) beta_phi[array2D_offset(idx_beta, idx_phi, num_beta)] = val;
//== }
//== 
//== 
//== extern "C" void generate_beta_phi_gpu(int num_gkvec, 
//==                                       int num_beta, 
//==                                       int num_phi, 
//==                                       int* beta_t_idx, 
//==                                       double* atom_pos,
//==                                       double* gkvec,
//==                                       void* beta_pw_type,
//==                                       void* phi,
//==                                       void* beta_phi)
//== {
//== 
//==     dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
//==     dim3 numBlocks(num_blocks(num_beta, BLOCK_SIZE), num_blocks(num_phi, BLOCK_SIZE));
//== 
//==     generate_beta_phi_gpu_kernel<<<
//==         numBlocks, 
//==         threadsPerBlock>>>(num_gkvec, 
//==                            num_beta,
//==                            num_phi,
//==                            beta_t_idx, 
//==                            atom_pos,
//==                            gkvec, 
//==                            (cuDoubleComplex*)beta_pw_type,
//==                            (cuDoubleComplex*)phi,
//==                            (cuDoubleComplex*)beta_phi);
//== }

//== __global__ void restore_valence_density_gpu_kernel(int num_gvec_loc,
//==                                                    int* atom_type,
//==                                                    int* num_beta, 
//==                                                    double* atom_pos,
//==                                                    int* gvec,
//==                                                    cuDoubleComplex* pp_complex_density_matrix,
//==                                                    int ldm,
//==                                                    cuDoubleComplex** q_pw,
//==                                                    cuDoubleComplex* f_pw)
//== {
//==     extern __shared__ char sdata_ptr[];
//==     cuDoubleComplex* sdata = (cuDoubleComplex*)&sdata_ptr[0];
//== 
//==     int ia = blockIdx.x;
//== 
//==     int iat = atom_type[ia];
//== 
//==     int nbf = num_beta[iat];
//== 
//==     cuDoubleComplex* q_pw_t = q_pw[iat];
//==     //printf("ia : %i, type : %i, nbf : %i, q_pw : %p", ia, iat, nbf, q_pw_t);
//== 
//==     double ax = atom_pos[array2D_offset(0, ia, 3)];
//==     double ay = atom_pos[array2D_offset(1, ia, 3)];
//==     double az = atom_pos[array2D_offset(2, ia, 3)];
//== 
//==     if (threadIdx.x == 0)
//==     {
//==         for (int xi2 = 0; xi2 < nbf; xi2++)
//==         {
//==             for (int xi1 = 0; xi1 <= xi2; xi1++)
//==             {
//==                 int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
//==                 sdata[idx12] = pp_complex_density_matrix[array4D_offset(xi2, xi1, 0, ia, ldm, ldm, 1)];
//==             }
//==         }
//==     }
//==     __syncthreads();
//== 
//==     cuDoubleComplex* f_pw_a = &f_pw[array2D_offset(0, ia, num_gvec_loc)];
//==     
//==     int N = num_blocks(num_gvec_loc, blockDim.x);
//== 
//==     for (int n = 0; n < N; n++)
//==     {
//==         int igloc = n * blockDim.x + threadIdx.x;
//==         if (igloc < num_gvec_loc)
//==         {
//==             int gvx = gvec[array2D_offset(0, igloc, 3)];
//==             int gvy = gvec[array2D_offset(1, igloc, 3)];
//==             int gvz = gvec[array2D_offset(2, igloc, 3)];
//== 
//==             double p = twopi * (ax * gvx + ay * gvy + az * gvz);
//==             
//==             double sinp = sin(p);
//==             double cosp = cos(p);
//== 
//==             cuDoubleComplex zval = make_cuDoubleComplex(0.0, 0.0);
//== 
//==             // \sum_{xi1, xi2} D_{xi2,xi1} * Q(G)_{xi1, xi2}
//==             for (int xi2 = 0; xi2 < nbf; xi2++)
//==             {
//==                 int idx12 = xi2 * (xi2 + 1) / 2;
//== 
//==                 //cuDoubleComplex q = cuCmul(make_cuDoubleComplex(cosp, -sinp), q_pw_t[array2D_offset(igloc, idx12 + xi2, num_gvec_loc)]);
//== 
//==                 // add diagonal term
//==                 //f_pw_a[igloc] = cuCadd(f_pw_a[igloc], cuCmul(sdata[idx12 + xi2], q));
//==                 zval = cuCadd(zval, cuCmul(sdata[idx12 + xi2], q_pw_t[array2D_offset(igloc, idx12 + xi2, num_gvec_loc)]));
//== 
//==                 // add non-diagonal terms
//==                 for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
//==                 {
//==                     cuDoubleComplex q = q_pw_t[array2D_offset(igloc, idx12, num_gvec_loc)];
//==                     //q = cuCmul(make_cuDoubleComplex(cosp, -sinp), q_pw_t[array2D_offset(igloc, idx12, num_gvec_loc)]);
//==                     
//==                     //double d = 2 * cuCreal(cuCmul(sdata[idx12], q));
//== 
//==                     //f_pw_a[igloc] = cuCadd(f_pw_a[igloc], make_cuDoubleComplex(d, 0));
//==                     //double d = 2 * cuCreal(cuCmul(sdata[idx12], q_pw_t[array2D_offset(igloc, idx12, num_gvec_loc)])
//==                     zval.x += 2 * (sdata[idx12].x * q.x - sdata[idx12].y * q.y);
//==                     //zval = cuCadd(zval, make_cuDoubleComplex(2 * cuCreal(cuCmul(sdata[idx12], q_pw_t[array2D_offset(igloc, idx12, num_gvec_loc)])), 0.0));
//==                 }
//==             }
//==             f_pw_a[igloc] = cuCadd(f_pw_a[igloc], cuCmul(zval, make_cuDoubleComplex(cosp, -sinp))); 
//==         }
//==     }
//== }
//== 
//== __global__ void reduce_rho_pw_kernel(int num_atoms, int num_gvec_loc, cuDoubleComplex* f_pw, cuDoubleComplex* rho_pw)
//== {
//==     int igloc = blockDim.x * blockIdx.x + threadIdx.x;
//== 
//==     if (igloc < num_gvec_loc)
//==     {
//==         for (int ia = 0; ia < num_atoms; ia++) 
//==             rho_pw[igloc] = cuCadd(rho_pw[igloc], f_pw[array2D_offset(igloc, ia, num_gvec_loc)]);
//==     }
//== }
//== 
//== 
//== extern "C" void restore_valence_density_gpu(int num_atoms, 
//==                                             int num_gvec_loc,
//==                                             int* atom_type,
//==                                             int* num_beta, 
//==                                             double* atom_pos, 
//==                                             int* gvec,
//==                                             void* pp_complex_density_matrix,
//==                                             int ldm,
//==                                             void** q_pw,
//==                                             void* rho_pw)
//== {
//==     dim3 threadsPerBlock(1024);
//==     dim3 numBlocks(num_atoms);
//== 
//==     cuDoubleComplex* f_pw;
//==     f_pw = (cuDoubleComplex*)cuda_malloc(num_gvec_loc * num_atoms * sizeof(cuDoubleComplex));
//==     cuda_memset(f_pw, 0, num_gvec_loc * num_atoms * sizeof(cuDoubleComplex));
//== 
//==     restore_valence_density_gpu_kernel<<<
//==         numBlocks,
//==         threadsPerBlock,
//==         sizeof(cuDoubleComplex) * ldm * (ldm + 1) / 2>>>(num_gvec_loc,
//==                                                          atom_type,
//==                                                          num_beta, 
//==                                                          atom_pos, 
//==                                                          gvec, 
//==                                                          (cuDoubleComplex*)pp_complex_density_matrix,
//==                                                          ldm,
//==                                                          (cuDoubleComplex**)q_pw,
//==                                                          f_pw);
//==     
//==     cuda_memset(rho_pw, 0, num_gvec_loc * sizeof(cuDoubleComplex));
//==     
//==     dim3 grid_t(128);
//==     dim3 grid_b(num_blocks(num_gvec_loc, grid_t.x));
//==     reduce_rho_pw_kernel<<<grid_b, grid_t>>>
//==         (num_atoms, num_gvec_loc, f_pw, (cuDoubleComplex*)rho_pw);
//==     
//==     cuda_device_synchronize();
//==     cuda_free(f_pw);
//== }




//== __global__ void restore_valence_density_gpu_kernel_v2
//== (
//==     int num_gvec_loc,
//==     int num_beta, 
//==     double ax,
//==     double ay,
//==     double az,
//==     int* gvec,
//==     cuDoubleComplex* pp_complex_density_matrix,
//==     int ldm,
//==     cuDoubleComplex* q_pw_t,
//==     cuDoubleComplex* rho_pw
//== )
//== {
//==     extern __shared__ char sdata_ptr[];
//==     cuDoubleComplex* sdata = (cuDoubleComplex*)&sdata_ptr[0];
//== 
//==     if (threadIdx.x == 0)
//==     {
//==         for (int xi2 = 0; xi2 < num_beta; xi2++)
//==         {
//==             for (int xi1 = 0; xi1 <= xi2; xi1++)
//==             {
//==                 int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
//==                 sdata[idx12] = pp_complex_density_matrix[array3D_offset(xi2, xi1, 0, ldm, ldm)];
//==             }
//==         }
//==     }
//==     __syncthreads();
//== 
//==     int igloc = blockIdx.x * blockDim.x + threadIdx.x;
//==     if (igloc < num_gvec_loc)
//==     {
//==         int gvx = gvec[array2D_offset(0, igloc, 3)];
//==         int gvy = gvec[array2D_offset(1, igloc, 3)];
//==         int gvz = gvec[array2D_offset(2, igloc, 3)];
//== 
//==         double p = twopi * (ax * gvx + ay * gvy + az * gvz);
//==         
//==         double sinp = sin(p);
//==         double cosp = cos(p);
//== 
//==         cuDoubleComplex zval = make_cuDoubleComplex(0.0, 0.0);
//== 
//==         // \sum_{xi1, xi2} D_{xi2,xi1} * Q(G)_{xi1, xi2}
//==         for (int xi2 = 0; xi2 < num_beta; xi2++)
//==         {
//==             int idx12 = xi2 * (xi2 + 1) / 2;
//== 
//==             // add diagonal term
//==             zval = cuCadd(zval, cuCmul(sdata[idx12 + xi2], q_pw_t[array2D_offset(igloc, idx12 + xi2, num_gvec_loc)]));
//== 
//==             // add non-diagonal terms
//==             for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
//==             {
//==                 cuDoubleComplex q = q_pw_t[array2D_offset(igloc, idx12, num_gvec_loc)];
//==                 zval.x += 2 * (sdata[idx12].x * q.x - sdata[idx12].y * q.y);
//==             }
//==         }
//==         rho_pw[igloc] = cuCadd(rho_pw[igloc], cuCmul(zval, make_cuDoubleComplex(cosp, -sinp))); 
//==     }
//== }
//== 
//== extern "C" void restore_valence_density_gpu_v2(int num_gvec_loc,
//==                                                int num_beta,
//==                                                double ax,
//==                                                double ay,
//==                                                double az,
//==                                                int* gvec,
//==                                                void* pp_complex_density_matrix,
//==                                                int ldm,
//==                                                void* q_pw_t,
//==                                                void* rho_pw,
//==                                                int stream_id)
//== {
//==     cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];
//== 
//==     dim3 grid_t(64);
//==     dim3 grid_b(num_blocks(num_gvec_loc, grid_t.x));
//== 
//==     restore_valence_density_gpu_kernel_v2<<<grid_b, grid_t, sizeof(cuDoubleComplex) * ldm * (ldm + 1) / 2, stream>>>
//==         (num_gvec_loc, num_beta, ax, ay, az, gvec, (cuDoubleComplex*)pp_complex_density_matrix, ldm,
//==          (cuDoubleComplex*)q_pw_t, (cuDoubleComplex*)rho_pw);
//== }

__global__ void mul_veff_with_phase_factors_kernel(int num_gvec_loc,
                                                   cuDoubleComplex* veff, 
                                                   int* gvec, 
                                                   double ax, 
                                                   double ay, 
                                                   double az, 
                                                   cuDoubleComplex* vtmp)
{
    int igloc = blockDim.x * blockIdx.x + threadIdx.x;
    if (igloc < num_gvec_loc)
    {
        int gvx = gvec[array2D_offset(0, igloc, 3)];
        int gvy = gvec[array2D_offset(1, igloc, 3)];
        int gvz = gvec[array2D_offset(2, igloc, 3)];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);
            
        vtmp[igloc] = cuCmul(veff[igloc], make_cuDoubleComplex(cos(p), sin(p)));
    }
}
 
extern "C" void mul_veff_with_phase_factors(int num_gvec_loc, 
                                            void* veff, 
                                            int* gvec, 
                                            double ax,
                                            double ay,
                                            double az,
                                            void* vtmp)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc, grid_t.x));

    mul_veff_with_phase_factors_kernel<<<grid_b, grid_t>>>
        (num_gvec_loc, (cuDoubleComplex*)veff, gvec, ax, ay, az, (cuDoubleComplex*)vtmp);
}

__global__ void compute_d_mtrx_gpu_kernel(int num_gvec_loc, 
                                          cuDoubleComplex* vtmp, 
                                          cuDoubleComplex* q_pw, 
                                          cuDoubleComplex* d_mtrx_gpu)
{
    int idx = blockIdx.x;

    int N = num_blocks(num_gvec_loc, blockDim.x);

    extern __shared__ char sdata_ptr[];
    cuDoubleComplex* sdata = (cuDoubleComplex*)&sdata_ptr[0];

    sdata[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);

    for (int n = 0; n < N; n++)
    {
        int igloc = n * blockDim.x + threadIdx.x;
        if (igloc < num_gvec_loc)
        {
            sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], 
                                        cuCmul(vtmp[igloc], 
                                               cuConj(q_pw[array2D_offset(igloc, idx,  num_gvec_loc)])));
        }
    }
    
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    d_mtrx_gpu[idx] = sdata[0];
}

extern "C" void compute_d_mtrx_valence_gpu(int num_gvec_loc,
                                           int num_elements,
                                           void* veff, 
                                           int* gvec, 
                                           double ax,
                                           double ay,
                                           double az,
                                           void* vtmp,
                                           void* q_pw_t,
                                           void* d_mtrx,
                                           int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    dim3 grid_t(64);

    dim3 grid_b(num_blocks(num_gvec_loc, grid_t.x));
    mul_veff_with_phase_factors_kernel<<<grid_b, grid_t, 0, stream>>>
        (num_gvec_loc, (cuDoubleComplex*)veff, gvec, ax, ay, az, (cuDoubleComplex*)vtmp);

    grid_b = dim3(num_elements);
    compute_d_mtrx_gpu_kernel<<<grid_b, grid_t, grid_t.x * sizeof(cuDoubleComplex), stream>>>
        (num_gvec_loc, (cuDoubleComplex*)vtmp, (cuDoubleComplex*)q_pw_t, (cuDoubleComplex*)d_mtrx);
}

__global__ void generate_phase_factors_gpu_kernel(int num_gvec_loc, 
                                                  int num_atoms, 
                                                  double* atom_pos, 
                                                  int* gvec, 
                                                  cuDoubleComplex* phase_factors)
{
    int ia = blockIdx.y;
    int igloc = blockIdx.x * blockDim.x + threadIdx.x;

    if (igloc < num_gvec_loc)
    {
        int gvx = gvec[array2D_offset(0, igloc, 3)];
        int gvy = gvec[array2D_offset(1, igloc, 3)];
        int gvz = gvec[array2D_offset(2, igloc, 3)];
    
        double ax = atom_pos[array2D_offset(ia, 0, num_atoms)];
        double ay = atom_pos[array2D_offset(ia, 1, num_atoms)];
        double az = atom_pos[array2D_offset(ia, 2, num_atoms)];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);

        double sinp = sin(p);
        double cosp = cos(p);

        phase_factors[array2D_offset(igloc, ia, num_gvec_loc)] = make_cuDoubleComplex(cosp, -sinp);
    }
}


extern "C" void generate_d_mtrx_pw_gpu(int num_atoms,
                                       int num_gvec_loc,
                                       int num_beta,
                                       double* atom_pos,
                                       int* gvec,
                                       cuDoubleComplex* d_mtrx_packed,
                                       cuDoubleComplex* d_mtrx_pw)
{
    CUDA_timer t("generate_d_mtrx_pw_gpu");

    cuDoubleComplex* phase_factors;
    phase_factors = (cuDoubleComplex*)cuda_malloc(num_gvec_loc * num_atoms * sizeof (cuDoubleComplex));

    dim3 grid_t(32);
    dim3 grid_b(num_blocks(num_gvec_loc, grid_t.x), num_atoms);

    generate_phase_factors_gpu_kernel<<<grid_b, grid_t>>>(num_gvec_loc, 
                                                          num_atoms, 
                                                          atom_pos, 
                                                          gvec, 
                                                          phase_factors);
    
    cuDoubleComplex zone = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zzero = make_cuDoubleComplex(0.0, 0.0);

    cublas_zgemm(0, 0, num_gvec_loc, num_beta * (num_beta + 1) / 2, num_atoms, &zone, 
                 phase_factors, num_gvec_loc, d_mtrx_packed, num_atoms, &zzero,
                 d_mtrx_pw, num_gvec_loc, -1);

    cuda_free(phase_factors);
}

__global__ void sum_q_pw_d_mtrx_pw_gpu_kernel
(
    int num_gvec_loc,
    int num_beta,
    cuDoubleComplex* q_pw_t,
    cuDoubleComplex* d_mtrx_pw,
    cuDoubleComplex* rho_pw
)
{
    int igloc = blockIdx.x * blockDim.x + threadIdx.x;
    if (igloc < num_gvec_loc)
    {
        cuDoubleComplex zval = make_cuDoubleComplex(0.0, 0.0);

        // \sum_{xi1, xi2} D_{xi2,xi1} * Q(G)_{xi1, xi2}
        for (int xi2 = 0; xi2 < num_beta; xi2++)
        {
            int idx12 = xi2 * (xi2 + 1) / 2;

            // add diagonal term
            zval = cuCadd(zval, cuCmul(d_mtrx_pw[array2D_offset(igloc, idx12 + xi2, num_gvec_loc)], 
                                       q_pw_t[array2D_offset(igloc, idx12 + xi2, num_gvec_loc)]));

            // add non-diagonal terms
            for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
            {
                cuDoubleComplex q = q_pw_t[array2D_offset(igloc, idx12, num_gvec_loc)];
                cuDoubleComplex d = d_mtrx_pw[array2D_offset(igloc, idx12, num_gvec_loc)];
                zval.x += 2 * (d.x * q.x - d.y * q.y);
            }
        }
        rho_pw[igloc] = cuCadd(rho_pw[igloc], zval);
    }
}

extern "C" void sum_q_pw_d_mtrx_pw_gpu(int num_gvec_loc,
                                       int num_beta,
                                       cuDoubleComplex* q_pw_t,
                                       cuDoubleComplex* d_mtrx_pw,
                                       cuDoubleComplex* rho_pw)
{
    CUDA_timer t("sum_q_pw_d_mtrx_pw_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc, grid_t.x));
    
    sum_q_pw_d_mtrx_pw_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gvec_loc, 
        num_beta, 
        q_pw_t, 
        d_mtrx_pw, 
        rho_pw
    );
}

__global__ void copy_beta_psi_gpu_kernel
(
    cuDoubleComplex const* beta_psi,
    int beta_psi_ld, 
    double const* wo,
    cuDoubleComplex* beta_psi_wo,
    int beta_psi_wo_ld
)
{
    int xi = threadIdx.x;
    int j = blockIdx.x;

    beta_psi_wo[array2D_offset(xi, j, beta_psi_wo_ld)] = cuCmul(cuConj(beta_psi[array2D_offset(xi, j, beta_psi_ld)]),
                                                                make_cuDoubleComplex(wo[j], 0.0));
}

extern "C" void copy_beta_psi_gpu(int nbf,
                                  int nloc,
                                  cuDoubleComplex const* beta_psi,
                                  int beta_psi_ld,
                                  double const* wo,
                                  cuDoubleComplex* beta_psi_wo,
                                  int beta_psi_wo_ld,
                                  int stream_id)
{
    dim3 grid_t(nbf);
    dim3 grid_b(nloc);
    
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];
    
    copy_beta_psi_gpu_kernel <<<grid_b, grid_t, 0, stream>>>
    (
        beta_psi,
        beta_psi_ld,
        wo,
        beta_psi_wo,
        beta_psi_wo_ld
    );
}

__global__ void compute_residuals_gpu_kernel
(
    int const num_gkvec_row,
    int const* res_idx,
    double const* eval,
    cuDoubleComplex const* hpsi,
    cuDoubleComplex const* opsi,
    cuDoubleComplex* res
)
{
    int igk = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = res_idx[blockIdx.y];

    if (igk < num_gkvec_row)
    {
        int k = array2D_offset(igk, blockIdx.y, num_gkvec_row);
        /* res = hpsi_j - e_j * opsi_j */
        res[k] = cuCsub(hpsi[k], cuCmul(make_cuDoubleComplex(eval[ibnd], 0), opsi[k]));
    }
}

__global__ void compute_residuals_norm_gpu_kernel
(
    int num_gkvec_row,
    int* res_idx,
    cuDoubleComplex const* res,
    double* res_norm
)
{
    int N = num_blocks(num_gkvec_row, blockDim.x);

    extern __shared__ char sdata_ptr[];
    double* sdata = (double*)&sdata_ptr[0];

    sdata[threadIdx.x] = 0.0;

    for (int n = 0; n < N; n++)
    {
        int igk = n * blockDim.x + threadIdx.x;
        if (igk < num_gkvec_row)
        {
            int k = array2D_offset(igk, blockIdx.x, num_gkvec_row);
            sdata[threadIdx.x] += res[k].x * res[k].x + res[k].y * res[k].y;
        }
    }

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    res_norm[res_idx[blockIdx.x]] = sdata[0];
}

extern "C" void compute_residuals_gpu(int num_gkvec_row,
                                      int num_res_local,
                                      int* res_idx,
                                      double* eval,
                                      cuDoubleComplex const* hpsi,
                                      cuDoubleComplex const* opsi,
                                      cuDoubleComplex* res,
                                      double* res_norm)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec_row, grid_t.x), num_res_local);

    compute_residuals_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gkvec_row,
        res_idx,
        eval,
        hpsi,
        opsi,
        res
    );

    grid_b = dim3(num_res_local);
    compute_residuals_norm_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
    (
        num_gkvec_row,
        res_idx,
        res,
        res_norm
    );
}

__global__ void apply_preconditioner_gpu_kernel
(
    int const num_gkvec_row,
    int const* res_idx,
    double const* eval,
    cuDoubleComplex const* h_diag,
    cuDoubleComplex const* o_diag,
    cuDoubleComplex* res
)
{
    int igk = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = res_idx[blockIdx.y];

    if (igk < num_gkvec_row)
    {
        cuDoubleComplex z = cuCsub(h_diag[igk], cuCmul(make_cuDoubleComplex(eval[ibnd], 0.0), o_diag[igk]));
        int k = array2D_offset(igk, blockIdx.y, num_gkvec_row);
        res[k] = cuCdiv(res[k], z);
    }
}

extern "C" void apply_preconditioner_gpu(int num_gkvec_row,
                                         int num_res_local,
                                         int* res_idx,
                                         double* eval,
                                         cuDoubleComplex const* h_diag,
                                         cuDoubleComplex const* o_diag,
                                         cuDoubleComplex* res,
                                         double* res_norm)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec_row, grid_t.x), num_res_local);

    apply_preconditioner_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gkvec_row,
        res_idx,
        eval,
        h_diag,
        o_diag,
        res
    );

    grid_b = dim3(num_res_local);
    compute_residuals_norm_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
    (
        num_gkvec_row,
        res_idx,
        res,
        res_norm
    );
}

__global__ void normalize_residuals_gpu_kernel
(
    int const num_gkvec_row,
    int const* res_idx,
    double const* norm2,
    cuDoubleComplex* res
)
{
    int igk = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = res_idx[blockIdx.y];

    if (igk < num_gkvec_row)
    {
        int k = array2D_offset(igk, blockIdx.y, num_gkvec_row);
        res[k] = cuCdiv(res[k], make_cuDoubleComplex(sqrt(norm2[ibnd]), 0.0));
    }
}

extern "C" void normalize_residuals_gpu(int num_gkvec_row,
                                        int num_res_local,
                                        int* res_idx,
                                        double* norm2,
                                        cuDoubleComplex* res)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec_row, grid_t.x), num_res_local);

    normalize_residuals_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gkvec_row,
        res_idx,
        norm2,
        res
    );
}

