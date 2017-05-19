// This file must be compiled with nvcc

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <magma.h>
#include <magma_z.h>
#include <magma_d.h>
#include <magma_threadsetting.h>

extern "C" void magma_init_wrapper()
{
    magma_init();
}

extern "C" void magma_finalize_wrapper()
{
    magma_finalize();
}

extern "C" int magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, void* b, 
                                            int32_t ldb, double* eval)
{
    int m;
    int info;
    
    int lwork;
    int lrwork;
    int liwork;
    magma_zheevdx_getworksize(matrix_size, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

    magmaDoubleComplex* h_work;
    if (cudaMallocHost((void**)&h_work, lwork * sizeof(magmaDoubleComplex)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    double* rwork;
    if (cudaMallocHost((void**)&rwork, lrwork * sizeof(double)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }

    bool is_ok = true;
    magma_zhegvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size, (magmaDoubleComplex*)a, lda, (magmaDoubleComplex*)b, ldb, 0.0, 0.0, 
                         1, nv, &m, w, h_work, lwork, rwork, lrwork, iwork, liwork, &info);

    if (info) {
        //printf("magma_zhegvdx_2stage returned : %i\n", info);
        //if (info == MAGMA_ERR_DEVICE_ALLOC) {
        //    printf("this is MAGMA_ERR_DEVICE_ALLOC\n");
        //}
        is_ok = false;
    }    

    if (m < nv) {
        //printf("Not all eigen-vectors are found.\n");
        //printf("requested number of eigen-vectors: %i\n", nv);
        //printf("found number of eigen-vectors: %i\n", m);
        //exit(-1);
        is_ok = false;
    }

    if (is_ok) {
        memcpy(eval, &w[0], nv * sizeof(double));
    }
    
    cudaFreeHost(h_work);
    cudaFreeHost(rwork);
    free(iwork);
    free(w);

    if (is_ok) {
        return 0;
    }
    return 1;
}

extern "C" int magma_dsygvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, void* b, 
                                            int32_t ldb, double* eval)
{
    int m;
    int info;
    
    int lwork;
    int liwork;
    magma_dsyevdx_getworksize(matrix_size, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

    double* h_work;
    if (cudaMallocHost((void**)&h_work, lwork * sizeof(double)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }

    magma_dsygvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size, (double*)a, lda, (double*)b, ldb, 0.0, 0.0, 
                         1, nv, &m, w, h_work, lwork, iwork, liwork, &info);
    
    bool is_ok = true;
    if (info) {
        //printf("magma_dsygvdx_2stage : %i\n", info);
        //if (info == MAGMA_ERR_DEVICE_ALLOC)
        //    printf("this is MAGMA_ERR_DEVICE_ALLOC\n");
        //exit(-1);
        is_ok = false;
    }    

    if (m < nv) {
        //printf("Not all eigen-vectors are found.\n");
        //printf("requested number of eigen-vectors: %i\n", nv);
        //printf("found number of eigen-vectors: %i\n", m);
        //exit(-1);
        is_ok = false;
    }
    
    if (is_ok) {
        memcpy(eval, &w[0], nv * sizeof(double));
    }
    
    cudaFreeHost(h_work);
    free(iwork);
    free(w);

    if (is_ok) {
        return 0;
    }
    return 1;
}

extern "C" int magma_dsyevdx_wrapper(int32_t matrix_size, int32_t nv, double* a, int32_t lda, double* eval)
{
    int info, m;

    int lwork;
    int liwork;
    magma_dsyevdx_getworksize(matrix_size, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

    double* h_work;
    if (cudaMallocHost((void**)&h_work, lwork * sizeof(double)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }

    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }

    magma_dsyevdx(MagmaVec, MagmaRangeI, MagmaLower, matrix_size, a, lda, 0.0, 0.0, 1, nv, &m, &w[0],
                  h_work, lwork, iwork, liwork, &info);
    
    bool is_ok = true;
    if (info) {
        is_ok = false;
        //printf("magma_dsyevdx : %i\n", info);
        //if (info == MAGMA_ERR_DEVICE_ALLOC)
        //    printf("this is MAGMA_ERR_DEVICE_ALLOC\n");
        //exit(-1);
    }    

    if (m < nv) {
        is_ok = false;
        //printf("Not all eigen-vectors are found.\n");
        //printf("requested number of eigen-vectors: %i\n", nv);
        //printf("found number of eigen-vectors: %i\n", m);
        //exit(-1);
    }
    
    if (is_ok) {
        memcpy(eval, &w[0], nv * sizeof(double));
    }

    cudaFreeHost(h_work);
    free(iwork);
    free(w);

    if (is_ok) {
        return 0;
    }
    return 1;
}

extern "C" int magma_zheevdx_wrapper(int32_t matrix_size, int32_t nv, cuDoubleComplex* a, int32_t lda, double* eval)
{
    int info, m;

    int lwork;
    int lrwork;
    int liwork;
    magma_zheevdx_getworksize(matrix_size, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

    magmaDoubleComplex* h_work;
    if (cudaMallocHost((void**)&h_work, lwork * sizeof(magmaDoubleComplex)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    double* rwork;
    if (cudaMallocHost((void**)&rwork, lrwork * sizeof(double)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL) {
        printf("malloc failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }

    magma_zheevdx(MagmaVec, MagmaRangeI, MagmaLower, matrix_size, a, lda, 0.0, 0.0, 1, nv, &m, &w[0],
                  h_work, lwork, rwork, lrwork, iwork, liwork, &info);
    
    bool is_ok = true;
    if (info) {
        //printf("magma_zheevdx : error code = %i\n", info);
        //if (info == MAGMA_ERR_DEVICE_ALLOC) {
        //    printf("this is MAGMA_ERR_DEVICE_ALLOC\n");
        //}
        is_ok = false;
    }    

    if (m < nv) {
        //printf("magma_zheevdx: not all eigen-vectors are found\n");
        //printf("  matrix size:                       %i\n", matrix_size);
        //printf("  target number of eigen-vectors:    %i\n", nv);
        //printf("  number of eigen-vectors found:     %i\n", m);
        is_ok = false;
    }
    
    if (is_ok) {
        memcpy(eval, &w[0], nv * sizeof(double));
    }

    cudaFreeHost(h_work);
    cudaFreeHost(rwork);
    free(iwork);
    free(w);

    if (is_ok) {
        return 0;
    }
    return 1;
}

extern "C" int magma_zheevdx_2stage_wrapper(int32_t matrix_size, int32_t nv, cuDoubleComplex* a, int32_t lda, double* eval)
{
    int info, m;

    int lwork;
    int lrwork;
    int liwork;
    magma_zheevdx_getworksize(matrix_size, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

    magmaDoubleComplex* h_work;
    if (cudaMallocHost((void**)&h_work, lwork * sizeof(magmaDoubleComplex)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    double* rwork;
    if (cudaMallocHost((void**)&rwork, lrwork * sizeof(double)) != cudaSuccess) {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL) {
        printf("malloc failed\n");
        exit(-1);
    }
    
    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL) {
        printf("malloc failed\n");
        exit(-1);
    }

    magma_zheevdx_2stage(MagmaVec, MagmaRangeI, MagmaLower, matrix_size, a, lda, 0.0, 0.0, 1, nv, &m, &w[0],
                         h_work, lwork, rwork, lrwork, iwork, liwork, &info);
    
    bool is_ok = true;
    if (info) {
        //printf("magma_zheevdx_2stage: error code = %i\n", info);
        //if (info == MAGMA_ERR_DEVICE_ALLOC) {
        //    printf("this is MAGMA_ERR_DEVICE_ALLOC\n");
        //}
        is_ok = false;
    }

    if (m < nv) {
        //printf("magma_zheevdx_2stage: not all eigen-vectors are found\n");
        //printf("  matrix size:                       %i\n", matrix_size);
        //printf("  target number of eigen-vectors:    %i\n", nv);
        //printf("  number of eigen-vectors found:     %i\n", m);
        is_ok = false;
    }
    
    if (is_ok) {
        memcpy(eval, &w[0], nv * sizeof(double));
    }

    cudaFreeHost(h_work);
    cudaFreeHost(rwork);
    free(iwork);
    free(w);
    
    if (is_ok) {
        return 0;
    }
    return 1;
}

extern "C" int magma_dpotrf_wrapper(char uplo, int n, double* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L'))
    {
        printf("magma_dpotrf_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_dpotrf_gpu(magma_uplo, n, A, lda, &info);
    return info;
}

extern "C" int magma_zpotrf_wrapper(char uplo, int n, magmaDoubleComplex* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L'))
    {
        printf("magma_zpotrf_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_zpotrf_gpu(magma_uplo, n, A, lda, &info);
    return info;
}

extern "C" int magma_dtrtri_wrapper(char uplo, int n, double* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L'))
    {
        printf("magma_dtrtri_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_dtrtri_gpu(magma_uplo, MagmaNonUnit, n, A, lda, &info);
    return info;
}

extern "C" int magma_ztrtri_wrapper(char uplo, int n, magmaDoubleComplex* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L'))
    {
        printf("magma_ztrtri_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_ztrtri_gpu(magma_uplo, MagmaNonUnit, n, A, lda, &info);
    return info;
}



