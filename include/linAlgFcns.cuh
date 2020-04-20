
#ifndef LINALGFCNS_CUH
#define LINALGFCNS_CUH

#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cusolverDn.h>
#include <math.h>

__global__ void add_vec_gpu(long n, float*A, float *B, float *C);
void add_vec_cpu(long n, float *A, float *B, float *C);
__global__ void matrix_multiply_gpu(float *A, float *B, float *C, int N, int M, int P);
void matrix_multiply_cpu(float *A, float *B, float *C, int N, int M, int P);

#endif