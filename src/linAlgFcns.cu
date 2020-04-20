#include "linAlgFcns.cuh"

__global__ void add_vec_gpu(long n, float*A, float *B, float *C) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) 
        C[i] = A[i] + B[i];
}

void add_vec_cpu(long n, float *A, float *B, float *C) {
    for(long i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// multiply M x N matrix by a N x P matrix
__global__ void matrix_multiply_gpu(float *A, float *B, float *C, int N, int M, int P) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0;
    if(col < P && row < M) {
        for(int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[col + N * i];
        }
        C[row * N + col] = sum;
    }
}

// multiply M x N matrix by a N x P matrix
void matrix_multiply_cpu(float *A, float *B, float *C, int N, int M, int P) {
    
    for(int row = 0; row < M; ++row) {
        for(int col = 0; col < P; ++col) {
            float sum = 0;
            for(int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[col + N * i];
            }
            C[row * N + col] = sum;
        }
    }
}