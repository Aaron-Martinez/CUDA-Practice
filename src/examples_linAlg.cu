#include <random>
#include "examples_linAlg.cuh"

void addComparison(int blockSize) {
    auto startTotal = std::chrono::high_resolution_clock::now();
    int N = 1<<29; // 1M elements
  
    float *x = new float[N];
    float *y = new float[N];
    float *sum = new float[N];
    float *sum_sequential = new float[N];
    int size = N * sizeof(float);
  
    // device var copies
    float *d_x = new float[N];
    float *d_y = new float[N];
    float *d_sum = new float[N];
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_sum, size);
    
    // initialize x and y arrays on the host
    auto startInitialize = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }
    auto endInitialize = std::chrono::high_resolution_clock::now();

    auto startSequential = std::chrono::high_resolution_clock::now();
    add_vec_cpu(N, x, y, sum_sequential);
    auto endSequential = std::chrono::high_resolution_clock::now();

    auto startParallel1 = std::chrono::high_resolution_clock::now();
  
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, sum, size, cudaMemcpyHostToDevice);
  
    // invoke kernel
    //int threadsPerBlock = 256;
    int threadsPerBlock = blockSize;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_vec_gpu<<<blocksPerGrid, threadsPerBlock>>>(N, d_x, d_y, d_sum);
  
    // copy vector from device to host mem
    cudaMemcpy(sum, d_sum, size, cudaMemcpyDeviceToHost);
  
    auto endParallel1 = std::chrono::high_resolution_clock::now();

    // Free memory
    delete [] x;
    delete [] y;
    delete [] sum;
    delete [] sum_sequential;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_sum);
    
    auto endTotal = std::chrono::high_resolution_clock::now();

    printf("\nN = %i \n", N);
    printMilliseconds(startInitialize, endInitialize, "Time to initialize vectors");
    printMilliseconds(startSequential, endSequential, "CPU single thread time");
    printMilliseconds(startParallel1, endParallel1, "GPU time, block size " + std::to_string(blockSize));
    printMilliseconds(startTotal, endTotal, "Total time");
  
}


void matrixMultiplyComparison() {
    // multiply M x N  matrix by N x P  matrix
    int M = 1840;
    int N = 1000;
    int P = 2250;
    float *A = new float[M * N];
    float *B = new float[N * P];
    float *C = new float[M * P];
    float *C_gpu = new float[M * P];

    // copies for device
    int sizeA = M * N * sizeof(float);
    int sizeB = N * P * sizeof(float);
    int sizeC = M * P * sizeof(float);
    float *d_A = new float[M * N];
    float *d_B = new float[N * P];
    float *d_C = new float[M * P];
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // initialize matrices
    auto startInitialize = std::chrono::high_resolution_clock::now();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 5.0);
    for(int n = 0; n < (M*N); ++n) {
        A[n] = dis(gen);
    }
    for(int n = 0; n < (N*P); ++n) {
        B[n] = dis(gen);
    }
    auto endInitialize = std::chrono::high_resolution_clock::now();

    auto startSequential = std::chrono::high_resolution_clock::now();
    matrix_multiply_cpu(A, B, C, N, M, P);
    auto endSequential = std::chrono::high_resolution_clock::now();

    auto startParallel1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);
  
    // invoke kernel
    dim3 threadsPerBlock(16, 16);
    int blocksPerGrid = ((M*P) + threadsPerBlock.x * threadsPerBlock.y - 1) / threadsPerBlock.x / threadsPerBlock.y;
    matrix_multiply_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, P);
  
    // copy vector from device to host mem
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
  
    auto endParallel1 = std::chrono::high_resolution_clock::now();

    // Free memory
    delete [] A;
    delete [] B;
    delete [] C;
    delete [] C_gpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nN = %i \n", N);
    printMilliseconds(startInitialize, endInitialize, "Time to initialize vectors");
    printMilliseconds(startSequential, endSequential, "CPU single thread time");
    printMilliseconds(startParallel1, endParallel1, "GPU time");

}
