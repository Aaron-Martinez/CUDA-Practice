#include "examples_linAlg.cuh"

void addComparison(int blockSize) {
    auto startTotal = std::chrono::high_resolution_clock::now();
    int N = 1<<30; // 1M elements
  
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
  
    auto startSequential = std::chrono::high_resolution_clock::now();
    add_vec_cpu(N, x, y, sum_sequential);
    auto endSequential = std::chrono::high_resolution_clock::now();
    
    // initialize x and y arrays on the host
    auto startInitialize = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }
    auto endInitialize = std::chrono::high_resolution_clock::now();
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
