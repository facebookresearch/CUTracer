// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFECALL(call)                                                                                  \
  {                                                                                                          \
    call;                                                                                                    \
    cudaError err = cudaGetLastError();                                                                      \
    if (cudaSuccess != err) {                                                                                \
      fprintf(stderr, "Cuda error in function '%s' file '%s' in line %i : %s.\n", #call, __FILE__, __LINE__, \
              cudaGetErrorString(err));                                                                      \
      fflush(stderr);                                                                                        \
      exit(EXIT_FAILURE);                                                                                    \
    }                                                                                                        \
  }

#define BLOCK_SIZE 32

// CUDA kernel using shared memory for vector addition
// This kernel demonstrates shared memory usage for CUTracer testing
__global__ void vecAddSmem(double* a, double* b, double* c, int n) {
  __shared__ double s_a[BLOCK_SIZE];
  __shared__ double s_b[BLOCK_SIZE];

  auto tid = threadIdx.x;
  auto gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    // Step 1: Load from global memory to shared memory
    s_a[tid] = a[gid];
    s_b[tid] = b[gid];

    __syncthreads();

    // Step 2: Read from shared memory and compute
    double sum = s_a[tid] + s_b[tid];

    // Step 3: Write result back to global memory
    c[gid] = sum;
  }
}

int main(int argc, char* argv[]) {
  int n = 32;
  if (argc > 1) n = atoi(argv[1]);

  double *h_a, *h_b, *h_c;
  double *d_a, *d_b, *d_c;
  size_t bytes = n * sizeof(double);

  // Allocate host memory
  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);

  // Allocate device memory
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Initialize host data
  for (int i = 0; i < n; i++) {
    h_a[i] = sin(i) * sin(i);
    h_b[i] = cos(i) * cos(i);
  }

  // Copy to device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // Launch kernel
  int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CUDA_SAFECALL((vecAddSmem<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_c, n)));

  // Copy result back
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // Verify result
  double sum = 0;
  for (int i = 0; i < n; i++) sum += h_c[i];
  printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
