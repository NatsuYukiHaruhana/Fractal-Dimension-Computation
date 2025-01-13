#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "FastDBC_GPU.cuh"

__global__ void initialDBCKernel(unsigned char* Imax, unsigned char* Imin, const int M, const unsigned char bits_M, const unsigned int sm, const float h, const unsigned char bits_s, const unsigned char bits_TPB, float* Nr) {
    register unsigned int tid = threadIdx.x;
    register unsigned int idx = (blockIdx.x << bits_TPB) + tid; // 2^bits_TPB = TPB
    register unsigned int i = idx >> (bits_M - bits_s); // i index: idx / (M/s)
    register unsigned int j = idx & ((M >> bits_s) - 1); // j index: idx mod (M/s)

    // identifies global location of position (0, 0) of the grid (i, j): ibsbM + jbs
    const register unsigned int ibsbM = (i << bits_s) << bits_M;
    const register unsigned int ibssmbM = ((i << bits_s) + sm) << bits_M;
    const register unsigned int jbs = j << bits_s;
    const register unsigned int jbssm = (j << bits_s) + sm;

    // computes and stores the maximum and minimum values of the grid (i, j)
    register unsigned char maxval = max(max(Imax[ibsbM + jbs], Imax[ibsbM + jbssm]), max(Imax[ibssmbM + jbs], Imax[ibssmbM + jbssm]));
    register unsigned char minval = min(min(Imax[ibsbM + jbs], Imax[ibsbM + jbssm]), min(Imax[ibssmbM + jbs], Imax[ibssmbM + jbssm]));
    Imax[ibsbM + jbs] = maxval;
    Imin[ibsbM + jbs] = minval;

    // computes the box-counting for grid (i, j) and adds it to Nr
    float invh = 1.0 / h;
    atomicAdd(Nr, ceilf(maxval * invh) - ceilf(minval * invh) + 1);
}

__global__ void DBCKernel(unsigned char* Imax, unsigned char* Imin, const int M, const unsigned char bits_M, const unsigned int sm, const float h, const unsigned char bits_s, const unsigned char bits_TPB, float* Nr) {
    register unsigned int tid = threadIdx.x;
    register unsigned int idx = (blockIdx.x << bits_TPB) + tid; // 2^bits_TPB = TPB
    register unsigned int i = idx >> (bits_M - bits_s); // i index: idx / (M/s)
    register unsigned int j = idx & ((M >> bits_s) - 1); // j index: idx mod (M/s)

    // identifies global location of position (0, 0) of the grid (i, j): ibsbM + jbs
    const register unsigned int ibsbM = (i << bits_s) << bits_M;
    const register unsigned int ibssmbM = ((i << bits_s) + sm) << bits_M;
    const register unsigned int jbs = j << bits_s;
    const register unsigned int jbssm = (j << bits_s) + sm;

    // computes and stores the maximum and minimum values of the grid (i, j)
    register unsigned char maxval = max(max(Imax[ibsbM + jbs], Imax[ibsbM + jbssm]), max(Imax[ibssmbM + jbs], Imax[ibssmbM + jbssm]));
    register unsigned char minval = min(min(Imin[ibsbM + jbs], Imin[ibsbM + jbssm]), min(Imin[ibssmbM + jbs], Imin[ibssmbM + jbssm]));
    Imax[ibsbM + jbs] = maxval;
    Imin[ibsbM + jbs] = minval;

    // computes the box-counting for grid (i, j) and adds it to Nr
    float invh = 1.0 / h;
    atomicAdd(Nr, ceilf(maxval * invh) - ceilf(minval * invh) + 1);
}

cudaError_t CudaDBC2D(unsigned char* I, const int M, const unsigned char G, float Nr[]) {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // allocating CPU pinned memory

    unsigned char* Imax = 0, * Imin = 0;
    cudaMallocHost((void**)&Imax, sizeof(unsigned char) * M * M);
    cudaMallocHost((void**)&Imin, sizeof(unsigned char) * M * M);

    memcpy(Imax, I, sizeof(unsigned char) * M * M);
    const int Numr = log2(M) - 1;

    //cudaMallocHost((void**)&Nr, sizeof(float) * Numr);

    // CPU-GPU data transfers (only transfer of Imax is needed)
    unsigned char* dev_Imax, * dev_Imin;
    float* dev_Nr;
    cudaStatus = cudaMalloc((void**)&dev_Imax, M * M * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_Imax, Imax, M * M * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_Imax);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_Imin, M * M * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_Imax);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_Nr, Numr * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_Imax);
        cudaFree(dev_Imin);
        return cudaStatus;
    }

    cudaStatus = cudaMemset(dev_Nr, 0, Numr * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        cudaFree(dev_Imax);
        cudaFree(dev_Imin);
        cudaFree(dev_Nr);
        return cudaStatus;
    }

    // initialDBCKernel call. Computes box-counting for 2x2 grids and initializes Imin
    const unsigned char bits_M = Numr + 1; // 2^bits_M = M
    unsigned int s = 2;
    float h = (G * s) / M;
    h = h == 0.0 ? 0.001 : h;

    // SELECT THE SIZE OF TPB (threads per block)
    unsigned int TPB = 128;
    dim3 grid, block(TPB, 1); // variables for kernel launching

    unsigned long long int num_box = (M * M) / (s * s);
    unsigned char b_TPB = log2(TPB); // 2^b_TPB = TPB

    if (num_box >= TPB) {
        grid.x = ceilf(num_box / (float)TPB); // m/s *m/s = (grid_size * TPB)
        grid.y = 1;
    }
    else {
        grid.x = 1; grid.y = 1;
        block.x = num_box; block.y = 1;
        b_TPB = log(num_box) / log(2);
    }

    initialDBCKernel<<<grid, block>>>(dev_Imax, dev_Imin, M, bits_M, s / 2, h, 1, b_TPB, &dev_Nr[0]);

    // subsequent DBCKernel calls. Compute box-counting for sxs grids
    s = s * 2;
    unsigned int size = M / 2;
    unsigned char Nri = 1;

    while (size > 2) {
        h = (G * s) / M;
        h = h == 0.0 ? 0.001 : h;

        unsigned long long int num_box = (M * M) / (s * s);
        unsigned char b_TPB = log2(TPB); // 2^b_TPB = TPB

        if (num_box >= TPB) {
            grid.x = ceilf(num_box / (float)TPB); // m/s *m/s = (grid_size * TPB)
            grid.y = 1;
        }
        else {
            grid.x = 1; grid.y = 1;
            block.x = num_box; block.y = 1;
            b_TPB = log(num_box) / log(2);
        }
        
        DBCKernel<<<grid, block>>>(dev_Imax, dev_Imin, M, bits_M, s / 2, h, Nri + 1, b_TPB, &dev_Nr[Nri]);
        Nri++;
        s = s * 2;
        size = size / 2;
    }

    // GPU-CPU data transfer of the box-counting
    cudaMemcpy(Nr, dev_Nr, Numr * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_Imax);
    cudaFree(dev_Imin);
    cudaFree(dev_Nr);

    return cudaStatus;
}