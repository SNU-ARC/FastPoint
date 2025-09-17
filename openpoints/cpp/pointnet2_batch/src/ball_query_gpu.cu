/*
batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"

__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}

__global__ void fused_ball_query_kernel_fast(int b, int n, int m, float *__restrict__ radius, int *__restrict__ nsample, const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *idx, int nthreads) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // idx: (B, M, sum(nsample[0:NLEVEL + 1]))
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bs_idx = blockIdx.y;
    int pt_idx = tid / nthreads;
    int thread_idx = tid % nthreads;

    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;

    int total_nsample = 0;
    float radius2[NLEVEL + 1];
    int offset[NLEVEL + 1];
    extern __shared__ int shared[];
    int *cnt = &shared[(threadIdx.x / nthreads) * (NLEVEL + 1)];
    offset[0] = 0;

    for (int i = 0; i < NLEVEL + 1; ++i) {
        total_nsample += nsample[i];
        radius2[i] = radius[i] * radius[i];
        if(thread_idx == 0)
            cnt[i] = 0;
        if (i > 0)
            offset[i] = offset[i-1] + nsample[i-1];
    }

    idx += bs_idx * m * total_nsample + pt_idx * total_nsample;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    __syncthreads();

    for (int k = thread_idx; k < n; k += nthreads) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);

        for (int i = 0; i < NLEVEL + 1; ++i) {
            if (d2 < radius2[i]) {
                for (int j = i; j < NLEVEL + 1; ++j) {
                    int cnt_j = atomicAdd(&cnt[j], 1);
                    if (cnt_j < nsample[j]) {
                        if (cnt_j == 0) {
                            for (int l = 0; l < nsample[j]; ++l) {
                                idx[offset[j] + l] = k;
                            }
                        }
                        idx[offset[j] + cnt_j] = k;
                    }
                }
                break;
            }
        }
    }
}

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<blocks, threads>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void fused_ball_query_kernel_launcher_fast(int b, int n, int m, float *radius, int *nsample, const float *new_xyz, const float *xyz, int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // idx: (B, M, sum(nsample[0:NLEVEL + 1]))

    cudaError_t err;

    int nthreads = 32;
    dim3 blocks(DIVUP(m * nthreads, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    size_t shared = (THREADS_PER_BLOCK / nthreads) * (NLEVEL + 1) * sizeof(int);

    fused_ball_query_kernel_fast<<<blocks, threads, shared>>>(b, n, m, radius, nsample, new_xyz, xyz, idx, nthreads);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
