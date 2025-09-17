/*
batch version of point interpolation, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "interpolate_gpu.h"


__global__ void three_nn_kernel_fast(int b, int n, int m, const float *__restrict__ unknown, 
    const float *__restrict__ known, float *__restrict__ dist2, int *__restrict__ idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output: 
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)
    
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    unknown += bs_idx * n * 3 + pt_idx * 3;
    known += bs_idx * m * 3;
    dist2 += bs_idx * n * 3 + pt_idx * 3;
    idx += bs_idx * n * 3 + pt_idx * 3;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = k;
        } 
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = k;
        } 
        else if (d < best3) {
            best3 = d; besti3 = k;
        }
    }
    dist2[0] = best1; dist2[1] = best2; dist2[2] = best3;
    idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;
}

__global__ void fused_three_nn_kernel_fast(int b, int n, int m, int nfilter, const float *__restrict__ unknown, 
    const float *__restrict__ known, const int *__restrict__ ball_query, float *__restrict__ dist2, int *__restrict__ idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // ball_query: (N, nfilter)
    // output: 
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)
    
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    unknown += bs_idx * n * 3 + pt_idx * 3;
    known += bs_idx * m * 3;
    ball_query += pt_idx * nfilter;
    dist2 += bs_idx * n * 3 + pt_idx * 3;
    idx += bs_idx * n * 3 + pt_idx * 3;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];

    float dist_array[1024]; // max nfilter
    int index_array[1024];
    int count = 0;

    int dup = ball_query[0];

    // Filter duplicated data and -1
    for (int i = 0; i < nfilter; ++i) {
        int k = ball_query[i];
        if (k == -1 || k == dup && i != 0) continue;

        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

        dist_array[count] = d;
        index_array[count] = k;
        count++;
    }
    
    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = -1, besti2 = -1, besti3 = -1;

    if(count >= 3) {
        // Exists at least 3 point in ball query result
        for (int i = 0; i < count; ++i) {
            float d = dist_array[i];
            int k = index_array[i];
            if (d < best1) {
                best3 = best2; besti3 = besti2;
                best2 = best1; besti2 = besti1;
                best1 = d; besti1 = k;
            } 
            else if (d < best2) {
                best3 = best2; besti3 = besti2;
                best2 = d; besti2 = k;
            } 
            else if (d < best3) {
                best3 = d; besti3 = k;
            }
        }
    } else {
        // Not enough point in ball query result
        for (int k = 0; k < m; ++k) {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            if (d < best1) {
                best3 = best2; besti3 = besti2;
                best2 = best1; besti2 = besti1;
                best1 = d; besti1 = k;
            } 
            else if (d < best2) {
                best3 = best2; besti3 = besti2;
                best2 = d; besti2 = k;
            } 
            else if (d < best3) {
                best3 = d; besti3 = k;
            }
        }
    }

    dist2[0] = best1; dist2[1] = best2; dist2[2] = best3;
    idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;
}

__global__ void update_distance_kernel_fast(int b, int n, int m, int nfilter, const float *__restrict__ xyz, const float *__restrict__ sampled_xyz, const int *__restrict__ ball_query, float *__restrict__ dist2) {
    // xyz: (B, N, 3)
    // ball_query: (N, nfilter)
    // output: 
    //      dist2: (B, N)
    
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    xyz += bs_idx * n * 3 + pt_idx * 3;
    sampled_xyz += bs_idx * m * 3;
    ball_query += pt_idx * nfilter;
    dist2 += bs_idx * n + pt_idx;

    float ux = xyz[0];
    float uy = xyz[1];
    float uz = xyz[2];

    xyz -= bs_idx * n * 3 + pt_idx * 3;  // xyz pointer will be reused

    float dist_array[1024]; // max nfilter
    int count = 0;

    int dup = ball_query[0];

    // Filter duplicated data and -1
    for (int i = 0; i < nfilter; ++i) {
        int k = ball_query[i];
        if (k == -1 || k == dup && i != 0) continue;

        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

        dist_array[count] = d;
        count++;
    }
    
    double best = 1e40;
    if(count >= 1) {
        // Exists at least 1 point in ball query result
        for (int i = 0; i < count; ++i) {
            float d = dist_array[i];
            if (d < best) {
                best = d;
            } 
        }
    } else {
        // Not enough point in ball query result
        for (int k = 0; k < m; ++k) {
            float x = sampled_xyz[k * 3 + 0];
            float y = sampled_xyz[k * 3 + 1];
            float z = sampled_xyz[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            if (d < best) {
                best = d;
            } 
        }
    }
    dist2[0] = best;
}

__global__ void inv_ball_query_kernel_fast(int b, int n, int m, int nfilter, 
    const int*__restrict__ idx, const int *__restrict__ ball_query, int *__restrict__ inv, int *__restrict__ pos) {
    // ball_query: (B, N, nfilter)
    // output: 
    //      inv: (B, N)
    
    int bs_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || tid >= n * nfilter) return;

    idx += bs_idx * m;
    ball_query += bs_idx * n * nfilter + tid;
    inv += bs_idx * n;
    pos += bs_idx * n * nfilter + tid;

    if(tid < m) {
        int j = idx[tid];
        if (j >= 0 && j < n) {
            inv[j] = j;
        }
    }
}

__global__ void extract_ball_query_kernel_fast(int b, int n, int m, int nfilter, 
    const int*__restrict__ idx, const int *__restrict__ ball_query, int *__restrict__ inv, int *__restrict__ pos) {
    // inv: (B, N)
    // output: 
    //      pos: (B, N, nfilter)
        // ball_query: (B, N, nfilter)
    // output: 
    //      inv: (B, N)
    int bs_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || tid >= n * nfilter) return;

    idx += bs_idx * m;
    ball_query += bs_idx * n * nfilter + tid;
    inv += bs_idx * n;
    pos += bs_idx * n * nfilter + tid;

    pos[0] = inv[ball_query[0]];
}

__global__ void fused_convert_ball_query_kernel_fast(int b, int n, int m, int nfilter, 
    const int*__restrict__ idx, const int *__restrict__ ball_query, int *__restrict__ inv, int *__restrict__ pos) {
    // ball_query: (B, N, nfilter)
    // inv: (B, N)
    // output: 
    //      pos: (B, N, nfilter)
    
    int bs_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || tid >= n * nfilter) return;

    idx += bs_idx * m;
    ball_query += bs_idx * n * nfilter + tid;
    inv += bs_idx * n;
    pos += bs_idx * n * nfilter + tid;

    if(tid < m) {
        int j = idx[tid];
        if (j >= 0 && j < n) {
            inv[j] = tid;
        }
    }

    __syncthreads();

    pos[0] = inv[ball_query[0]];
}

void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown, 
    const float *known, float *dist2, int *idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output: 
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    three_nn_kernel_fast<<<blocks, threads>>>(b, n, m, unknown, known, dist2, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void fused_three_nn_kernel_launcher_fast(int b, int n, int m, int nfilter, const float *unknown, 
    const float *known, const int *ball_query, float *dist2, int *idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output: 
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    fused_three_nn_kernel_fast<<<blocks, threads>>>(b, n, m, nfilter, unknown, known, ball_query, dist2, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void update_distance_kernel_launcher_fast(int b, int n, int m, int nfilter, const float *xyz, const float *sampled_xyz, const int *ball_query, float *dist2) {
    // xyz: (B, N, 3)
    // output: 
    //      dist2: (B, N)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    update_distance_kernel_fast<<<blocks, threads>>>(b, n, m, nfilter, xyz, sampled_xyz, ball_query, dist2);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void extract_ball_query_launcher_fast(int b, int n, int m, int nfilter, const int *idx, const int *ball_query,
	int *inv, int *pos) {
    // idx: (B, M)
    // ball_query: (B, N, nfilter)
    // output: 
    //      pos: (B, N, nfilter)
        cudaError_t err;

    dim3 blocks(DIVUP(n * nfilter, THREADS_PER_BLOCK), b);
    dim3 threads(THREADS_PER_BLOCK);

    inv_ball_query_kernel_fast<<<blocks, threads>>>(b, n, m, nfilter, idx, ball_query, inv, pos);
    extract_ball_query_kernel_fast<<<blocks, threads>>>(b, n, m, nfilter, idx, ball_query, inv, pos);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void fused_convert_ball_query_launcher_fast(int b, int n, int m, int nfilter, const int *idx, const int *ball_query,
	int *inv, int *pos) {
    // idx: (B, M)
    // ball_query: (B, N, nfilter)
    // output: 
    //      pos: (B, N, nfilter)
        cudaError_t err;

    dim3 blocks(DIVUP(n * nfilter, THREADS_PER_BLOCK), b);
    dim3 threads(THREADS_PER_BLOCK);

    fused_convert_ball_query_kernel_fast<<<blocks, threads>>>(b, n, m, nfilter, idx, ball_query, inv, pos);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void three_interpolate_kernel_fast(int b, int c, int m, int n, const float *__restrict__ points, 
    const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ out) {
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    weight += bs_idx * n * 3 + pt_idx * 3;
    points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;
    out += bs_idx * c * n + c_idx * n;

    out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] + weight[2] * points[idx[2]];
}

void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n, 
    const float *points, const int *idx, const float *weight, float *out) {
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_kernel_fast<<<blocks, threads>>>(b, c, m, n, points, idx, weight, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void three_interpolate_grad_kernel_fast(int b, int c, int n, int m, const float *__restrict__ grad_out, 
    const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ grad_points) {
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;
    
    grad_out += bs_idx * c * n + c_idx * n + pt_idx;
    weight += bs_idx * n * 3 + pt_idx * 3;
    grad_points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;


    atomicAdd(grad_points + idx[0], grad_out[0] * weight[0]);
    atomicAdd(grad_points + idx[1], grad_out[0] * weight[1]);
    atomicAdd(grad_points + idx[2], grad_out[0] * weight[2]);
}

void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m, const float *grad_out, 
    const int *idx, const float *weight, float *grad_points) {
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_grad_kernel_fast<<<blocks, threads>>>(b, c, n, m, grad_out, idx, weight, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
