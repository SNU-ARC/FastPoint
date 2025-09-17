#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "fused_group_and_reduce_pe_gpu.h"


__global__ void fused_group_and_reduce_pe_grad_kernel_fast(int b, int c, int n, int npoints,
    const float *__restrict__ grad_out, const int *__restrict__ source_idx, const int *__restrict__ source_idx_pe, float *__restrict__ grad_points, float *__restrict__ grad_pe) {
    // grad_out: (B, C, npoints)
    // source_idx: (B, C, npoints)
    // source_idx_pe: (B, C, npoints)
    // output:
    //      grad_points: (B, C, N)
    //      grad_pe: (B, C, npoints, nsample)
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int output_idx = bs_idx * c * npoints + c_idx * npoints + pt_idx;
    grad_out += output_idx;
    source_idx += output_idx;
    source_idx_pe += output_idx;
 
    float grad = grad_out[0];
    atomicAdd(grad_points + source_idx[0] , grad);  // grad_points update
    grad_pe[source_idx_pe[0]] = grad;
}

void fused_group_and_reduce_pe_grad_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *grad_out, const int *source_idx, const int *source_idx_pe, float *grad_points, float *grad_pe) {
    // grad_out: (B, C, npoints)
    // source_idx: (B, C, npoints)
    // source_idx_pe: (B, C, npoints)
    // output:
    //      grad_points: (B, C, N)
    //      grad_pe: (B, C, npoints, nsample)
    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    fused_group_and_reduce_pe_grad_kernel_fast<<<blocks, threads, 0>>>(b, c, n, npoints, grad_out, source_idx, source_idx_pe, grad_points, grad_pe);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void fused_group_and_reduce_pe_kernel_fast(int b, int c, int n, int npoints, int nsample, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ pe, float *__restrict__ out, int *__restrict__ source_idx, int *__restrict__ source_idx_pe) {
    // points: (B, C, N)
    // pe: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints)
    //      source_idx: (B, C, npoints)
    //      source_idx_pe: (B, C, npoints)
    __shared__ float sdata[THREADS_PER_BLOCK/2];
    __shared__ int sidx[THREADS_PER_BLOCK/2];
    __shared__ int sidxpe[THREADS_PER_BLOCK/2];

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / (nsample/2);
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % (nsample/2);

    // Gather input data to the shared mem.
    int smem_out_idx = threadIdx.x;
    const int *idx1 = idx + bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    const int *idx2 = idx + bs_idx * npoints * nsample + pt_idx * nsample + sample_idx + nsample/2;
    int pe_idx1 = bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    int pe_idx2 = bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx + nsample/2;
    int smem_in_idx1 = bs_idx * c * n + c_idx * n + idx1[0];
    int smem_in_idx2 = bs_idx * c * n + c_idx * n + idx2[0];

    float val1 = points[smem_in_idx1] + pe[pe_idx1];
    float val2 = points[smem_in_idx2] + pe[pe_idx2];
    float result = fmaxf(val1, val2);
    sdata[smem_out_idx] = result;
    sidx[smem_out_idx] = (result == val1) ? (smem_in_idx1) : (smem_in_idx2);
    sidxpe[smem_out_idx] = (result == val1) ? (pe_idx1) : (pe_idx2);
    __syncthreads();

    // Perform reduction.
    for (int s = nsample/4; s > 32; s>>=1) {
        if (sample_idx < s) {
            float val1 = sdata[smem_out_idx];
            float val2 = sdata[smem_out_idx + s];
            float result = fmaxf(val1, val2);
            sdata[smem_out_idx] = result;
            sidx[smem_out_idx] = (result == val1) ? (sidx[smem_out_idx]) : (sidx[smem_out_idx + s]);
            sidxpe[smem_out_idx] = (result == val1) ? (sidxpe[smem_out_idx]) : (sidxpe[smem_out_idx + s]);
        }
        __syncthreads();
    }
    for (int s = MIN(nsample/4, 32); s > 0; s>>=1) {
        if (sample_idx < s) {
            float val1 = sdata[smem_out_idx];
            float val2 = sdata[smem_out_idx + s];
            float result = fmaxf(val1, val2);
            sdata[smem_out_idx] = result;
            sidx[smem_out_idx] = (result == val1) ? (sidx[smem_out_idx]) : (sidx[smem_out_idx + s]);
            sidxpe[smem_out_idx] = (result == val1) ? (sidxpe[smem_out_idx]) : (sidxpe[smem_out_idx + s]);
        }
    }

    int in_idx = smem_out_idx;
    int out_idx = bs_idx * c * npoints + c_idx * npoints + pt_idx;

    if (sample_idx == 0) {
        out[out_idx] = sdata[in_idx];
        source_idx[out_idx] = sidx[in_idx];
        source_idx_pe[out_idx] = sidxpe[in_idx];
    }
}


void fused_group_and_reduce_pe_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *pe, float *out, int *source_idx, int *source_idx_pe) {
    // points: (B, C, N)
    // pe: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints)
    //      source_idx: (B, C, npoints)
    //      source_idx_pe: (B, C, npoints)
    cudaError_t err;
    int blocksize = MAX(nsample, THREADS_PER_BLOCK);
    dim3 blocks(DIVUP(npoints * nsample, blocksize), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(blocksize/2);

    fused_group_and_reduce_pe_kernel_fast<<<blocks, threads, 0>>>(b, c, n, npoints, nsample, points, idx, pe, out, source_idx, source_idx_pe);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


