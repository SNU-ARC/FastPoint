/*
batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <stdio.h>
#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "cuda_utils.h"
#include "sampling_gpu.h"

#define numOfCudaCores 1024
#define MergeLen 2048

using namespace thrust::placeholders;

struct my_sort_functor
{
  int *d;
  int cols;
  my_sort_functor(int *_d, int _cols) : d(_d), cols(_cols) {};
  __host__ __device__
  bool operator()(const int &t1, const int &t2){
    int row1 = t1/cols;
    int row2 = t2/cols;
    if  (row1 < row2) return true;
    if  (row1 > row2) return false;
    if  (d[t1] < d[t2]) return true;
    return false;
  }
};


__global__ void morton_code_generation_kernel_v2(
    int b, int n, const float *__restrict__ xs, 
    const float *__restrict__ ys, const float *__restrict__ zs,
    int *__restrict__ temp,
    const float *mins, const float *grid_size) {
  // if (m <= 0) return;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= b*n) return;
  float cur_grid_size = grid_size[tid/n];
  if (cur_grid_size == 0) {
    temp[tid] = 0;
    return;
  }
  // int idx = tid * 3;
  float x = xs[tid];
  float y = ys[tid];
  float z = zs[tid];
  int batch = tid / n;
  float minx = mins[3*batch];
  float miny = mins[3*batch+1];
  float minz = mins[3*batch+2];
  int cellx = (x - minx) / cur_grid_size;
  int celly = (y - miny) / cur_grid_size;
  int cellz = (z - minz) / cur_grid_size;

  // spreadBits
  cellx = (cellx | (cellx << 10)) & 0x000f801f; //............98765..........43210
  cellx = (cellx | (cellx <<  4)) & 0x00e181c3; //........987....56......432....10
  cellx = (cellx | (cellx <<  2)) & 0x03248649; //......98..7..5..6....43..2..1..0
  cellx = (cellx | (cellx <<  2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

  celly = (celly | (celly << 10)) & 0x000f801f; //............98765..........43210
  celly = (celly | (celly <<  4)) & 0x00e181c3; //........987....56......432....10
  celly = (celly | (celly <<  2)) & 0x03248649; //......98..7..5..6....43..2..1..0
  celly = (celly | (celly <<  2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

  cellz = (cellz | (cellz << 10)) & 0x000f801f; //............98765..........43210
  cellz = (cellz | (cellz <<  4)) & 0x00e181c3; //........987....56......432....10
  cellz = (cellz | (cellz <<  2)) & 0x03248649; //......98..7..5..6....43..2..1..0
  cellz = (cellz | (cellz <<  2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

  temp[tid] = (cellx << 0) | (celly << 1) | (cellz << 2);
}

__global__ void customized_sampling_kernel(const int total_output, const int n, const int npoint, const int *keys, int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_output) return;
  int batch_idx = tid / npoint;
  keys += batch_idx * n;
  int point_idx = tid - batch_idx * npoint;
  int output = keys[point_idx * int(n / npoint)];
  out[tid] = output;
}

__global__ void gather_points_kernel_fast(int b, int c, int n, int m, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, M)
    // output:
    //      out: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    points += bs_idx * c * n + c_idx * n;
    out[0] = points[idx[0]];
}

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *points, const int *idx, float *out) {
    // points: (B, C, N)
    // idx: (B, npoints)
    // output:
    //      out: (B, C, npoints)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, points, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_points_grad_kernel_fast(int b, int c, int n, int m, const float *__restrict__ grad_out, 
    const int *__restrict__ idx, float *__restrict__ grad_points) {
    // grad_out: (B, C, M)
    // idx: (B, M)
    // output:
    //      grad_points: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    grad_out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    grad_points += bs_idx * c * n + c_idx * n;

    atomicAdd(grad_points + idx[0], grad_out[0]);
}

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *grad_out, const int *idx, float *grad_points) {
    // grad_out: (B, C, npoints)
    // idx: (B, npoints)
    // output:
    //      grad_points: (B, C, N)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_grad_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}


template <unsigned int block_size>
__global__ void find_mps_kernel(int b, int n, int m, 
    const float *__restrict__ dataset, float *__restrict__ temp, float *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
    idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
        float x2, y2, z2;
        x2 = dataset[k * 3 + 0];
        y2 = dataset[k * 3 + 1];
        z2 = dataset[k * 3 + 2];
        // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        // if (mag <= 1e-3)
        // continue;

        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        float d2 = min(d, temp[k]);
        temp[k] = d2;
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = dists[0];
    }
}


template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m, 
    const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
    idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride) {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            // if (mag <= 1e-3)
            // continue;

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
                __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }

        if (block_size >= 512) {
            if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            idxs[j] = old;
    }
}


__global__ void multi_level_filtering_gmem_kernel(int b, int n, int m, int total_nfilter, int *__restrict__ nfilter, float *__restrict__ darray, const int *__restrict__ fmatrix, int *bitmap, int *__restrict__ output) {
    // fmatrix: (n, sum(nfilter[:nlevel]))
    // bitmap: (b, nlevel * n)
    // output: (b, m)

    if (m <= 0) return;

    __shared__ int old;    // used for synchronization within threadblock

    int batch_index = blockIdx.x;

    int bitlen = (n + 31) / 32;
    bitmap = bitmap + batch_index * NLEVEL * bitlen;
    output += batch_index * m;

    int tid = threadIdx.x;
    const int stride = blockDim.x;

    unsigned int bit;

    if (tid == 0) {
        old = 0;
        output[0] = old;
    }

    if (tid < NLEVEL) {
        int remaining_bits = n % 32;
        if (remaining_bits != 0)
            bitmap[tid * bitlen + n / 32] &= (unsigned int)(1 << remaining_bits) - 1;
    }

    int offset[NLEVEL+1];
    int psum[NLEVEL];
    offset[0] = 1;
    if (threadIdx.x == 0)
        output[0] = 0;
    psum[0] = nfilter[0];
    for (int i = 1; i < NLEVEL; i++) {
        offset[i] = (int)(darray[i-1] * m);
        psum[i] = psum[i-1] + nfilter[i];
    }
    offset[NLEVEL] = m;

    __syncthreads();
    int stage_real = 0;
    for (int stage = 0; stage < NLEVEL; stage++) {
        for (int j = offset[stage]; j < offset[stage+1]; j++) {
            // Handle cases when no more points can be found to be sampled
            stage_real = max(stage, stage_real);
            if (stage_real >= NLEVEL)   // if no more stages left, just return
                return;

            // update bitmap
            // Handle cases where nfilter is greater than the number of threads
            for (int idx = tid; idx < psum[NLEVEL - stage_real - 1]; idx += blockDim.x) {
                int fidx = fmatrix[old * total_nfilter + idx];
                int bits = ~(1 << (fidx % 32));
                int base;
                for (int i = 0; i < NLEVEL - stage_real; i++) {
                    if (idx < psum[i]) {
                        base = (NLEVEL - i - 1) * bitlen;
                        break;
                    }
                }
                atomicAnd(&bitmap[base + fidx / 32], bits);
            }
            __syncthreads();

            if (tid < NLEVEL) {
                bitmap[tid * bitlen + old / 32] &= ~(1 << (old % 32));
            }
            __syncthreads();

            // select next point (This incurs race condition, but we don't care.)
            if (tid == 0)
                old = -1;
            __syncthreads();
            for (int k = tid * 32; k < n; k += stride * 32) {
                if (old != -1)
                    break;
                if (bitmap[stage_real * bitlen + k / 32])
                    old = k;
            }
            __syncthreads();

            if (old == -1) {
                if (tid == 0) {
                    old = 0;
                }
                __syncthreads();
                stage_real += 1;
                j--;
                continue;
            }

            // binary search to find the exact bit level location
            if (tid == 0) {
                bit = (unsigned int)bitmap[stage_real * bitlen + old / 32];
                for (int tmp = 16; tmp > 0; tmp >>= 1) {
                    if (bit >> tmp) {
                        old += tmp;
                        bit = bit >> tmp;
                    }
                }
                output[j] = old;
            }
            __syncthreads();
        }
    }
}


__global__ void multi_level_filtering_smem_kernel(int b, int n, int m, int total_nfilter, int *__restrict__ nfilter, float *__restrict__ darray, const int *__restrict__ fmatrix, int *bitmap, int *__restrict__ output) {
    // fmatrix: (n, sum(nfilter[:nlevel]))
    // bitmap: (b, nlevel * n)
    // output: (b, m)

    if (m <= 0) return;

    extern __shared__ int sbitmap[]; // Last element is used for sampled idx. The rest is used for bitmap.

    int batch_index = blockIdx.x;

    int bitlen = (n + 31) / 32;
    int last = NLEVEL * bitlen;
    bitmap = bitmap + batch_index * NLEVEL * bitlen;
    output += batch_index * m;

    int tid = threadIdx.x;
    const int stride = blockDim.x;

    unsigned int bit;

    for (int i = 0; i < NLEVEL; i++) {
        for (int k = tid * 32; k < n; k += stride * 32) {
            sbitmap[i * bitlen + k / 32] = -1;
        }
    }
    __syncthreads();

    if (tid == 0) {
        sbitmap[last] = 0;
        output[0] = sbitmap[last];
    }

    if (tid < NLEVEL) {
        int remaining_bits = n % 32;
        if (remaining_bits != 0) 
            sbitmap[tid * bitlen + n / 32] &= (unsigned int)(1 << remaining_bits) - 1;
    }

    int offset[NLEVEL+1];
    int psum[NLEVEL];
    offset[0] = 1;
    if (threadIdx.x == 0)
        output[0] = 0;
    psum[0] = nfilter[0];
    for (int i = 1; i < NLEVEL; i++) {
        offset[i] = (int)(darray[i-1] * m);
        psum[i] = psum[i-1] + nfilter[i];
    }
    offset[NLEVEL] = m;

    __syncthreads();
    int stage_real = 0;
    for (int stage = 0; stage < NLEVEL; stage++) { 
        for (int j = offset[stage]; j < offset[stage+1]; j++) {
            // Handle cases when no more points can be found to be sampled
            stage_real = max(stage, stage_real);
            if (stage_real >= NLEVEL)   // if no more stages left, just return
                return;

            // update bitmap
            // Handle cases where nfilter is greater than the number of threads
            int old = sbitmap[last];
            for (int idx = tid; idx < psum[NLEVEL - stage_real - 1]; idx += blockDim.x) {
                int fidx = fmatrix[old * total_nfilter + idx];
                int bits = ~(1 << (fidx % 32));
                int base;
                for (int i = 0; i < NLEVEL - stage_real; i++) {
                    if (idx < psum[i]) {
                        base = (NLEVEL - i - 1) * bitlen;
                        break;
                    }
                }
                atomicAnd(&sbitmap[base + fidx / 32], bits);
            }
            __syncthreads();
 
            if (tid < NLEVEL) {
                sbitmap[tid * bitlen + old / 32] &= ~(1 << (old % 32));
            }
    
            // select next point (This incurs race condition, but we don't care.)
            if (tid == 0)
                sbitmap[last] = -1;
            __syncthreads();
            for (int k = tid * 32; k < n; k += stride * 32) {
                if (sbitmap[last] != -1)
                    break;
                if (sbitmap[stage_real * bitlen + k / 32])
                    sbitmap[last] = k;
            }
            __syncthreads();

            if (sbitmap[last] == -1) {
                if (tid == 0) {
                    sbitmap[last] = 0;
                }
                __syncthreads();
                stage_real += 1;
                j--;
                continue;
            }
    
            // binary search to find the exact bit level location
            if (tid == 0) {
                bit = (unsigned int)sbitmap[stage_real * bitlen + sbitmap[last] / 32];
                for (int tmp = 16; tmp > 0; tmp >>= 1) {
                    if (bit >> tmp) {
                        sbitmap[last] += tmp;
                        bit = bit >> tmp;
                    }
                }
                output[j] = sbitmap[last];
            }
            __syncthreads();
        }
    }
}


void find_mps_kernel_launcher(int b, int n, int m, 
    const float *dataset, float *temp, float *idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        find_mps_kernel<1024><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
        find_mps_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
        find_mps_kernel<256><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
        find_mps_kernel<128><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
        find_mps_kernel<64><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
        find_mps_kernel<32><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
        find_mps_kernel<16><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
        find_mps_kernel<8><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
        find_mps_kernel<4><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
        find_mps_kernel<2><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
        find_mps_kernel<1><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        default:
        find_mps_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void furthest_point_sampling_kernel_launcher(int b, int n, int m, 
    const float *dataset, float *temp, int *idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        furthest_point_sampling_kernel<1024><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
        furthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
        furthest_point_sampling_kernel<256><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
        furthest_point_sampling_kernel<128><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
        furthest_point_sampling_kernel<64><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
        furthest_point_sampling_kernel<32><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
        furthest_point_sampling_kernel<16><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
        furthest_point_sampling_kernel<8><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
        furthest_point_sampling_kernel<4><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
        furthest_point_sampling_kernel<2><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
        furthest_point_sampling_kernel<1><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        default:
        furthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// points (b, n, 3) -> (3, b, n)
// keys torch.arange(b*n)
// temp (b, n)
// out (b, npoint)
// mins (b, 3)
// grid_size (b)
void edgepc_kernel_launcher(const int b, const int n, const int npoint, float *points, int *keys, int *temp, int *out, const float *mins, const float *grid_size) {
    // morton code generation
    morton_code_generation_kernel_v2<<<b*n/1024+1, 1024>>>(b, n, points, points+b*n, points+2*b*n, temp, mins, grid_size);
    cudaDeviceSynchronize();
    // sort by morton codes, get sorted index array
    thrust::device_ptr<int> keys_device(keys);
    thrust::sort(keys_device, keys_device+b*n, my_sort_functor(temp, n));
    thrust::transform(keys_device, keys_device+b*n, keys_device, _1%n);
    // fetch and gather
    int NUM_THREAD = 1024;
    int total_output = b*npoint;
    customized_sampling_kernel<<<total_output/NUM_THREAD+1, NUM_THREAD>>>(total_output, n, npoint, keys, out);
    cudaDeviceSynchronize();
}


void multi_level_filtering_kernel_launcher(int b, int n, int m, int total_nfilter, int *nfilter, float *darray, const int *filter_matrix, int *bitmap, int *output) {

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    int bitlen = (n + 31) / 32;
    if ((bitlen*NLEVEL + 1) * sizeof(int) < 90000) {
        cudaFuncSetAttribute(multi_level_filtering_smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 90000);
        multi_level_filtering_smem_kernel<<<b, n_threads, (bitlen*NLEVEL+1)*sizeof(int)>>>(b, n, m, total_nfilter, nfilter, darray, filter_matrix, bitmap, output);
    }
    else {
        multi_level_filtering_gmem_kernel<<<b, n_threads>>>(b, n, m, total_nfilter, nfilter, darray, filter_matrix, bitmap, output);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

struct float5 {
    float x, y, z, w, i;
    __host__ __device__
    float5() : x(0), y(0), z(0), w(1e20), i(0) {}
    __host__ __device__
    float5(float x, float y, float z, float w, float i) : x(x), y(y), z(z), w(w), i(i) {}

};

__global__ void devide(float4* dPoints, float4 * dtemp, int * bucketIndex, int * bucketLength, int numPartition, int bufferLength, int offset){
    extern __shared__ float shareBuffer[];

    float3* up = (float3*)&shareBuffer[0];
    float3* down = (float3*)&up[bufferLength];
    float3* sum = (float3*)&down[bufferLength];
    int* partitionDim = (int*)&sum[bufferLength];
    float* partitionValue = (float*)&partitionDim[1];

    int* shareMid = (int*)&shareBuffer[offset];

    int* lessWriteBackPtr = (int*)&shareMid[1];
    int * greaterWriteBackPtr = (int*)&lessWriteBackPtr[1];

    float4* buffer = (float4*)&shareBuffer[0];


    const int blockId = blockIdx.x;
    const int partitionStride = gridDim.x;

    const int threadId = threadIdx.x;
    const int threadStride = blockDim.x;

    for(int partitionId = blockId ; partitionId < numPartition; partitionId += partitionStride){

        float3 dimUp = {-1e10, -1e10, -1e10};
        float3 dimDown = {1e10, 1e10, 1e10};
        float3 dimSum = {0, 0, 0};

        int partitionOffset = bucketIndex[partitionId];
        int partitionLen = bucketLength[partitionId];


        float4* dataset = dPoints + partitionOffset;
        float4* dataTemp = dtemp + partitionOffset;

        for(int i = threadId; i < partitionLen; i += threadStride){
            float4 data = dataset[i];

            dimUp.x = max(dimUp.x, data.x);
            dimUp.y = max(dimUp.y, data.y);
            dimUp.z = max(dimUp.z, data.z);

            dimDown.x = min(dimDown.x, data.x);
            dimDown.y = min(dimDown.y, data.y);
            dimDown.z = min(dimDown.z, data.z);

            dimSum.x  += data.x;
            dimSum.y  += data.y;
            dimSum.z  += data.z;
        }
        up[threadId] = dimUp;
        down[threadId] = dimDown;
        sum[threadId] = dimSum;
        __syncthreads();
        //reduce
        for (int32_t active_thread_num = threadStride / 2; active_thread_num >= 1; active_thread_num /= 2) {
            if (threadId < active_thread_num) {
                up[threadId].x = max(up[threadId].x, up[threadId + active_thread_num].x);
                up[threadId].y = max(up[threadId].y, up[threadId + active_thread_num].y);
                up[threadId].z = max(up[threadId].z, up[threadId + active_thread_num].z);

                down[threadId].x = min(down[threadId].x, down[threadId + active_thread_num].x);
                down[threadId].y = min(down[threadId].y, down[threadId + active_thread_num].y);
                down[threadId].z = min(down[threadId].z, down[threadId + active_thread_num].z);

                sum[threadId].x += sum[threadId + active_thread_num].x;
                sum[threadId].y += sum[threadId + active_thread_num].y;
                sum[threadId].z += sum[threadId + active_thread_num].z;
            }
            __syncthreads();
        }
        if (threadId == 0) {
            //find out the split dim and middle value
            float3 range = {up[0].x - down[0].x,
                            up[0].y - down[0].y,
                            up[0].z - down[0].z };
            int dim = 0;
            float middleValue = sum[0].x / (partitionLen+0.0);

            if(range.x > range.y && range.x > range.z) dim = 0;
            if(range.y > range.x && range.y > range.z) {dim = 1;middleValue = sum[0].y / (partitionLen+0.0);}
            if(range.z > range.x && range.z > range.y) {dim = 2;middleValue = sum[0].z / (partitionLen+0.0);}

            (*partitionDim) = dim;
            (*partitionValue) = middleValue;

            (* lessWriteBackPtr) = 0;
            (* greaterWriteBackPtr) = 0;
        }
        __syncthreads();
        //merge sort
        int divideDim = (*partitionDim);
        float divideValue = (*partitionValue);
        const int partMergeLen = MergeLen;


        for(int dataPtr = 0; dataPtr < partitionLen; dataPtr += partMergeLen){
            const int currentPartLen = min(partMergeLen, partitionLen - dataPtr);

            //copy global memory to share memory
            float4* partData = (float4*) &dataset[dataPtr];

            for(int i = threadId; i < currentPartLen; i += threadStride){
                buffer[i] = partData[i];
            }
            __syncthreads();

            //merge sort
            int mid = 0;
            for(int stride = 1; stride < currentPartLen; stride *= 2){
                for(int threadStart = threadId * 2 * stride; threadStart < currentPartLen; threadStart += threadStride * stride * 2){
                    int left = threadStart;
                    int endLeft = (left + stride)  < currentPartLen ?  (left + stride) : currentPartLen; //边界条件: left < endLeft

                    int right = (left + 2 * stride) < currentPartLen ? (left + 2 * stride - 1) : (currentPartLen -1);
                    int endRight = (left + stride)  < currentPartLen ? (left + stride - 1) : (currentPartLen - 1); //边界条件: right > endRight

                    if(divideDim == 0) {
                        while (left < endLeft) {
                            if (buffer[left].x <= divideValue) left++;
                            else break;

                        }
                        while (right > endRight) {
                            if (buffer[right].x >= divideValue) right--;
                            else break;
                        }
                    } else{
                        if(divideDim == 1){
                            while (left < endLeft) {
                                if (buffer[left].y <= divideValue) left++;
                                else break;
                            }
                            while (right > endRight) {
                                if (buffer[right].y >= divideValue) right--;
                                else break;
                            }
                        } else{
                            while (left < endLeft) {
                                if (buffer[left].z <= divideValue) left++;
                                else break;
                            }
                            while (right > endRight) {
                                if (buffer[right].z >= divideValue) right--;
                                else break;
                            }
                        }
                    }

                    while((left < endLeft) && (right > endRight)){
                        //swap
                        float4 tmp = buffer[left];
                        buffer[left] = buffer[right];
                        buffer[right] = tmp;
                        left ++;
                        right --;
                    }
                    if(left < endLeft) mid = left;
                    else mid = right + 1;
                }
                __syncthreads();
            }
            // sync mid
            if(threadId == 0){
                (*shareMid) = mid;
            }
            __syncthreads();

            mid = (*shareMid);

            int lessPtr = *lessWriteBackPtr;
            int greaterPtr = *greaterWriteBackPtr;

            //copy back to data
            float4 * lessGlobalData = (float4 * )&(dataset[lessPtr]);
            float4 * greaterGlobalData = (float4 * )&(dataTemp[greaterPtr]);

            for(int i = threadId; i < mid; i += threadStride){
                lessGlobalData[i] = buffer[i];
            }
            //copy to temp

            float4 * greaterBuffer = (float4 * )&buffer[mid];

            const int greaterLen = currentPartLen - mid;

            for(int i = threadId ; i < greaterLen; i += threadStride){
                greaterGlobalData[i] = greaterBuffer[i];
            }
            // update lessWriteBackPtr and greaterWriteBackPtr
            if(threadId == 0){
                (*lessWriteBackPtr) += mid;
                (*greaterWriteBackPtr) += greaterLen;
            }
            __syncthreads();
        }
        //copy tempdata to dataset
        const int greaterLen = (*greaterWriteBackPtr);
        const int lessLen = (*lessWriteBackPtr);
        float4* dataset_2 = (float4*)&dataset[lessLen];

        for(int i = threadId; i < greaterLen; i += threadStride){
            dataset_2[i] = dataTemp[i];
        }
        if(threadId == 0){
            //update bucketIndex and bucketLength
            bucketIndex[partitionId + numPartition] = partitionOffset + lessLen ;
            bucketLength[partitionId + numPartition] = partitionLen - lessLen;
            bucketLength[partitionId] = lessLen;
        }
        __syncthreads();
    }
}


__global__ void generateBoundbox(int * bucketIndex, int * bucketLength, float4 * dPoints, int numPartition, int bufferLength, float3 * up, float3 * down){
    extern __shared__ float3 buffer[];

    float3* shareUp = buffer;
    float3* shareDown = (float3*)&up[bufferLength];

    const int partitionStride = gridDim.x;
    const int threadStride = blockDim.x;

    for(int partitionId = blockIdx.x; partitionId < numPartition; partitionId += partitionStride) {
        const int shareMemoryIdx = threadIdx.x + blockIdx.x * blockDim.x;

        float3 *threadUp = (float3 *) &shareUp[shareMemoryIdx];
        float3 *threadDown = (float3 *) &shareDown[shareMemoryIdx];

        float3 dimUp = {-1e10, -1e10, -1e10};
        float3 dimDown = {1e10, 1e10, 1e10};

        const int partitionOffset = bucketIndex[partitionId];
        const int partitionLen = bucketLength[partitionId];

        float4 *dataset = dPoints + partitionOffset;
        for (int i = threadIdx.x; i < partitionLen; i += threadStride) {
            float4 data = dataset[i];
            dimUp.x = max(dimUp.x, data.x);
            dimUp.y = max(dimUp.y, data.y);
            dimUp.z = max(dimUp.z, data.z);

            dimDown.x = min(dimDown.x, data.x);
            dimDown.y = min(dimDown.y, data.y);
            dimDown.z = min(dimDown.z, data.z);
        }
        threadUp[0] = dimUp;
        threadDown[0] = dimDown;
        __syncthreads();
        //reduce
        for (int32_t active_thread_num = blockDim.x / 2; active_thread_num >= 1; active_thread_num /= 2) {
            if (threadIdx.x < active_thread_num) {
                threadUp[0].x = max(threadUp[0].x, threadUp[active_thread_num].x);
                threadUp[0].y = max(threadUp[0].y, threadUp[active_thread_num].y);
                threadUp[0].z = max(threadUp[0].z, threadUp[active_thread_num].z);

                threadDown[0].x = min(threadDown[0].x, threadDown[active_thread_num].x);
                threadDown[0].y = min(threadDown[0].y, threadDown[active_thread_num].y);
                threadDown[0].z = min(threadDown[0].z, threadDown[active_thread_num].z);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            up[partitionId] = threadUp[0];
            down[partitionId] = threadDown[0];
        }
        __syncthreads();
    }
}

void buildKDTree(int * bucketIndex, int * bucketLength, float4 * ptr, int kd_high, float3 * up, float3 * down, int point_data_size ){
    int currentLevel=0;
    int nThreads, nBlocks;
    cudaError_t err;
    float4 * dtemp;
    cudaMalloc((void **)&dtemp, point_data_size*sizeof(float4));

    while(currentLevel<kd_high)
    {
        nBlocks =  ((int) pow(2.0f,currentLevel+0.0f));
        nThreads = currentLevel > 2 ? std::max(32, 4096/nBlocks) : 1024;

        const int bytes = std::max( nThreads*3*sizeof(float3) + sizeof(int) + sizeof(float) , MergeLen * sizeof(float4)) + 3 * sizeof(int);
        const int offset = (bytes/sizeof(float)) - 3;
        devide<<<nBlocks, nThreads,bytes>>>
            (ptr,dtemp, bucketIndex, bucketLength, nBlocks, nThreads, offset);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        currentLevel++;
    }
    cudaDeviceSynchronize();
    nBlocks = 1 << kd_high;
    nThreads = numOfCudaCores/nBlocks;
    int ThreadSize = nBlocks * nThreads;
    generateBoundbox<<<nBlocks, nThreads, ThreadSize * 2 * sizeof(float3) >>>(bucketIndex, bucketLength, ptr, nBlocks,ThreadSize, up, down);
    cudaFree(dtemp);
}

__device__ void merge(float *__restrict__ dists, int *__restrict__ dists_i,int tid, int block_size){
    if (block_size >= 4096) {
        if (tid < 2048) {
            __update(dists, dists_i, tid, tid + 2048);
        }
        __syncthreads();
    }
    if (block_size >= 2048) {
        if (tid < 1024) {
            __update(dists, dists_i, tid, tid + 1024);
        }
        __syncthreads();
    }

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }

    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }

    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }
}

template <unsigned int block_size>  __global__
void reduce_kernel(float5* bucketTable, float5 * result, int offset){
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    const int tid = threadIdx.x;

    dists[tid] = bucketTable[tid].w;
    dists_i[tid] = tid;

    merge(dists, dists_i, tid, block_size);

    if(tid == 0){
        const float5 maxPoint = bucketTable[dists_i[0]];
        result[offset] = float5({maxPoint.x, maxPoint.y, maxPoint.z, dists[0], maxPoint.i});
    }
    __syncthreads();
}

void reduce(int bucketSize, float5* bucketTable, float5 * result, int offset){
    assert(bucketSize <=numOfCudaCores);
    dim3 BucketDim(bucketSize);
    switch (bucketSize) {
        case 1:reduce_kernel<1><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 2:reduce_kernel<2><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 4:reduce_kernel<4><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 8:reduce_kernel<8><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 16:reduce_kernel<16><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 32:reduce_kernel<32><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 64:reduce_kernel<64><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 128:reduce_kernel<128><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 256:reduce_kernel<256><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 512:reduce_kernel<512><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 1024:reduce_kernel<1024><<<1, BucketDim>>>(bucketTable, result, offset);break;
    }
}

__device__ float pow2(float a){
    return a*a;
}

template <unsigned int block_size> __global__
void sample_kernel(int *bucketIndex, int *bucketLength, float4 *ptr,float* temp, float5 *result, int offset, bool * needToDeal, float5 * bucketTable) {
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    const int bucketPtr = blockIdx.x;
    if (needToDeal[bucketPtr]) {

        const int tid = threadIdx.x;

        const float origin_x = result[offset - 1].x;
        const float origin_y = result[offset - 1].y;
        const float origin_z = result[offset - 1].z;

        const int partitionLen = bucketLength[bucketPtr];
        const int partitionOffset = bucketIndex[bucketPtr];

        float4 *dataset = (float4 *) &ptr[partitionOffset];
        float *distTemp = (float *) &temp[partitionOffset];

        float best = -1;
        int besti = 0;
        for (int k = tid; k < partitionLen; k += block_size) {
            const float4 point = dataset[k];
            const float d = pow2((point.x - origin_x)) + pow2((point.y - origin_y)) + pow2((point.z - origin_z));
            const float d2 = min(d, distTemp[k]);
            distTemp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        merge(dists, dists_i, tid, block_size);

        if (tid == 0) {
            const float4 maxPoint_ = dataset[dists_i[0]];
            bucketTable[bucketPtr] = float5(maxPoint_.x, maxPoint_.y, maxPoint_.z, dists[0], maxPoint_.w);
        }
        __syncthreads();
    }
}


__global__ void checkBucket(float5* bucketTable ,float5 *result,int i,float3 *up,float3 *down,bool *needToDeal) {
    const int tid = threadIdx.x;

    const float5 origin_point = result[i-1];

    const float5 bucketMaxPoint = bucketTable[tid];
    const float3 bucketUp = up[tid];
    const float3 bucketDown = down[tid];

    const float last_dist = bucketMaxPoint.w;
    const float cur_dist = pow2((origin_point.x - bucketMaxPoint.x)) +
                               pow2((origin_point.y - bucketMaxPoint.y))  +
                               pow2((origin_point.z - bucketMaxPoint.z));

    const float bound_dist = pow2(max(origin_point.x, bucketUp.x) - bucketUp.x) + pow2(bucketDown.x - min(origin_point.x, bucketDown.x)) +
                                 pow2(max(origin_point.y, bucketUp.y) - bucketUp.y) + pow2(bucketDown.y - min(origin_point.y, bucketDown.y)) +
                                 pow2(max(origin_point.z, bucketUp.z) - bucketUp.z) + pow2(bucketDown.z - min(origin_point.z, bucketDown.z)) ;
    needToDeal[tid] = (cur_dist <= last_dist || bound_dist < last_dist);
}

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}


void sample(int * bucketIndex, int * bucketLength, float4 * ptr, int pointSize,  int bucketSize, float3 * up, float3 * down, int sample_number, float5 * result, float4 *seed){

    thrust::device_vector<float> tempVector(pointSize);
    thrust::fill(tempVector.begin(), tempVector.end(), 1e20);
    float * temp = thrust::raw_pointer_cast(&tempVector[0]);

    thrust::device_vector<float5> bucketTableVector(bucketSize);
    thrust::fill(bucketTableVector.begin(), bucketTableVector.end(), float5(0,0,0,1e20,0));
    float5 * bucketTable = thrust::raw_pointer_cast(&bucketTableVector[0]);

    thrust::device_vector<bool> needToDealVector(bucketSize);
    bool * needToDeal = thrust::raw_pointer_cast(&needToDealVector[0]);

#ifdef DEBUG_GG
    printf("bytes:%d\n", bytes);
#endif
    cudaMemcpy(result, seed, sizeof(float5), cudaMemcpyDeviceToDevice); //first point

    dim3 bucketDim(bucketSize);
    for(int i = 1; i < sample_number; i++){
        checkBucket<<<1,bucketDim>>>(bucketTable, result, i, up, down, needToDeal);
        CudaCheckError();
        sample_kernel<numOfCudaCores><<<bucketDim,numOfCudaCores >>>(bucketIndex, bucketLength, ptr, temp , result, i ,needToDeal, bucketTable);
        CudaCheckError();
        reduce(bucketSize, bucketTable,result,i);
        CudaCheckError();
    }

}

void QuickFPS_launcher(int b, int n, int sample_number, int kd_high, float *xyz, float *output, int *bucketIndex, int *bucketLength) {
    int bucketSize = 1 << kd_high;

    float4 *ptr = reinterpret_cast<float4*>(xyz);
    const int point_data_size = n;

    float4* seed;
    cudaMalloc((void**)&seed, sizeof(float4));
    cudaMemcpy(seed, ptr, sizeof(float4), cudaMemcpyDeviceToDevice);

    float3 *up;
    float3 *down;

    cudaMalloc((void **)&up, bucketSize*sizeof(float3));
    cudaMalloc((void **)&down, bucketSize*sizeof(float3));
    
    float5 *result = reinterpret_cast<float5*>(output);

    buildKDTree(bucketIndex, bucketLength, ptr, kd_high, up, down, point_data_size);
    sample(bucketIndex, bucketLength, ptr, point_data_size, bucketSize, up, down, sample_number, result, seed);

    cudaFree(up);
    cudaFree(down);
    cudaFree(seed);
}
