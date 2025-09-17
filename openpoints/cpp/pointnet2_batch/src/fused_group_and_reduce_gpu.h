#ifndef _FUSED_GROUP_AND_REDUCE_GPU_H
#define _FUSED_GROUP_AND_REDUCE_GPU_H

#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>


int fused_group_and_reduce_wrapper_fast(int b, int c, int n, int npoints, int nsample, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor, at::Tensor source_idx);

void fused_group_and_reduce_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *out, int *source_idx);

int fused_group_and_reduce_grad_wrapper_fast(int b, int c, int n, int npoints,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor);

void fused_group_and_reduce_grad_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *grad_out, const int *source_idx, float *grad_points);

#endif
