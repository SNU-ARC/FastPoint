#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include "fused_group_and_reduce_gpu.h"


int fused_group_and_reduce_grad_wrapper_fast(int b, int c, int n, int npoints,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    float *grad_points = grad_points_tensor.data_ptr<float>();
    const int *source_idx = idx_tensor.data_ptr<int>();
    const float *grad_out = grad_out_tensor.data_ptr<float>();

    fused_group_and_reduce_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, source_idx, grad_points);
    return 1;
}


int fused_group_and_reduce_wrapper_fast(int b, int c, int n, int npoints, int nsample,
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor, at::Tensor source_idx) {

    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();
    int *s_idx = source_idx.data_ptr<int>();

    fused_group_and_reduce_kernel_launcher_fast(b, c, n, npoints, nsample, points, idx, out, s_idx);
    return 1;
}
