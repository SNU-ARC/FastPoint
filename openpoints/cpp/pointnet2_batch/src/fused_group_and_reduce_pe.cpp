#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include "fused_group_and_reduce_pe_gpu.h"


int fused_group_and_reduce_pe_grad_wrapper_fast(int b, int c, int n, int npoints,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor idx_pe_tensor, at::Tensor grad_points_tensor, at::Tensor grad_pe_tensor) {

    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *source_idx = idx_tensor.data_ptr<int>();
    const int *source_idx_pe = idx_pe_tensor.data_ptr<int>();
    float *grad_points = grad_points_tensor.data_ptr<float>();
    float *grad_pe = grad_pe_tensor.data_ptr<float>();

    fused_group_and_reduce_pe_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, source_idx, source_idx_pe, grad_points, grad_pe);
    return 1;
}


int fused_group_and_reduce_pe_wrapper_fast(int b, int c, int n, int npoints, int nsample,
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor pe_tensor, at::Tensor out_tensor, at::Tensor source_idx, at::Tensor source_idx_pe) {

    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *pe = pe_tensor.data_ptr<float>();
    float *out = out_tensor.data_ptr<float>();
    int *s_idx = source_idx.data_ptr<int>();
    int *s_idx_pe = source_idx_pe.data_ptr<int>();

    fused_group_and_reduce_pe_kernel_launcher_fast(b, c, n, npoints, nsample, points, idx, pe, out, s_idx, s_idx_pe);
    return 1;
}
