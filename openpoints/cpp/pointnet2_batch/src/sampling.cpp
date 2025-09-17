/*
batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "sampling_gpu.h"




int gather_points_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out);
    return 1;
}


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_points = grad_points_tensor.data<float>();

    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points);
    return 1;
}


int find_mps_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    float *idx = idx_tensor.data<float>();

    find_mps_kernel_launcher(b, n, m, points, temp, idx);
    return 1;
}


int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    return 1;
}

int edgepc_sampling_wrapper(int b, int n, int npoint, at::Tensor points_tensor, at::Tensor keys_tensor, at::Tensor temp_tensor, at::Tensor out_tensor, at::Tensor mins_tensor, at::Tensor grid_size_tensor) {
    float *points = points_tensor.data<float>();
    int *keys = keys_tensor.data<int>();
    int *temp = temp_tensor.data<int>();
    int *out = out_tensor.data<int>();
    const float *mins = mins_tensor.data<float>();
    const float *grid_size = grid_size_tensor.data<float>();
    edgepc_kernel_launcher(b, n, npoint, points, keys, temp, out, mins, grid_size);
    return 1;
}


int multi_level_filtering_wrapper(int b, int n, int m, int total_nfilter, at::Tensor nfilter, at::Tensor darray, at::Tensor filter_matrix, at::Tensor bitmap, at::Tensor output) {

    int *nf = nfilter.data<int>();
    float *darr = darray.data<float>();
    const int *fm = filter_matrix.data<int>();
    int *bm = bitmap.data<int>();
    int *out = output.data<int>();

    multi_level_filtering_kernel_launcher(b, n, m, total_nfilter, nf, darr, fm, bm, out);
    return 1;
}

int QuickFPS_wrapper(int b, int n, int m, int kd_high, at::Tensor xyz_tensor, at::Tensor output_tensor, at::Tensor bucketIndex_tensor, at::Tensor bucketLength_tensor) {
    float *xyz = xyz_tensor.data<float>();
    float *output = output_tensor.data<float>();
    int *bucketIndex = bucketIndex_tensor.data<int>();
    int *bucketLength = bucketLength_tensor.data<int>();
    QuickFPS_launcher(b, n, m, kd_high, xyz, output, bucketIndex, bucketLength);
    return 1;
}
