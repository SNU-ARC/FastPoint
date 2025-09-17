/*
batch version of point interpolation, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interpolate_gpu.h"




void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data<float>();
    const float *known = known_tensor.data<float>();
    float *dist2 = dist2_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx);
}

void fused_three_nn_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor unknown_tensor, 
    at::Tensor known_tensor, at::Tensor ball_query_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data<float>();
    const float *known = known_tensor.data<float>();
    const int *ball_query = ball_query_tensor.data<int>();
    float *dist2 = dist2_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    fused_three_nn_kernel_launcher_fast(b, n, m, nfilter, unknown, known, ball_query, dist2
    , idx);
}

void update_distance_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor xyz_tensor, at::Tensor sampled_xyz_tensor, at::Tensor ball_query_tensor, at::Tensor dist2_tensor) {
    const float *xyz = xyz_tensor.data<float>();
    const float *sampled_xyz = sampled_xyz_tensor.data<float>();
    const int *ball_query = ball_query_tensor.data<int>();
    float *dist2 = dist2_tensor.data<float>();

    update_distance_kernel_launcher_fast(b, n, m, nfilter, xyz, sampled_xyz, ball_query, dist2);
}

void extract_ball_query_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor idx_tensor, at::Tensor ball_query_tensor,
	at::Tensor inv_tensor, at::Tensor pos_tensor) {
    const int *idx = idx_tensor.data<int>();
    const int *ball_query = ball_query_tensor.data<int>();
    int *inv = inv_tensor.data<int>();
    int *pos = pos_tensor.data<int>();

    extract_ball_query_launcher_fast(b, n, m, nfilter, idx, ball_query, inv, pos);
}

void fused_convert_ball_query_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor idx_tensor, at::Tensor ball_query_tensor,
	at::Tensor inv_tensor, at::Tensor pos_tensor) {
    const int *idx = idx_tensor.data<int>();
    const int *ball_query = ball_query_tensor.data<int>();
    int *inv = inv_tensor.data<int>();
    int *pos = pos_tensor.data<int>();

    fused_convert_ball_query_launcher_fast(b, n, m, nfilter, idx, ball_query, inv, pos);
}

void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                         at::Tensor points_tensor,
                         at::Tensor idx_tensor,
                         at::Tensor weight_tensor,
                         at::Tensor out_tensor) {

    const float *points = points_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *out = out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out);
}

void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                            at::Tensor grad_out_tensor,
                            at::Tensor idx_tensor,
                            at::Tensor weight_tensor,
                            at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *grad_points = grad_points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points);
}
