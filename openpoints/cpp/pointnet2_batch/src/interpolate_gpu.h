#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H

#include <torch/serialize/tensor.h>
#include<vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
  at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);

void fused_three_nn_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor unknown_tensor, 
  at::Tensor known_tensor, at::Tensor ball_query_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);

void update_distance_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor xyz_tensor, at::Tensor sampled_xyz_tensor, at::Tensor ball_query_tensor, at::Tensor dist2_tensor);

void extract_ball_query_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor idx_tensor, at::Tensor ball_query_tensor,
	at::Tensor inv_tensor, at::Tensor pos_tensor);

void fused_convert_ball_query_wrapper_fast(int b, int n, int m, int nfilter, at::Tensor idx_tensor, at::Tensor ball_query_tensor,
	at::Tensor inv_tensor, at::Tensor pos_tensor);

void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown,
	const float *known, float *dist2, int *idx);

void fused_three_nn_kernel_launcher_fast(int b, int n, int m, int nfilter, const float *unknown,
	const float *known, const int *ball_query, float *dist2, int *idx);

void update_distance_kernel_launcher_fast(int b, int n, int m, int nfilter, const float *xyz, const float *sampled_xyz, const int *ball_query, float *dist2);

void extract_ball_query_launcher_fast(int b, int n, int m, int nfilter, const int *idx, const int *ball_query,
	int *inv, int *pos);

void fused_convert_ball_query_launcher_fast(int b, int n, int m, int nfilter, const int *idx, const int *ball_query,
	int *inv, int *pos);

void three_interpolate_wrapper_fast(int b, int c, int m, int n, at::Tensor points_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor);

void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n, 
    const float *points, const int *idx, const float *weight, float *out);


void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m, at::Tensor grad_out_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_points_tensor);

void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m, const float *grad_out, 
    const int *idx, const float *weight, float *grad_points);

#endif
