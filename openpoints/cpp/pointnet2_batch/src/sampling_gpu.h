#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>


int gather_points_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *points, const int *idx, float *out);


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor);

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *grad_out, const int *idx, float *grad_points);

int find_mps_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void find_mps_kernel_launcher(int b, int n, int m, 
    const float *dataset, float *temp, float *idxs);

int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void furthest_point_sampling_kernel_launcher(int b, int n, int m, 
    const float *dataset, float *temp, int *idxs);

int edgepc_sampling_wrapper(int b, int n, int npoint, at::Tensor points_tensor, at::Tensor keys_tensor, at::Tensor temp_tensor, at::Tensor out_tensor, at::Tensor mins_tensor, at::Tensor grid_size_tensor);

void edgepc_kernel_launcher(const int b, const int n, const int npoint, float *points, int *keys, int *temp, int *out, const float *mins, const float *grid_size);

int multi_level_filtering_wrapper(int b, int n, int m, int total_nfilter, at::Tensor nfilter, at::Tensor darray, at::Tensor filter_matrix, at::Tensor bitmap, at::Tensor output);

void multi_level_filtering_kernel_launcher(int b, int n, int m, int total_nfilter, int *nfilter, float *darray, const int *filter_matrix, int *bitmap, int *output);

int QuickFPS_wrapper(int b, int n, int m, int kd_high, at::Tensor xyz_tensor, at::Tensor output_tensor, at::Tensor bucketIndex_tensor, at::Tensor bucketLength_tensor);

void QuickFPS_launcher(int b, int n, int sample_number, int kd_high, float *xyz, float *output, int *bucketIndex, int *bucketLength);

#endif
