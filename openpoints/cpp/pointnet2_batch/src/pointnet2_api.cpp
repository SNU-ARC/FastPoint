#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "sampling_gpu.h"
#include "interpolate_gpu.h"
#include "fused_group_and_reduce_gpu.h"
#include "fused_group_and_reduce_pe_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper_fast, "ball_query_wrapper_fast");
    m.def("fused_ball_query_wrapper", &fused_ball_query_wrapper_fast, "fused_ball_query_wrapper_fast");

    m.def("group_points_wrapper", &group_points_wrapper_fast, "group_points_wrapper_fast");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast, "group_points_grad_wrapper_fast");

    m.def("fused_group_and_reduce_wrapper", &fused_group_and_reduce_wrapper_fast, "fused_group_and_reduce_wrapper_fast");
    m.def("fused_group_and_reduce_grad_wrapper", &fused_group_and_reduce_grad_wrapper_fast, "fused_group_and_reduce_grad_wrapper_fast");

    m.def("fused_group_and_reduce_pe_wrapper", &fused_group_and_reduce_pe_wrapper_fast, "fused_group_and_reduce_pe_wrapper_fast");
    m.def("fused_group_and_reduce_pe_grad_wrapper", &fused_group_and_reduce_pe_grad_wrapper_fast, "fused_group_and_reduce_pe_grad_wrapper_fast");

    m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast, "gather_points_grad_wrapper_fast");

    m.def("find_mps_wrapper", &find_mps_wrapper, "find_mps_wrapper");
    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
    m.def("edgepc_sampling_wrapper", &edgepc_sampling_wrapper, "edgepc_sampling_wrapper");
    m.def("multi_level_filtering_wrapper", &multi_level_filtering_wrapper, "multi_level_filtering_wrapper");
    m.def("QuickFPS_wrapper", &QuickFPS_wrapper, "QuickFPS_wrapper");
    
    m.def("three_nn_wrapper", &three_nn_wrapper_fast, "three_nn_wrapper_fast");
    m.def("fused_three_nn_wrapper", &fused_three_nn_wrapper_fast, "fused_three_nn_wrapper_fast");
    m.def("update_distance_wrapper", &update_distance_wrapper_fast, "update_distance_wrapper_fast");
    m.def("extract_ball_query_wrapper", &extract_ball_query_wrapper_fast, "extract_ball_query_wrapper_fast");
    m.def("fused_convert_ball_query_wrapper", &fused_convert_ball_query_wrapper_fast, "fused_convert_ball_query_wrapper_fast");
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast, "three_interpolate_wrapper_fast");
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast, "three_interpolate_grad_wrapper_fast");
}
