// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include "ms_deform_attn_cuda.h"

namespace groundingdino {

at::Tensor ms_deform_attn_forward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                  const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
                                  const at::Tensor &attn_weight, const int im2col_step) {
    if (value.is_cuda()) {
        return ms_deform_attn_cuda_forward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight,
                                           im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> ms_deform_attn_backward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                                const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
                                                const at::Tensor &attn_weight, const at::Tensor &grad_output,
                                                const int im2col_step) {
    if (value.is_cuda()) {
        return ms_deform_attn_cuda_backward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight,
                                            grad_output, im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}

}  // namespace groundingdino

void pybind_grounding_dino_ext(py::module &m) {
    m.def("ms_deform_attn_forward", &groundingdino::ms_deform_attn_forward, "ms_deform_attn_forward");
    m.def("ms_deform_attn_backward", &groundingdino::ms_deform_attn_backward, "ms_deform_attn_backward");
}