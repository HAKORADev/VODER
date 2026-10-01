/**
 * This file includes code originally from the Pytorch Scatter repository:
 * https://github.com/rusty1s/pytorch_scatter
 * Licensed under the MIT License. See THIRD_PARTY_LICENSES.md for details.
 */

#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, std::optional<torch::Tensor>> scatter_cuda(torch::Tensor src, torch::Tensor index,
                                                                     int64_t dim,
                                                                     std::optional<torch::Tensor> optional_out,
                                                                     std::optional<int64_t> dim_size,
                                                                     std::string reduce);
