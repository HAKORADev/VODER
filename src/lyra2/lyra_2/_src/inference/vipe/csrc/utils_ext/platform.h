/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CODE_ __host__ __device__  // for CUDA device code
#else
#define _CPU_AND_GPU_CODE_
#endif

#if defined(__CUDACC__)
#define _CPU_AND_GPU_CODE_TEMPLATE_ __host__ __device__  // for CUDA device code
#else
#define _CPU_AND_GPU_CODE_TEMPLATE_
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CONSTANT_ __constant__  // for CUDA device code
#else
#define _CPU_AND_GPU_CONSTANT_ const
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#include <THC/THCAtomics.cuh>

template <typename ScalarT>
static inline __device__ void atomicAdd(ScalarT* addr, ScalarT value) {
    gpuAtomicAddNoReturn(addr, value);
}
#else
template <typename ScalarT>
static inline void atomicAdd(ScalarT* addr, ScalarT value) {
    *addr += value;
}
#endif
