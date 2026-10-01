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

#include <torch/extension.h>

void pybind_droid_net_ext(py::module &m);
void pybind_grounding_dino_ext(py::module &m);
void pybind_utils_ext(py::module &m);
void pybind_slam_ext(py::module &m);
void pybind_scatter_ext(py::module &m);
void pybind_lietorch_ext(py::module &m);
void pybind_corr_ext(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::module m_droid_net = m.def_submodule("droid_net_ext");
    pybind_droid_net_ext(m_droid_net);

    py::module m_gdino = m.def_submodule("grounding_dino_ext");
    pybind_grounding_dino_ext(m_gdino);

    py::module m_utils = m.def_submodule("utils_ext");
    pybind_utils_ext(m_utils);

    py::module m_slam = m.def_submodule("slam_ext");
    pybind_slam_ext(m_slam);

    py::module m_scatter = m.def_submodule("scatter_ext");
    pybind_scatter_ext(m_scatter);

    py::module m_lietorch = m.def_submodule("lietorch_ext");
    pybind_lietorch_ext(m_lietorch);

    py::module m_corr = m.def_submodule("corr_ext");
    pybind_corr_ext(m_corr);
}
