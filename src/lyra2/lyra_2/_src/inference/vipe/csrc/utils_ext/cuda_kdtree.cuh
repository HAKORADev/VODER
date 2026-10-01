/***********************************************************************
 * The code in this file is modified from the original code in the
 * https://github.com/jlblancoc/nanoflann repository.
 *
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2025  Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_CUDA_KD_TREE_BUILDER_H_
#define FLANN_CUDA_KD_TREE_BUILDER_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include "math_util.h"
#include <cstdlib>
#include <map>

namespace tinyflann {

// Distance types.
struct CudaL1;
struct CudaL2;

// Parameters
struct KDTreeCuda3dIndexParams {
    int leaf_max_size = 64;
};

struct SearchParams {
    SearchParams(int checks_ = 32, float eps_ = 0.0, bool sorted_ = true)
        : checks(checks_), eps(eps_), sorted(sorted_) {
        max_neighbors = -1;
        use_heap = true;
    }

    // how many leafs to visit when searching for neighbours (-1 for unlimited)
    int checks;
    // search for eps-approximate neighbours (default: 0)
    float eps;
    // only for radius search, require neighbours sorted by distance (default: true)
    bool sorted;
    // maximum number of neighbors radius search should return (-1 for unlimited)
    int max_neighbors;
    // use a heap to manage the result set (default: FLANN_Undefined)
    bool use_heap;
};

template <typename Distance>
class KDTreeCuda3dIndex {
   public:
    int visited_leafs;
    KDTreeCuda3dIndex(const float* input_data, size_t input_count,
                      const KDTreeCuda3dIndexParams& params = KDTreeCuda3dIndexParams())
        : dataset_(input_data), leaf_count_(0), visited_leafs(0), node_count_(0), current_node_count_(0) {
        size_ = input_count;
        leaf_max_size_ = params.leaf_max_size;
        gpu_helper_ = 0;
    }

    /**
     * Standard destructor
     */
    ~KDTreeCuda3dIndex() { clearGpuBuffers(); }

    /**
     * Builds the index
     */
    void buildIndex() {
        leaf_count_ = 0;
        node_count_ = 0;
        uploadTreeToGpu();
    }

    /**
     * queries: (N, p) flattened float cuda array where only first 3 elements are used.
     * n_query: N
     * n_query_stride: p
     * indices: (N, knn) int cuda array
     * dists: (N, knn) float cuda array
     */
    void knnSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, size_t knn,
                   const SearchParams& params = SearchParams()) const;
    int radiusSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, float radius,
                     const SearchParams& params = SearchParams()) const;

   private:
    void uploadTreeToGpu();
    void clearGpuBuffers();

   private:
    struct GpuHelper;
    GpuHelper* gpu_helper_;

    const float* dataset_;
    int leaf_max_size_;
    int leaf_count_;
    int node_count_;
    //! used by convertTreeToGpuFormat
    int current_node_count_;
    size_t size_;

};  // class KDTreeCuda3dIndex

}  // namespace tinyflann

#endif