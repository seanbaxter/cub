/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/util_ptx_dispatch.cuh>

#include "test_util.h"

struct no_type {};

using TestPolicy350 = cub::detail::ptx_tag<350, no_type>;
using TestPolicy600 = cub::detail::ptx_tag<600, no_type>;
using TestPolicy800 = cub::detail::ptx_tag<800, no_type>;

void test_ptx_arch_lookup()
{
  using cub::detail::type_list;
  using cub::detail::ptx_arch_lookup_t;

  {
    using PtxTypes = type_list<TestPolicy350>;
    using Lookup350 = ptx_arch_lookup_t<350, PtxTypes>;
    using Lookup800 = ptx_arch_lookup_t<800, PtxTypes>;
    AssertEquals((std::is_same<Lookup350, TestPolicy350>::value), true);
    AssertEquals((std::is_same<Lookup800, TestPolicy350>::value), true);
  }

  {
    using PtxTypes = type_list<TestPolicy350, TestPolicy600, TestPolicy800>;
    using Lookup350 = ptx_arch_lookup_t<350, PtxTypes>;
    using Lookup520 = ptx_arch_lookup_t<520, PtxTypes>;
    using Lookup600 = ptx_arch_lookup_t<600, PtxTypes>;
    using Lookup700 = ptx_arch_lookup_t<700, PtxTypes>;
    using Lookup800 = ptx_arch_lookup_t<800, PtxTypes>;
    using Lookup860 = ptx_arch_lookup_t<860, PtxTypes>;
    AssertEquals((std::is_same<Lookup350, TestPolicy350>::value), true);
    AssertEquals((std::is_same<Lookup520, TestPolicy350>::value), true);
    AssertEquals((std::is_same<Lookup600, TestPolicy600>::value), true);
    AssertEquals((std::is_same<Lookup700, TestPolicy600>::value), true);
    AssertEquals((std::is_same<Lookup800, TestPolicy800>::value), true);
    AssertEquals((std::is_same<Lookup860, TestPolicy800>::value), true);
  }

}

int main()
{
  test_ptx_arch_lookup();
}
