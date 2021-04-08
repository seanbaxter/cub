/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Properties of a given CUDA device and the corresponding PTX bundle
 */

#pragma once

#include "util_device.cuh"
#include "util_namespace.cuh"

#include <type_traits>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub
{

namespace detail
{

template <typename...>
struct type_list;

/**
 * Helper class to tag a type T with a PtxArch, suitable for use with
 * ptx_arch_lookup.
 *
 * \tparam PtxArch 3-digit PTX architecture version (e.g. 750 for Turing)
 * \tparam T The type to tag with `::ptx_arch = PtxArch`.
 */
template <int PtxArch, typename T>
struct ptx_tag : T
{
  static constexpr int ptx_arch = PtxArch;
};

/** Placeholder for invalid ptx_arch_lookup results. */
struct no_ptx_type
{
  static constexpr int ptx_arch = 0;
};

/**
 * Compile-time lookup of types based on PTX version.
 *
 * Given:
 *
 * - `TargetPtxArch`, a 3-digit PTX architecture version, e.g. 750 for Turing.
 * - `PtxTypes`, a type_list of types with `::ptx_arch` members defined to
 *   3-digit PTX architecture versions.
 *
 * Defines `type` to the `PtxType` with the highest `ptx_arch` that does not
 * exceed `TargetPtx`.
 *
 * Each `PtxType` must have a unique `ptx_arch` and at least one
 * `PtxType::ptx_arch` must be less-than-or-equal-to TargetPtxArch, otherwise
 * the behavior is undefined.
 *
 * /sa ptx_tag
 * /sa type_list
 * /sa ptx_arch_lookup_t
 */
template <int TargetPtxArch, typename PtxTypes>
struct ptx_arch_lookup;

template <int TargetPtxArch, typename PtxType, typename... PtxTypeTail>
struct ptx_arch_lookup<TargetPtxArch, type_list<PtxType, PtxTypeTail...>>
{
private:
  using this_type =
    typename std::conditional<(PtxType::ptx_arch <= TargetPtxArch),
                              PtxType,
                              no_ptx_type>::type;

  using next_lookup = ptx_arch_lookup<TargetPtxArch, type_list<PtxTypeTail...>>;
  using next_type   = typename next_lookup::type;

public:
  using type =
    typename std::conditional<(this_type::ptx_arch > next_type::ptx_arch),
                              this_type,
                              next_type>::type;
};

template <int TargetPtxArch>
struct ptx_arch_lookup<TargetPtxArch, type_list<>>
{
  using type = no_ptx_type;
};

template <int TargetPtxArch, typename PtxTypes>
using ptx_arch_lookup_t = typename ptx_arch_lookup<TargetPtxArch, PtxTypes>::type;

// TODO there's gotta be a better way to silence unused parameter pack warnings.
template <typename T, typename...Ts>
__host__ __device__
void mark_as_used(T& t, Ts&...ts)
{
  (void)t;
  mark_as_used(ts...);
}

__host__ __device__
void mark_as_used() {}

/**
 * Host-side implementation of ptx_dispatch. Target ptx_arch is provided at
 * runtime.
 */
template <typename PtxTypes, template <typename> class Functor>
struct ptx_dispatch_host_impl;

template <template <typename> class Functor,
          typename PtxHeadType,
          typename ...PtxTailTypes>
struct ptx_dispatch_host_impl<type_list<PtxHeadType, PtxTailTypes...>, Functor>
{
  template <typename... Args>
  __host__
  static void exec(int target_ptx_arch, Args &&...args)
  {
    if (target_ptx_arch < PtxHeadType::ptx_arch)
    {
      using next = ptx_dispatch_host_impl<type_list<PtxTailTypes...>, Functor>;
      next::exec(target_ptx_arch, std::forward<Args>(args)...);
    }
    else
    {
      Functor<PtxHeadType>{}(std::forward<Args>(args)...);
    }
  }
};

template <template <typename> class Functor>
struct ptx_dispatch_host_impl<type_list<>, Functor>
{
  template <typename... Args>
  __host__
  static void exec(int /* target_ptx_arch */, Args &&...args)
  {
    // TODO throw? Didn't find a matching arch.
    mark_as_used(args...);
  }
};

/**
 * Executes `Functor<PtxType>{}(args...)`, after finding the best match in
 * `PtxTypes` for `TargetPtxArch`.
 */
template <int TargetPtxArch, typename PtxTypes, template <typename> class Functor>
struct ptx_dispatch_device_impl
{

#pragma nv_exec_check_disable
  template <typename... Args>
  __device__
  static void exec(Args &&...args)
  {
    using ptx_type = ptx_arch_lookup_t<TargetPtxArch, PtxTypes>;
    Functor<ptx_type>{}(std::forward<Args>(args)...);
  }
};

template <typename PtxTypes, template <typename> class Functor>
struct ptx_dispatch
{
  template <typename... Args>
  __host__ __device__
  static void exec(Args &&...args)
  {
    mark_as_used(args...);

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_80,
      (ptx_dispatch_device_impl<800, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_75,
      (ptx_dispatch_device_impl<750, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_72,
      (ptx_dispatch_device_impl<720, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_70,
      (ptx_dispatch_device_impl<700, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_62,
      (ptx_dispatch_device_impl<620, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_61,
      (ptx_dispatch_device_impl<610, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_60,
      (ptx_dispatch_device_impl<600, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_53,
      (ptx_dispatch_device_impl<530, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_52,
      (ptx_dispatch_device_impl<520, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_50,
      (ptx_dispatch_device_impl<500, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_37,
      (ptx_dispatch_device_impl<370, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_PROVIDES_SM_35,
      (ptx_dispatch_device_impl<350, PtxTypes, Functor>::exec(std::forward<Args>(args)...);),
      NV_IS_HOST,
      (exec_host(std::forward<Args>(args)...);)
    );
  }

private:
  template <typename... Args>
  __host__
  static void exec_host(Args &&...args)
  {
    mark_as_used(args...);

    int ptx_version = 0;
    if (CubDebug(cub::PtxVersion(ptx_version)))
    { // TODO throw?
      return;
    }

    using dispatcher_t = ptx_dispatch_host_impl<PtxTypes, Functor>;
    dispatcher_t::exec(ptx_version, std::forward<Args>(args)...);
  }

  template <typename... Args>
  __device__
  static void exec_device(Args &&...args)
  {

  }
};

} // namespace detail
} // namespace cub
