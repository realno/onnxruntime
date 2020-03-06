// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "contrib_ops/cpu/murmur_hash3.h"

// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER)

#define FORCE_INLINE __forceinline

#include <stdlib.h>

#define xROTL32(x, y) _rotl(x, y)
#define xROTL64(x, y) _rotl64(x, y)

#define xBIG_CONSTANT(x) (x)

// Other compilers

#else  // defined(_MSC_VER)

#if defined(GNUC) && ((GNUC > 4) || (GNUC == 4 && GNUC_MINOR >= 4))

// gcc version >= 4.4 4.1 = RHEL 5, 4.4 = RHEL 6.
// Don't inline for RHEL 5 gcc which is 4.1
#define FORCE_INLINE attribute((always_inline))

#else

#define FORCE_INLINE

#endif

inline uint32_t xrotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

inline uint64_t xrotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define xROTL32(x, y) xrotl32(x, y)
#define xROTL64(x, y) xrotl64(x, y)

#define xBIG_CONSTANT(x) (x##LLU)

#endif  // !defined(_MSC_VER)

//-----------------------------------------------------------------------------
// If your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

FORCE_INLINE uint32_t xgetblock(const uint32_t* p, int i) {
  return p[i];
}

FORCE_INLINE uint64_t xgetblock(const uint64_t* p, int i) {
  return p[i];
}

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

FORCE_INLINE uint32_t xfmix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//----------

FORCE_INLINE uint64_t xfmix(uint64_t k) {
  k ^= k >> 33;
  k *= xBIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= xBIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

namespace onnxruntime {
namespace contrib {

class HashingVectorizer final : public OpKernel {
 public:
  explicit HashingVectorizer(const OpKernelInfo& info);
  ~HashingVectorizer() = default;
  Status Compute(OpKernelContext* context) const override;
 private:
    template <typename T> Status ComputeImpl(OpKernelContext* ctx, int64_t output_size) const;
  int64_t n_features = 1 << 20;
  std::string alternate_sign = "False";
  std::string norm = "None";
  void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out, bool is_positive) const;

    Status ComputeImpl(OpKernelContext *ctx) const;
};
}  // namespace contrib
}  // namespace onnxruntime
