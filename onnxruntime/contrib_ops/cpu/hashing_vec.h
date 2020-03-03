// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class HashingVectorizer final : public OpKernel {
 public:
  explicit HashingVectorizer(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;
 private:
  int64_t n_features = 1 << 20;
  void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out) const;
};
}  // namespace contrib
}  // namespace onnxruntime
