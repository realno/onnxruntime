// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sample.h"
#include "onnx/defs/schema.h"

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    HashingVectorizer,
    1,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<std::string>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::HashingVectorizer);
} // namespace contrib
}  // namespace onnxruntime

HashingVectorizer::HashingVectorizer(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("n_features", &n_features);
}

Status HashingVectorizer::Compute(OpKernelContext* context) const override {
    // Very simple implementation assume ngram = (1,1), utf-8 encoding, all lower case and default value for other parameters
    // unless stated otherwise. 
    auto X = context->Input<Tensor>(0);
    auto& dims = X->Shape();
    auto Y = context->Output(0, dims);
    auto X_Data = (X->template Data<T>());
    auto Y_Data = (Y->template MutableData<T>());

    for (int64_t i = 0, sz = dims.Size(); i < sz; ++i) {
      *Y_Data++ = *X_Data++;
    }

    return Status::OK();
  }

// Cpoied MurmurHash 32 signed to be compatible with sklearn
void HashingVectorizer::MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out) const {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(key);
  const int nblocks = len / 4;
  uint32_t h1 = seed;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  //----------
  // body
  const uint32_t* blocks = reinterpret_cast<const uint32_t*>(data + static_cast<int64_t>(nblocks) * 4);

  for (int i = -nblocks; i; i++) {
    uint32_t k1 = getblock(blocks, i);

    k1 *= c1;
    k1 = ROTL32(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = ROTL32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }

  //----------
  // tail
  const uint8_t* tail = reinterpret_cast<const uint8_t*>(data + static_cast<int64_t>(nblocks) * 4);

  uint32_t k1 = 0;

  switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;  // Fallthrough.
    case 2:
      k1 ^= tail[1] << 8;  // Fallthrough.
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = ROTL32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  };

  //----------
  // finalization
  h1 ^= len;

  h1 = fmix(h1);

  if (is_positive_) {
    *(uint32_t*)out = h1;
  } else {
    *(int32_t*)out = h1;
  }
}