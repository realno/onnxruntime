// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hashing_vec.h"
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

HashingVectorizer::HashingVectorizer(const OpKernelInfo& info) : OpKernel(info) {
    Status status = info.GetAttr("n_features", &n_features);
    status = info.GetAttr("norm", &norm);
    if (status.IsOK()) {
        ORT_ENFORCE(norm == "None", "only None is supported for norm");
    }
    status = info.GetAttr("alternate_sign", &alternate_sign);
    if (status.IsOK()) {
        ORT_ENFORCE(alternate_sign == "False", "only False is supported for alternate_sign");
    }
}

Status HashingVectorizer::Compute(OpKernelContext* context) const {
    // Very simple implementation assume ngram = (1,1), utf-8 encoding, all lower case 
    // norm=None, alternate_sign=False and default value for other parameters
    // unless stated otherwise.

    context->Input<Tensor>(0); 
    // auto X = context->Input<Tensor>(0);
    // auto& dims = X->Shape();
    // auto Y = context->Output(0, dims);
    //auto X_Data = (X->template Data<T>());
    //auto Y_Data = (Y->template MutableData<T>());

    // for (int64_t i = 0, sz = dims.Size(); i < sz; ++i) {
    //   *Y_Data++ = *X_Data++;
    // }

    return Status::OK();
  }

// Cpoied MurmurHash 32 signed to be compatible with sklearn
void HashingVectorizer::MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out, bool is_positive) const {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(key);
  const int nblocks = len / 4;
  uint32_t h1 = seed;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  //----------
  // body
  const uint32_t* blocks = reinterpret_cast<const uint32_t*>(data + static_cast<int64_t>(nblocks) * 4);

  for (int i = -nblocks; i; i++) {
    uint32_t k1 = xgetblock(blocks, i);

    k1 *= c1;
    k1 = xROTL32(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = xROTL32(h1, 13);
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
      k1 = xROTL32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  };

  //----------
  // finalization
  h1 ^= len;

  h1 = xfmix(h1);

  if (is_positive) {
    *(uint32_t*)out = h1;
  } else {
    *(int32_t*)out = h1;
  }
}

} // namespace contrib
}  // namespace onnxruntime