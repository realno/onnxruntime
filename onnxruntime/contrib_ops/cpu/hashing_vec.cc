// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hashing_vec.h"
#include "onnx/defs/schema.h"
#include "core/framework/tensor.h"

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
    // norm=None, alternate_sign=False, n_features=5000 and default value for other parameters
    // unless stated otherwise.
    Status s;

    auto X = context->Input<Tensor>(0);

    if (X->IsDataType<int32_t>()) {
        s = ComputeImpl<int32_t>(context);
    } else if (X->IsDataType<int64_t>()) {
        s = ComputeImpl<int64_t>(context);
    } else if (X->IsDataTypeString()) {
        s = ComputeImpl<std::string>(context);
    } else {
        s = Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                   "Invalid type of the input argument");
    }

    return s;
}

template <typename T>
Status HashingVectorizer::ComputeImpl(OpKernelContext* ctx) const {
    auto X = ctx->Input<Tensor>(0);
    auto& input_shape = X->Shape();
    const size_t total_items = input_shape.Size();

    size_t b_dim = 0;
    size_t B = 0;
    size_t C = 0;
    auto& input_dims = input_shape.GetDims();
    if (input_dims.empty()) {
        b_dim = 1;
        C = 1;
        assert(total_items == 1);
    } else if (input_dims.size() == 1) {
        b_dim = 1;
        C = input_dims[0];
    } else if (input_dims.size() == 2) {
        B = input_dims[0];
        C = input_dims[1];
        b_dim = B;
        if (B < 1) {
            return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                          "Input shape must have either [C] or [B,C] dimensions with B > 0.");
        }
    } else {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                      "Input shape must have either [C] or [B,C] dimensions with B > 0.");
    }

    float temp_buff[b_dim][n_features+1] = {};

    if (input_shape.Size() != 0) {
        // When this equals 0
        // TfidfVectorizer may receive an empty input when it follows a Tokenizer
        // (for example for a string containing only stopwords).
        // TfidfVectorizer returns a zero tensor of shape
        // {b_dim, output_size} when b_dim is the number of received observations
        // and output_size the is the maximum value in ngram_indexes attribute plus 1.

        assert((b_dim * C) == total_items);

        auto const input_data = X->template Data<T>();
        auto const end_data = input_data + total_items;
        //auto std::vector<Microsoft::Featurizer::Featurizers::SparseVectorEncoding<std::int64_t>::ValueEncoding> sparseVector;

        size_t row_num = 0;
        // Use ngram (1,1) for now
        auto n1_start = input_data;
        while (n1_start < end_data) {
            auto const n1_row_end = n1_start + C;
            while (n1_start < n1_row_end) {
                uint32_t hash;
                MurmurHash3_x86_32(n1_start, n_features, seed, &hash, true);
                temp_buff[row_num][hash]++;
                ++n1_start;
            }
            ++row_num;
            n1_start = n1_row_end;
        }
    }

    std::vector<int64_t> output_dims;
    if (B == 0) {
        output_dims.push_back(C);
        B = 1; // For use in the loops below
    } else {
        output_dims.push_back(B);
        output_dims.push_back(C);
    }

    TensorShape output_shape(output_dims);
    assert(frequences.size() == static_cast<size_t>(output_shape.Size()));

    auto Y = ctx->Output(0, output_shape);
    auto output_data = Y->MutableData<float>();

    uint64_t i = 0;
    while (i<b_dim*n_features) {
        *output_data++ = **(temp_buff+i);
        i++;
    }

    return Status::OK();
}

// Copy MurmurHash 32 signed to be compatible with sklearn
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