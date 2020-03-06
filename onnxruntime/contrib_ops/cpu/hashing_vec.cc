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
    // norm=None, alternate_sign=False and default value for other parameters
    // unless stated otherwise.
    Status s;

    auto X = context->Input<Tensor>(0);
    auto& dims = X->Shape();
    auto Y = context->Output(0, dims);
    int64_t  output_size = dims.Size();

    if (X->IsDataType<int32_t>()) {
        s = ComputeImpl<int32_t>(context, output_size);
    } else if (X->IsDataType<int64_t>()) {
        s = ComputeImpl<int64_t>(context, output_size);
    } else if (X->IsDataTypeString()) {
        s = ComputeImpl<std::string>(context, output_size);
    } else {
        s = Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                   "Invalid type of the input argument");
    }

    return s;
    // auto X = context->Input<Tensor>(0);
    // auto& dims = X->Shape();
    // auto Y = context->Output(0, dims);
    //auto X_Data = (X->template Data<T>());
    //auto Y_Data = (Y->template MutableData<T>());

    // for (int64_t i = 0, sz = dims.Size(); i < sz; ++i) {
    //   *Y_Data++ = *X_Data++;
    // }

    // return Status::OK();
}

void HashingVectorizer::OutputResult(OpKernelContext* ctx, size_t B, const std::vector<uint32_t>& frequences) const {
    const Impl& impl = *impl_;
    std::vector<int64_t> output_dims;
    if (B == 0) {
        output_dims.push_back(impl.output_size_);
        B = 1; // For use in the loops below
    } else {
        output_dims.push_back(B);
        output_dims.push_back(impl.output_size_);
    }

    const auto row_size = impl.output_size_;

    TensorShape output_shape(output_dims);
    assert(frequences.size() == static_cast<size_t>(output_shape.Size()));

    auto Y = ctx->Output(0, output_shape);
    auto output_data = Y->MutableData<float>();
    const auto& w = impl.weights_;
    switch (impl.weighting_criteria_) {
        case kTF: {
            for (auto f : frequences) {
                *output_data++ = static_cast<float>(f);
            }
        } break;
        case kIDF: {
            if (!w.empty()) {
                const auto* freqs = frequences.data();
                for (size_t batch = 0; batch < B; ++batch) {
                    for (size_t i = 0; i < row_size; ++i) {
                        *output_data++ = (*freqs++ > 0) ? w[i] : 0;
                    }
                }
            } else {
                for (auto f : frequences) {
                    *output_data++ = (f > 0) ? 1.0f : 0;
                }
            }
        } break;
        case kTFIDF: {
            if (!w.empty()) {
                const auto* freqs = frequences.data();
                for (size_t batch = 0; batch < B; ++batch) {
                    for (size_t i = 0; i < row_size; ++i) {
                        *output_data++ = *freqs++ * w[i];
                    }
                }
            } else {
                for (auto f : frequences) {
                    *output_data++ = static_cast<float>(f);
                }
            }
        } break;
        case kNone:  // fall-through
        default:
            assert(false);
    }
}

template <typename T>
Status HashingVectorizer::ComputeImpl(OpKernelContext* ctx, int64_t  output_size) const {
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

    // Frequency holder allocate [B..output_size_]
    // and init all to zero
    std::vector<uint32_t> counts;
    counts.resize(b_dim * output_size, 0);

    if (input_shape.Size() == 0) {
        // TfidfVectorizer may receive an empty input when it follows a Tokenizer
        // (for example for a string containing only stopwords).
        // TfidfVectorizer returns a zero tensor of shape
        // {b_dim, output_size} when b_dim is the number of received observations
        // and output_size the is the maximum value in ngram_indexes attribute plus 1.
        OutputResult(ctx, B, counts);
        return Status::OK();
    }

    assert((b_dim * C) == total_items);

    auto const input_data = X->template Data<T>();
    auto const end_data = input_data + total_items;

    // Treat 1-grams in a special way
    if (start_ngram_size == 1) {
        size_t row_num = 0;
        auto ngram_start = input_data;
        while (ngram_start < end_data) {
            auto const ngram_row_end = ngram_start + C;
            while (ngram_start < ngram_row_end) {
                sample.Clear();
                sample.AddItem(*ngram_start);
                auto hit = impl.PoolFind<T>(sample);
                if (hit != set_end) {
                    // record frequency
                    auto ngram_id = hit->Id();
                    impl.IncrementCount(ngram_id, row_num, frequencies);
                }
                ++ngram_start;
            }
            ++row_num;
            ngram_start = ngram_row_end;
        }
        if (++start_ngram_size > max_gram_length) {
            OutputResult(ctx, B, frequencies);
            return Status::OK();
        }
    }

    for (auto skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
        auto ngram_start = input_data;
        size_t row_num = 0;
        while (ngram_start < end_data) {
            assert((B == 0) || (row_num < B));
            auto const ngram_row_end = ngram_start + C;
            assert(ngram_row_end <= end_data);
            while (ngram_start < ngram_row_end) {
                // Check if any n-gram size in [start_ngram_size..max_gram_length] range
                // fit before the end of the row so we do not waste time adding [1..start_ngram_size)
                // At least items of start_ngram_size should fit
                // last row should match end_data
                auto at_least_this = ngram_start + skip_distance * (start_ngram_size - 1);
                if (at_least_this >= ngram_row_end) {
                    break;
                }
                sample.Clear();
                auto ngram_item = ngram_start;
                for (auto ngram_size = 1;
                     ngram_size <= max_gram_length &&
                     ngram_item < ngram_row_end;
                     ++ngram_size, ngram_item += skip_distance) {
                    sample.AddItem(*ngram_item);

                    // Do not test anything before start_ngram_size
                    if (ngram_size >= start_ngram_size) {
                        auto hit = impl.PoolFind<T>(sample);
                        if (hit != set_end) {
                            // record frequency
                            auto ngram_id = hit->Id();
                            impl.IncrementCount(ngram_id, row_num, frequencies);
                        }
                    }
                }
                // Sliding window shift
                ++ngram_start;
            }
            // Next row
            ngram_start = ngram_row_end;
            ++row_num;
        }
    }
    OutputResult(ctx, B, frequencies);
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