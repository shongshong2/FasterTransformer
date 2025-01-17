/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/th_op/llama/LLaMA.h"

namespace th = torch;
namespace torch_ext {

LLaMA::LLaMA(const int64_t            num_heads,
             const int64_t            size_per_head,
             const int64_t            inter_size,
             const int64_t            num_layers,
             const int64_t            vocab_size,
             const int64_t            rotary_embedding_dim,
             const int64_t            random_seed,
             const int64_t            max_seq_len,
             const int64_t            rank,
             const int64_t            world_size,
             const vector<th::Tensor> weights):
    vocab_size_(vocab_size), st_(weights[0].scalar_type())
{
    for (auto t : weights) {
        CHECK_INPUT(t, st_);
    }

    switch (st_) {
        case at::ScalarType::Float:
            ftllama = new FTLLaMA<float>((size_t)num_heads,
                                         (size_t)size_per_head,
                                         (size_t)inter_size,
                                         (size_t)num_layers,
                                         (size_t)vocab_size,
                                         (size_t)rotary_embedding_dim,
                                         (size_t)random_seed,
                                         (size_t)max_seq_len,
                                         (size_t)rank,
                                         (size_t)world_size,
                                         weights);
            break;
        case at::ScalarType::Half:
            ftllama = new FTLLaMA<half>((size_t)num_heads,
                                        (size_t)size_per_head,
                                        (size_t)inter_size,
                                        (size_t)num_layers,
                                        (size_t)vocab_size,
                                        (size_t)rotary_embedding_dim,
                                        (size_t)random_seed,
                                        (size_t)max_seq_len,
                                        (size_t)rank,
                                        (size_t)world_size,
                                        weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

LLaMA::~LLaMA()
{
    delete ftllama;
}

std::vector<th::Tensor> LLaMA::forward(th::Tensor&   hidden_vector,
                                       th::Tensor&   cum_probs,
                                       th::Tensor&   input_ids,
                                       th::Tensor&   input_lengths,
                                       th::Tensor&   target_ids,
                                       th::Tensor&   context_lengths,
                                       const int64_t seq_len,
                                       const int64_t attn_len,
                                       const int64_t is_context)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");

    ftllama->forward(hidden_vector,
                     cum_probs,
                     input_ids,
                     input_lengths,
                     target_ids,
                     context_lengths,
                     seq_len,
                     attn_len,
                     is_context);
    return std::vector<th::Tensor>{hidden_vector, cum_probs};
}

}  // namespace torch_ext

static auto fasterTransformerGptTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::LLaMA>("FasterTransformerLLaMA")
#else
    torch::jit::class_<torch_ext::LLaMA>("FasterTransformer", "LLaMA")
#endif
        .def(torch::jit::init<int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              std::vector<th::Tensor>>())
        .def("forward", &torch_ext::LLaMA::forward);
