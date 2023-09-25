/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/llama/LLaMAContextDecoder.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/LLaMAContextAttentionLayer.h"
#include "src/fastertransformer/utils/llama_utils.h"

namespace fastertransformer {

template<typename T>
void LLaMAContextDecoder<T>::initialize()
{
    self_attention_layer_ = new LLaMAContextAttentionLayer<T>(head_num_,
                                                              size_per_head_,
                                                              head_num_,
                                                              rotary_embedding_dim_,
                                                              stream_,
                                                              cublas_wrapper_,
                                                              allocator_,
                                                              is_free_buffer_after_forward_,
                                                              is_qk_buf_float_);

    ffn_layer_ = new SiluFfnLayer<T>(0,  // max_batch_size
                                     0,  // max_seq_len
                                     head_num_,
                                     size_per_head_,
                                     0,  // expert_num
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     false,
                                     true  // use_gated_activation = false
    );
}

template<typename T>
void LLaMAContextDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void LLaMAContextDecoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    self_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), false, false);
    padding_offset_ =
        reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false));
    cu_seqlens_ = reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false));
    is_allocate_buffer_ = true;
}

template<typename T>
void LLaMAContextDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&h_pinned_token_num_ptr_), true);
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&cu_seqlens_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool LLaMAContextDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool LLaMAContextDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool LLaMAContextDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int LLaMAContextDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
LLaMAContextDecoder<T>::LLaMAContextDecoder(size_t           head_num,
                                            size_t           size_per_head,
                                            size_t           inter_size,
                                            size_t           num_layer,
                                            size_t           rotary_embedding_dim,
                                            float            layernorm_eps,
                                            NcclParam        pipeline_para,
                                            cudaStream_t     stream,
                                            cublasMMWrapper* cublas_wrapper,
                                            IAllocator*      allocator,
                                            bool             is_free_buffer_after_forward,
                                            bool             is_qk_buf_float,
                                            AttentionType    attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    rotary_embedding_dim_(rotary_embedding_dim),
    layernorm_eps_(layernorm_eps),
    hidden_units_(head_num * size_per_head),
    pipeline_para_(pipeline_para),
    is_qk_buf_float_(is_qk_buf_float),
    attention_type_(attention_type)
{
    initialize();
}

template<typename T>
LLaMAContextDecoder<T>::LLaMAContextDecoder(LLaMAContextDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    rotary_embedding_dim_(decoder.rotary_embedding_dim_),
    layernorm_eps_(decoder.layernorm_eps_),
    hidden_units_(decoder.hidden_units_),
    pipeline_para_(decoder.pipeline_para_),
    is_qk_buf_float_(decoder.is_qk_buf_float_),
    attention_type_(decoder.attention_type_)
{
    initialize();
}

template<typename T>
LLaMAContextDecoder<T>::~LLaMAContextDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void LLaMAContextDecoder<T>::forward(std::vector<Tensor>*                            output_tensors,
                                     const std::vector<Tensor>*                      input_tensors,
                                     const std::vector<LLaMADecoderLayerWeight<T>*>* llama_decoder_layer_weight)
{
    std::unordered_map<std::string, Tensor> input_tensors_map{{"decoder_input", input_tensors->at(0)},
                                                              {"attention_mask", input_tensors->at(1)},
                                                              {"input_lengths", input_tensors->at(2)},
                                                              {"start_pos", input_tensors->at(3)}};
    std::unordered_map<std::string, Tensor> output_tensors_map{{"decoder_output", output_tensors->at(0)},
                                                               {"key_cache", output_tensors->at(1)},
                                                               {"value_cache", output_tensors->at(2)}};

    forward(&output_tensors_map, &input_tensors_map, llama_decoder_layer_weight);
}

template<typename T>
void LLaMAContextDecoder<T>::forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                                     const std::unordered_map<std::string, Tensor>*  input_tensors,
                                     const std::vector<LLaMADecoderLayerWeight<T>*>* llama_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_dimension],
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size]
    //      start_pos [1]

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_dimension],
    //      key_cache [num_layer, batch, max_seq_len, local_head_num, size_per_head]
    //      value_cache [num_layer, batch, max_seq_len, local_head_num, size_per_head]

    // To use layer/pipeline parallelism, we view the shape of 'batch_size' to 'ite * batch_size'.
    // For example, the shape of decoder_input becomes [ite, batch_size, seq_len, hidden_dimension] during
    // computing.

    FT_CHECK(input_tensors->size() == 4);
    FT_CHECK(output_tensors->size() == 3);

    const int      batch_size = input_tensors->at("decoder_input").shape[0];
    const int      seq_len    = input_tensors->at("decoder_input").shape[1];
    const int      start_pos  = input_tensors->at("start_pos").max<int>();
    const DataType data_type  = getTensorType<T>();
    allocateBuffer(batch_size, seq_len);

    T*       decoder_input  = input_tensors->at("decoder_input").getPtr<T>();
    T*       decoder_output = output_tensors->at("decoder_output").getPtr<T>();
    const T* attention_mask = input_tensors->at("attention_mask").getPtr<const T>();

    Tensor&             k_cache = output_tensors->at("key_cache");
    Tensor&             v_cache = output_tensors->at("value_cache");
    std::vector<size_t> self_k_cache_size;
    self_k_cache_size.push_back(batch_size);
    for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
        self_k_cache_size.push_back(*t);
    }
    std::vector<size_t> self_v_cache_size;
    self_v_cache_size.push_back(batch_size);
    for (auto t = v_cache.shape.begin() + 2; t != v_cache.shape.end(); ++t) {
        self_v_cache_size.push_back(*t);
    }

    AttentionType attention_type  = attention_type_;
    const bool    is_unpadded_mha = isUnPaddedMHA(attention_type);

    size_t h_token_num = batch_size * seq_len;
    if (is_unpadded_mha) {
        const int* input_lengths = input_tensors->at("input_lengths").getPtr<int>();
        invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                           &h_token_num,
                                           padding_offset_,
                                           cu_seqlens_,
                                           input_lengths,
                                           batch_size,
                                           seq_len,
                                           stream_);
        sync_check_cuda_error();
    }

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }

        if (l == 0 && is_unpadded_mha) {
            invokeRemovePadding(
                decoder_layer_output_, decoder_input, padding_offset_, h_token_num, hidden_units_, stream_);
            sync_check_cuda_error();
        }

        const bool is_final     = false;
        T*         layer_input  = decoder_layer_output_;
        T*         layer_output = decoder_layer_output_;
        if (!is_unpadded_mha) {
            if (l == 0) {
                layer_input = decoder_input;
            }
            if (l == num_layer_ - 1) {
                layer_output = decoder_output;
            }
        }

        if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            int data_size = h_token_num * hidden_units_;
            ftNcclRecv(layer_input, data_size, pipeline_para_.rank_ - 1, pipeline_para_, stream_);
            sync_check_cuda_error();
        }

        invokeGeneralLLaMALayerNorm(decoder_normed_input_,
                                    layer_input,
                                    llama_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                                    layernorm_eps_,
                                    h_token_num,
                                    hidden_units_,
                                    stream_);
        sync_check_cuda_error();

        if (false) {
            std::cout << l << "=================="
                      << "ATTN_NORM\n";
            print_tensor3(decoder_normed_input_, batch_size, seq_len, hidden_units_);
            std::cout << l << "=================="
                      << "ATTN_NORM\n";
            std::cout << std::flush;
        }

        TensorMap self_attention_input_tensors{
            {"input_query", Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, decoder_normed_input_}},
            {"attention_mask",
             Tensor{MEMORY_GPU,
                    data_type,
                    {(size_t)batch_size, (size_t)1, (size_t)seq_len, (size_t)(seq_len)},
                    attention_mask}},
            {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}},
            {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
            {"start_pos", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &start_pos}}};

        if (is_unpadded_mha) {
            self_attention_input_tensors.insert("padding_offset",
                                                Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num}, padding_offset_});
            self_attention_input_tensors.insert("cu_seqlens",
                                                Tensor{MEMORY_GPU, TYPE_INT32, {size_t(batch_size + 1)}, cu_seqlens_});
        }

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = k_cache.shape.begin() + 1; t != k_cache.shape.end(); ++t) {
            cache_offset *= *t;
        };

        TensorMap self_attention_output_tensors{
            {"hidden_features", Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, self_attn_output_}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache.getPtrWithOffset(cache_offset)}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache.getPtrWithOffset(cache_offset)}}};

        //std::cout << l << "==================" << "ATTENTION\n";
        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &llama_decoder_layer_weight->at(l)->self_attention_weights);
        //std::cout << l << "==================" << "ATTENTION\n";
        //std::cout << std::flush;

        if (false) {
            std::cout << l << "=================="
                      << "ATTENTION\n";
            print_tensor3(self_attn_output_, batch_size, seq_len, hidden_units_);
            std::cout << l << "=================="
                      << "ATTENTION\n";
            std::cout << std::flush;
        }

        invokeGeneralLLaMAAddBiasResidualPreLayerNorm(
            self_attn_output_,
            decoder_normed_input_,
            self_attn_output_,
            layer_input,
            llama_decoder_layer_weight->at(l)->post_attention_layernorm_weights.gamma,
            llama_decoder_layer_weight->at(l)->post_attention_layernorm_weights.beta,
            llama_decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.bias,
            layernorm_eps_,
            h_token_num,
            hidden_units_,
            stream_);
        sync_check_cuda_error();

        TensorMap ffn_input_tensors(
            {{"ffn_input",
              Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, decoder_normed_input_}}});
        TensorMap ffn_output_tensors(
            {{"ffn_output", Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, layer_output}}});
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &llama_decoder_layer_weight->at(l)->ffn_weights);

        invokeAddBiasResidual(layer_output,
                              self_attn_output_,
                              llama_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                              h_token_num,
                              hidden_units_,
                              stream_);

        sync_check_cuda_error();

        if (isLastLayerParallelId(l) && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {
            int data_size = h_token_num * hidden_units_;
            ftNcclSend(layer_output, data_size, pipeline_para_.rank_ + 1, pipeline_para_, stream_);
            sync_check_cuda_error();
        }

        if ((l == num_layer_ - 1) && is_unpadded_mha) {
            invokeRebuildPadding(decoder_output,
                                 decoder_layer_output_,
                                 padding_offset_,
                                 h_token_num,
                                 head_num_ * size_per_head_,
                                 stream_);
            sync_check_cuda_error();
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class LLaMAContextDecoder<float>;
template class LLaMAContextDecoder<half>;

}  // namespace fastertransformer
