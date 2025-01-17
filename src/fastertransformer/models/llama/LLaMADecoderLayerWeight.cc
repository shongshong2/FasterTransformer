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

#include "src/fastertransformer/models/llama/LLaMADecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
LLaMADecoderLayerWeight<T>::LLaMADecoderLayerWeight(const int hidden_units, const int inter_size):
    hidden_units_(hidden_units), inter_size_(inter_size)
{
    mallocWeights();
    setWeightPtr();
}

template<typename T>
LLaMADecoderLayerWeight<T>::~LLaMADecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 14; i++) {
            if (i != attention_dense_bias_weight_id) {
                cudaFree(weights_ptr[i]);
            }
        }

        pre_layernorm_weights.beta                            = nullptr;
        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;
        post_attention_layernorm_weights.beta                 = nullptr;
        post_attention_layernorm_weights.gamma                = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias   = nullptr;
        ffn_weights.output_weight.kernel       = nullptr;
        ffn_weights.output_weight.bias         = nullptr;
        is_maintain_buffer                     = false;
    }
}

template<typename T>
LLaMADecoderLayerWeight<T>::LLaMADecoderLayerWeight(const LLaMADecoderLayerWeight& other):
    hidden_units_(other.hidden_units_), inter_size_(other.inter_size_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_);
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], inter_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], hidden_units_ * inter_size_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], inter_size_);
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
    setWeightPtr();
}

template<typename T>
LLaMADecoderLayerWeight<T>& LLaMADecoderLayerWeight<T>::operator=(const LLaMADecoderLayerWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_   = other.inter_size_;

    mallocWeights();

    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_);
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], inter_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], hidden_units_ * inter_size_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], inter_size_);
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
    setWeightPtr();
    return *this;
}

template<typename T>
void LLaMADecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(
        weights_ptr[0], {(size_t)hidden_units_}, dir_path + ".attention_norm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[1], {(size_t)hidden_units_}, dir_path + ".attention_norm.weight.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[2],
                         {(size_t)hidden_units_, (size_t)(3 * hidden_units_)},
                         dir_path + ".attention.query_key_value.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {(size_t)(3 * hidden_units_)},
                         dir_path + ".attention.query_key_value.bias.bin",
                         model_file_type);

    loadWeightFromBin<T>(weights_ptr[4],
                         {(size_t)(hidden_units_), (size_t)hidden_units_},
                         dir_path + ".attention.wo.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5], {(size_t)hidden_units_}, dir_path + ".attention.wo.bias.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[6],
                         {(size_t)hidden_units_, (size_t)(inter_size_)},
                         dir_path + ".feed_forward.w1.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[7], {(size_t)(inter_size_)}, dir_path + ".feed_forward.w1.bias.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[8],
                         {(size_t)(inter_size_), (size_t)hidden_units_},
                         dir_path + ".feed_forward.w2.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[9], {(size_t)hidden_units_}, dir_path + ".feed_forward.w2.bias.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[10],
                         {(size_t)hidden_units_, (size_t)(inter_size_)},
                         dir_path + ".feed_forward.w3.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[11], {(size_t)(inter_size_)}, dir_path + ".feed_forward.w3.bias.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[12], {(size_t)hidden_units_}, dir_path + ".ffn_norm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[13], {(size_t)hidden_units_}, dir_path + ".ffn_norm.weight.bin", model_file_type);
}

template<typename T>
void LLaMADecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta                            = weights_ptr[0];
    pre_layernorm_weights.gamma                           = weights_ptr[1];
    self_attention_weights.query_weight.kernel            = weights_ptr[2];
    self_attention_weights.query_weight.bias              = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias   = weights_ptr[5];

    ffn_weights.intermediate_weight.kernel  = weights_ptr[6];
    ffn_weights.intermediate_weight.bias    = weights_ptr[7];
    ffn_weights.output_weight.kernel        = weights_ptr[8];
    ffn_weights.output_weight.bias          = weights_ptr[9];
    ffn_weights.intermediate_weight2.kernel = weights_ptr[10];
    ffn_weights.intermediate_weight2.bias   = weights_ptr[11];

    post_attention_layernorm_weights.beta  = weights_ptr[12];
    post_attention_layernorm_weights.gamma = weights_ptr[13];
    is_maintain_buffer                     = true;
}

template<typename T>
void LLaMADecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_);
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_);
    deviceMalloc(&weights_ptr[4], hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[5], hidden_units_);

    deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_);
    deviceMalloc(&weights_ptr[7], inter_size_);
    deviceMalloc(&weights_ptr[8], inter_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[9], hidden_units_);
    deviceMalloc(&weights_ptr[10], hidden_units_ * inter_size_);
    deviceMalloc(&weights_ptr[11], inter_size_);
    deviceMalloc(&weights_ptr[12], hidden_units_);
    deviceMalloc(&weights_ptr[13], hidden_units_);
}

template struct LLaMADecoderLayerWeight<float>;
template struct LLaMADecoderLayerWeight<half>;

}  // namespace fastertransformer
