# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(LLaMADecoderLayerWeight STATIC LLaMADecoderLayerWeight.cc)
set_property(TARGET LLaMADecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LLaMADecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LLaMADecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(LLaMAContextDecoder STATIC LLaMAContextDecoder.cc)
set_property(TARGET LLaMAContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LLaMAContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LLaMAContextDecoder PUBLIC -lcudart cublasMMWrapper
                      LLaMAContextAttentionLayer
                      FfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      llama_kernels
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(LLaMAWeight STATIC LLaMAWeight.cc)
set_property(TARGET LLaMAWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LLaMAWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LLaMAWeight PUBLIC LLaMADecoderLayerWeight cuda_utils logger)

add_library(LLaMA STATIC LLaMA.cc)
set_property(TARGET LLaMA PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LLaMA PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LLaMA PUBLIC -lcudart
                      LLaMAContextDecoder
                      decoding_kernels
                      llama_kernels
                      tensor
                      LLaMAWeight
                      cuda_utils
                      logger)
