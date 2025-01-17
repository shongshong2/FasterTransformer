/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "3rdparty/INIReader.h"
#include "examples/cpp/llama/llama_example_utils.h"
#include "src/fastertransformer/models/llama/LLaMA.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/word_list.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace fastertransformer;

template<typename T>
void llama_example(const INIReader reader);

int main(int argc, char* argv[])
{
    mpi::initialize(&argc, &argv);
    srand(0);

    std::string ini_name;
    if (argc == 2) {
        ini_name = std::string(argv[1]);
    }
    else {
        ini_name = "../examples/cpp/llama/llama_config.ini";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        llama_example<float>(reader);
    }
    else if (data_type == "fp16") {
        llama_example<half>(reader);
    }
    else {
        FT_LOG_ERROR("is_fp16 should be 0 (use float) or 1 (use half).");
        return -1;
    }
    mpi::finalize();
    return 0;
}

template<typename T>
void llama_example(const INIReader reader)
{
    const std::string model_name         = reader.Get("ft_instance_hyperparameter", "model_name");
    std::string       model_dir          = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));
    int               pipeline_para_size = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");

    const size_t head_num             = reader.GetInteger(model_name, "head_num");
    const size_t size_per_head        = reader.GetInteger(model_name, "size_per_head");
    const size_t vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t decoder_layers       = reader.GetInteger(model_name, "decoder_layers");
    const size_t rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
    const int    multiple_of          = reader.GetInteger(model_name, "multiple_of");
    const size_t max_seq_len          = reader.GetInteger(model_name, "max_seq_len");

    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size   = multiple_of * (((8 * hidden_units / 3) + multiple_of - 1) / multiple_of);

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");
    const int    padding_id         = reader.GetInteger(model_name, "padding_id");
    int          start_pos          = reader.GetInteger("request", "start_pos", 0);
    unsigned long long random_seed = reader.GetInteger("request", "random_seed", 0);

    FT_CHECK(decoder_layers % pipeline_para_size == 0);

    // Prepare the parallelism parameters
    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }
    int device, device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    printf("P%d is running with GPU #%d.\n", rank, device);
    if (pipeline_para_size != world_size) {
        printf("[ERROR] pipeline_para_size should equal to world_size \n");
        exit(-1);
    }

    const int layers_per_group = decoder_layers / pipeline_para_size;
    if (layers_per_group * pipeline_para_size != (int)decoder_layers) {
        printf("[ERROR] layers_per_group (%d) * pipeline_para_size (%d) should equal to decoder_layers (%ld) \n",
               layers_per_group,
               pipeline_para_size,
               decoder_layers);
        exit(-1);
    }

    NcclParam tensor_para;
    NcclParam pipeline_para;
    ftNcclInitialize(tensor_para, pipeline_para, 1, pipeline_para_size);

    // Read ids of request from file.
    size_t           max_input_len = -1;
    std::vector<int> v_start_lengths;
    std::vector<int> v_start_ids;
    read_start_ids(request_batch_size,
                   &v_start_lengths,
                   &v_start_ids,
                   max_input_len,
                   padding_id,
                   1,
                   "../examples/cpp/llama/start_ids.csv");

    int* d_input_ids;
    int* d_input_lengths;
    if (max_input_len == 0) {
        // unconditional case, no input ids, so do nothing.
        d_input_ids     = nullptr;
        d_input_lengths = nullptr;
    }
    else {
        // conditional case.
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);
        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size);
    }

    const int total_output_len = max_input_len;

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex*     cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    fastertransformer::LLaMAWeight<T> llama_weights(
        hidden_units, inter_size, vocab_size, decoder_layers, pipeline_para.world_size_, pipeline_para.rank_);

    model_dir = model_dir + "/" + std::to_string(tensor_para.world_size_) + "-gpu";
    llama_weights.loadModel(model_dir);

    if (world_size > 1) {
        mpi::bcast(&random_seed, 1, mpi::MPI_TYPE_UNSIGNED_LONG_LONG, 0, mpi::COMM_WORLD);
    }

    AttentionType attention_type =
        getAttentionType<T>(size_per_head,
                            getSMVersion(),
                            !((std::getenv("SHONG_PADDING") != nullptr)
                              && (std::string(std::getenv("SHONG_PADDING")) == "ON")),  // true,  // remove_padding
                            0,      // llama supports any-seq-length fmha
                            true,   // is_fuse
                            false,  // with_relative_position_bias
                            true);  // causal_mask

    switch (attention_type) {
        case AttentionType::UNFUSED_MHA:
            std::cout << "UNFUSED_MHA\n";
            break;
        case AttentionType::UNFUSED_PADDED_MHA:
            std::cout << "UNFUSED_PADDED_MHA\n";
            break;
        case AttentionType::FUSED_MHA:
            std::cout << "FUSED_MHA\n";
            break;
        case AttentionType::FUSED_PADDED_MHA:
            std::cout << "FUSED_PADDED_MHA\n";
            break;
    }

    LLaMA<T> llama = LLaMA<T>(head_num,
                              size_per_head,
                              inter_size,
                              decoder_layers,
                              vocab_size,
                              rotary_embedding_dim,
                              random_seed,
                              max_seq_len,
                              tensor_para,
                              pipeline_para,
                              stream,
                              &cublas_wrapper,
                              &allocator,
                              false,  // is_free_buffer_after_forward
                              &prop,
                              attention_type);

    float* d_output_logits;
    deviceMalloc(&d_output_logits, request_batch_size * total_output_len * vocab_size, false);
    std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
        {"input_ids",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, (size_t)max_input_len}, d_input_ids}},
        {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths}},
        {"start_pos", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &start_pos}}};

    std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
        {"output_logits",
         Tensor{MEMORY_GPU,
                TYPE_FP32,
                std::vector<size_t>{request_batch_size, (size_t)total_output_len, vocab_size},
                d_output_logits}}};

    print_mem_usage();

    int ite = 1;
    cudaDeviceSynchronize();
    mpi::barrier();

    // warm up
    ite = 1;
    ft_nvtx::setScope("warmup_time");
    PUSH_RANGE("warmup time")
    for (int i = 0; i < ite; ++i) {
        llama.forward(&output_tensors, &input_tensors, &llama_weights);
    }
    cudaDeviceSynchronize();
    mpi::barrier();

    POP_RANGE;
    ft_nvtx::resetScope();

    /*
    if (rank == world_size - 1) {
        float* out = (float*)malloc(sizeof(float) * request_batch_size * total_output_len * vocab_size);
        cudaMemcpy(out,
                   d_output_logits,
                   sizeof(float) * request_batch_size * total_output_len * vocab_size,
                   cudaMemcpyDeviceToHost
                   );
        for (int b = 0; b < request_batch_size; ++b) {
            std::cout << "[";
            for (int s = 0; s < total_output_len; ++s) {
                std::cout << "[";
                for (int v = vocab_size - 8; v < vocab_size; ++v) {
                    std::cout << out[b * total_output_len * vocab_size + s * vocab_size + v] << " ";
                }
                std::cout << "]\n";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
        free(out);
    }
    */

    // test time
    cudaProfilerStart();
    struct timeval start, end;
    cudaDeviceSynchronize();
    mpi::barrier();

    gettimeofday(&start, NULL);

    ft_nvtx::setScope("total_time");
    PUSH_RANGE("total time")
    // warm up
    ite = 10;
    for (int i = 0; i < ite; ++i) {
        llama.forward(&output_tensors, &input_tensors, &llama_weights);
    }

    cudaDeviceSynchronize();
    mpi::barrier();

    POP_RANGE;
    ft_nvtx::resetScope();
    gettimeofday(&end, NULL);
    cudaProfilerStop();

    printf("[INFO] request_batch_size %ld head_num %ld size_per_head %ld total_output_len %d"
           " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
           request_batch_size,
           head_num,
           size_per_head,
           total_output_len,
           decoder_layers,
           vocab_size,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    ftNcclParamDestroy(tensor_para);
    ftNcclParamDestroy(pipeline_para);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    if (d_input_ids != nullptr) {
        cudaFree(d_input_ids);
    }
    if (d_input_lengths != nullptr) {
        cudaFree(d_input_lengths);
    }
    if (d_output_logits != nullptr) {
        deviceFree(d_output_logits);
    }

    return;
}
