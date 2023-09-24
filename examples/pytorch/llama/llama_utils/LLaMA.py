import torch
import torch.nn as nn
import os
import numpy as np
from llama_utils.model_args import ModelArgs
import torch.distributed as dist

class LLaMA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Load the C++ model into Pytorch model.
        torch.classes.load_library(os.path.abspath(args.lib_path))
        
        # Prepare model weight
        weight_data_type = np.float16 if args.weight_data_type == 'fp16' else np.float32
        inference_data_type = torch.float16 if args.inference_data_type == 'fp16' else torch.float32
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        layers_per_device = args.decoder_layers / self.world_size

        self.w = []
        file_names = [
                        "attention_norm.bias",
                        "attention_norm.weight",
                        "attention.query_key_value.weight",
                        "attention.query_key_value.bias",
                        "attention.wo.weight",
                        "attention.wo.bias",
                        "feed_forward.w1.weight",
                        "feed_forward.w1.bias",
                        "feed_forward.w2.weight",
                        "feed_forward.w2.bias",
                        "feed_forward.w3.weight",
                        "feed_forward.w3.bias",
                        "ffn_norm.bias",
                        "ffn_norm.weight"
                    ]
        for file_name in file_names:
            for i in range(args.decoder_layers):
                if file_name is not None and i >= self.local_rank * layers_per_device and i < (self.local_rank + 1) * layers_per_device:
                    self.w.append(torch.from_numpy(np.fromfile(
                                "%s/model.layers.%d.%s.bin" % (args.ckpt_path, i, file_name),
                                dtype=weight_data_type)).to(inference_data_type).cuda())

        self.w.append(torch.from_numpy(np.fromfile(args.ckpt_path + "/model.tok_embeddings.weight.bin", dtype=weight_data_type)).to(inference_data_type).cuda())
        self.w.append(torch.from_numpy(np.fromfile(args.ckpt_path + "/model.norm.weight.bin", dtype=weight_data_type)).to(inference_data_type).cuda())
        self.w.append(torch.from_numpy(np.fromfile(args.ckpt_path + "/model.norm.bias.bin", dtype=weight_data_type)).to(inference_data_type).cuda())
        self.w.append(torch.from_numpy(np.fromfile(args.ckpt_path + "/model.output.weight.bin", dtype=weight_data_type)).to(inference_data_type).cuda())

        # Init C++ model
        self.model = torch.classes.FasterTransformer.LLaMA(
                                                            args.head_num,
                                                            args.size_per_head,
                                                            args.inter_size,
                                                            args.decoder_layers,
                                                            args.vocab_size,
                                                            args.rotary_embedding,
                                                            0,  # start_id
                                                            args.padding_id, # end_id
                                                            1,  # tensor para size
                                                            self.world_size, # pipeline_para_size
                                                            args.max_seq_len,
                                                            args.use_gptj_residual,
                                                            self.w
                                                            )



    def forward(self,
                input_ids,
                input_lengths,
                max_input_len):
        
        input_ids = torch.stack(input_ids).cuda() # batch size x max_input_len
        input_lengths = torch.stack(input_lengths).cuda()
        max_input_len = len(input_ids[0])
        total_output_len = max_input_len
        
        # request_batch_size, total_output_len, vocab_size, max_input_len
        outputs = self.model.forward(
                                        input_ids,
                                        input_lengths,
                                        total_output_len,
                                        1, # optional beam_width_opt
                                        torch.Tensor(0).cuda(), # optional top_k_opt
                                        torch.Tensor(0).cuda(), # optional top_p_opt
                                        torch.Tensor(0).cuda(), # optional beam_search_diversity_rate_opt
                                        torch.Tensor(0).cuda(), # optional temperature_opt
                                        torch.Tensor(0).cuda(), # optional len_penalty_opt
                                        torch.Tensor(0).cuda(), # optional repetition_penalty_opt
                                        torch.Tensor(0).cuda(), # optional random_seed_opt
                                        0  # optional return_cum_log_probs_opt 
                                    )
        
        dist.barrier()
        
        if self.local_rank == self.world_size - 1:
            print(len(outputs))
            print(outputs)
            return outputs 
        else:
            return None
    

