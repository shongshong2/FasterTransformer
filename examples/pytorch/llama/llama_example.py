import argparse
import configparser
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from llama_utils.tokenizer import Tokenizer
from llama_utils.LLaMA import LLaMA
from llama_utils.model_args import ModelArgs
from torch.nn.utils.rnn import pad_sequence

def main():
    model_args = ModelArgs()

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--lib_path', type=str, default='/home/n2/junsik/lab/FasterTransformer/build/lib/libth_transformer.so')
    parser.add_argument('--sample_input_file', type=str, default='input_ids.csv')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer/tokenizer.model')
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    lib_path = args.lib_path
    sample_input_file = args.sample_input_file
    tokenizer_path = args.tokenizer_path

    # read hyperparameters from llama_config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(args.ckpt_path, "llama_config.ini"))
    head_num = int(config.get('llama', 'head_num'))
    size_per_head = int(config.get('llama', 'size_per_head'))
    vocab_size = int(config.get('llama', 'vocab_size'))
    decoder_layers = int(config.get('llama', 'decoder_layers'))
    rotary_embedding = int(config.get('llama', 'rotary_embedding'))
    multiple_of = int(config.get('llama', 'multiple_of'))
    max_cache_seq_len = int(config.get('llama', 'max_cache_seq_len'))
    padding_id = int(config.get('llama', 'padding_id'))
    weight_data_type = config.get('llama', 'weight_data_type')
    inference_data_type = config.get('llama', 'inference_data_type')
    pipeline_para_size = int(config.get('ft_instance_hyperparameter', 'pipeline_para_size'))
    beam_width = int(config.get('request', 'beam_width'))
    request_batch_size = int(config.get('request', 'request_batch_size'))

    hidden_units = head_num * size_per_head
    inter_size = multiple_of * int(((8 * int(hidden_units / 3)) + multiple_of - 1) / multiple_of)
    min_length = 0
    
    # prepare model arguments
    model_args.ckpt_path = ckpt_path 
    model_args.lib_path = lib_path 
    model_args.sample_input_file = sample_input_file 
    model_args.tokenizer_path = tokenizer_path 
    model_args.head_num = int(head_num)
    model_args.size_per_head = int(size_per_head)
    model_args.vocab_size = int(vocab_size)
    model_args.decoder_layers = int(decoder_layers)
    model_args.rotary_embedding = int(rotary_embedding)
    model_args.multiple_of = int(multiple_of)
    model_args.max_cache_seq_len = int(max_cache_seq_len)
    model_args.padding_id = int(padding_id)
    model_args.weight_data_type = weight_data_type 
    model_args.inference_data_type = inference_data_type 
    model_args.pipeline_para_size = int(pipeline_para_size)
    model_args.beam_width = int(beam_width)
    model_args.request_batch_size = int(request_batch_size)
    model_args.hidden_units = int(hidden_units)
    model_args.inter_size = int(inter_size)
    model_args.min_length = int(min_length)
   
    
    # Prepare for pipeline parallel
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    local_rank = comm.Get_rank()
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = str('192.168.0.101')
    os.environ["MASTER_PORT"] = str('1234')
    
    if pipeline_para_size > 1 :
        dist.init_process_group(backend='nccl')
    
    assert dist.is_initialized() != 0, "nccl init failed!"
    assert decoder_layers % pipeline_para_size == 0
    
    device = local_rank % world_size
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()
    layers_per_group = decoder_layers / pipeline_para_size
    
    assert layers_per_group * pipeline_para_size == decoder_layers

    # tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_path)
    
    # read ids from input_ids.csv file
    tokens = []
    input_lengths = []
    max_input_len = -1
    with open(sample_input_file, 'r') as f:
        sentences = f.read().splitlines()
        for i, s in enumerate(sentences[:request_batch_size]):
            tokens.append(sentences[i].split(','))
            tokens[i] = [int(t.strip(' ')) for t in tokens[i]]
            curr_input_len = len(tokens[i])
            input_lengths.append(curr_input_len)
            if curr_input_len > max_input_len:
                max_input_len = curr_input_len

    input_ids = [torch.tensor(c, dtype=torch.int32, device=device) for c in tokens]
    input_lengths = [torch.tensor(c, dtype=torch.int32, device=device) for c in input_lengths]
    max_seq_len = 1024
    model_args.max_seq_len = max_seq_len
    max_input_len = -1
    for i, l in enumerate(input_lengths):
        max_input_len = l if l > max_input_len else max_input_len

    # init model
    llama = LLaMA(model_args)

    # forward
    # input_ids:4,20,32000
    # input_lengths:4,20
    # output_seq_len:4
    # output_logits:4,20,32000
    output = llama.forward(input_ids, input_lengths, max_input_len)

if __name__ == '__main__':
    main()
