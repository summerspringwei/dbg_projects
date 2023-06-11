import math
import os

import torch
import numpy as np


# Get data
# shape: [bsz, nh, t, hd]
folder_path = "data"
query_name = "bert_encoder_layer_0_attention_self_query_kernel_0.npy"
key_name = "bert_encoder_layer_0_attention_self_key_kernel_0.npy"
query_np = np.load(os.path.join(folder_path, query_name))
key_np = np.load(os.path.join(folder_path, key_name))
print(query_np.shape)
print(key_np.shape)
query = torch.from_numpy(query_np)
key = torch.from_numpy(key_np)
batch_size, num_head, query_seq_length, key_seq_length, hidden_size = 2, 12, 384, 384, 64
# query = torch.ones((batch_size, num_head, query_seq_length, hidden_size), dtype=torch.float32)
# key = torch.ones((batch_size, num_head, query_seq_length, hidden_size), dtype=torch.float32)

def compare(query, key, dtype1=torch.float32, dtype2=torch.float32):
    # llama
    bsz, nh, t, hd = batch_size, num_head, query_seq_length, hidden_size
    query_states = torch.reshape(query, (bsz, nh, t, hd)).to('cuda').to(dtype1)
    key_states = torch.reshape(key, (bsz, nh, t, hd)).to('cuda').to(dtype1)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(hd)
    attn_weights = attn_weights.reshape((bsz * nh, t, t)).to(torch.float32)

    # Megatron-LM
    b, np, sq, sk, hn = batch_size, num_head, query_seq_length, key_seq_length, hidden_size
    matmul_input_buffer = torch.zeros((b * np, sq, sk), dtype=torch.float32).to('cuda')
    query_layer = torch.reshape(query, (b * np, sq, hn)).to('cuda').to(dtype2)
    key_layer = torch.reshape(key, (b * np, sq, hn)).to('cuda').to(dtype2)
    matmul_result = torch.baddbmm(
        matmul_input_buffer,
        query_layer,   # [b * np, sq, hn]
        key_layer.transpose(1, 2),  # [b * np, hn, sk]
        beta=0.0, alpha=(1.0 / math.sqrt(hd))).to(torch.float32)

    # print(attn_weights)
    # print(matmul_result)
    equal = torch.allclose(attn_weights, matmul_result)
    print(f"attn_weights and matmul_result equal? {equal}")
    if not equal:
        print(attn_weights - matmul_result)

# Version 1, both LLAMA and Megatron use float32
# we can verify that the results are equal
compare(query, key)
# Version 2, LLAMA uses float16 and Megatron use float32, 
# We can verify that there is precision loss for LLAMA (with around 10e-7 on average)
compare(query, key, dtype1=torch.float16, dtype2=torch.float32)
# Version 3, both LLAMA and Megatron use TensorFloat32,
# we can verify that the results are equal
torch.backends.cuda.matmul.allow_tf32 = True
compare(query, key)
