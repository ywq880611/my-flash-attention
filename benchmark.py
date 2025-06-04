import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
lib = load(name='lib', sources=['main.cpp', 'flash-attn-fwd.cu'], extra_cuda_cflags=['-O2'])
enable_profiler = False

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()


# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

naive_attn_time = 0
flash_attn_time = 0
test_round = 10

if enable_profiler:
    print('=== profiling manual attention ===')
    with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
        manual_result = manual_attn(q, k, v)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
else:
    # warm up
    for _ in range(test_round):
        manual_result = manual_attn(q, k, v)
    
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start1.record()
    # test
    for _ in range(test_round):
        manual_attn(q, k, v)
    end1.record()
    torch.cuda.synchronize()

    naive_attn_time = start1.elapsed_time(end1)

if enable_profiler:
    print('=== profiling minimal flash attention === ')
    with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
        minimal_result = lib.flash_attn_fwd_v1(q, k, v)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
else:
    # warm up
    for _ in range(test_round):
        minimal_result = lib.flash_attn_fwd_v1(q, k, v)
    
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start2.record()
    # test
    for _ in range(test_round):
        lib.flash_attn_fwd_v1(q, k, v)
    end2.record()
    torch.cuda.synchronize()

    flash_attn_time_v1 = start2.elapsed_time(end2)

if(torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02)):
    print("Results true!")
else:
    print("Error results:")
    print(minimal_result)
    print("================================")
    print(manual_result)

if not enable_profiler:
    print("time for naive attn   : {:.3f}".format(naive_attn_time))
    print("time for flash attn v1: {:.3f}".format(flash_attn_time_v1))