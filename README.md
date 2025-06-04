# my-flash-attention
My simple flash attention

The idea came from this [repo](https://github.com/tspeterkim/flash-attention-minimal).

## Output
The output looks like:
```
Results true!
time for naive attn   : 0.763
time for flash attn v1: 11.108
```

We could see the perf of flash attn is worse than naive attn, the reason is that the kernel just use **32 threads** in a block, and didn't do optimization for **dot product** between $Q@K^T$, and $P@V$, so we'd expect a worse perf.