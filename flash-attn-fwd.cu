#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void load_into_smem(float* dst, const float* src, int n) {
    int tx = threadIdx.x;
    for(int i = tx; i < n; i += blockDim.x) {
        // printf("load_into_smem: %d\n", i);
        dst[i] = src[i];
    }
}

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}


__global__
void fwd_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int thread_cnt = blockDim.x;
    int bx = blockIdx.x; 
    int by = blockIdx.y; 

    const int r_tile_size = Br * d;
    const int c_tile_size = d * Bc;
    const int p_tile_size = Br * Bc;
    const int tile_size = Bc * d;


    float* Qi = smem;
    float* Kj = Qi + r_tile_size;
    float* Vj = Kj + c_tile_size;
    float* S = Vj + c_tile_size;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m
    // const int qkv_offset = bx * gridDim.y * N * d + by * N * d;
    // const int lm_offset = bx * gridDim.y * N + by * N; 

    // const float* q_ptr = Q + qkv_offset;
    // const float* k_ptr = K + qkv_offset;
    // const float* v_ptr = V + qkv_offset;

    for (int j = 0; j < Tc; j ++) {
        // printf("loop for %d\n", j);
        // load_into_smem(sKj, k_ptr + c_tile_size * j, c_tile_size);
        // load_into_smem(sVj, v_ptr + r_tile_size * j, r_tile_size);
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();

        
        for (int i = 0; i < Tr; i++) {
            // printf("loop 1 for %d\n", i);
            // load_into_smem(sQi, q_ptr + r_tile_size * i, r_tile_size);
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            __syncthreads();

            // float r_max_prev = m[lm_offset + Br * i + tx];
            // float r_logits_prev = l[lm_offset + Br * i + tx];
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // TODO(Wenqin): support more than one row for each thread here, we
            // may have to introduce a new loop here for multiple row for one
            // thread.

            // Get QK^T and r_max for a tile.
            // float r_max = -INFINITY;
            // for(int c = 0; c < Bc; c ++) {
            //     int sum = 0;
            //     for(int t = 0; t < d; t ++) {
            //         sum += Qi[tx * d + t] * Kj[c * d + t];
            //     }
            //     sum *= softmax_scale;
            //     S[tx * Bc + c] = sum;

            //     if(sum > r_max) r_max = sum;
            // }
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_max), r_logits = rowsum(P)
            // float r_logits = 0;
            // for(int y = 0; y < Bc; y ++) {
            //     // sPij[tx * Bc + c] = __expf(sPij[tx * Bc + c] - r_max);
            //     // r_logits += sPij[tx * Bc + c];
            //     S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
            //     r_logits += S[(Bc * tx) + y];
            // }
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Now is the critical parts for flash attention, we should do combination
            // between current sPij in SRAM with O in HBM. 

            // Get the previous stored data.
            
            
            // float r_logits_new = __expf(r_max_prev - r_max_new) * r_logits_prev + 
            //                         __expf(r_max - r_max_new) * r_logits;
            // float r_max_new = max(row_m_prev, row_m);
            // float r_logits_new = (__expf(row_m_prev - r_max_new) * row_l_prev) + (__expf(row_m - r_max_new) * row_l);
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m back to HBM
            // for(int c = 0; c < d; c ++) {
            //     float pv = 0;
            //     for(int x = 0; x < Bc; x ++) {
            //         pv += sPij[tx * Bc + x] * sVj[x * d + c];
            //     }
            //     // const int o_offset = qkv_offset + (i * Br + tx) * d + c;
            //     // const int o_offset = qkv_offset + (r_tile_size * i) + (tx * d) + c;
            //     // O[o_offset] = (__expf(r_max_prev - r_max_new) * r_logits_prev * O[o_offset] +
            //     //                 __expf(r_max - r_max_new) * pv) * (1 / r_logits_new);
            //     O[qkv_offset + (r_tile_size * i) + (tx * d) + c] = (1 / r_logits_new) \
            //         * ((r_logits_prev * __expf(r_max_prev - r_max_new) * O[qkv_offset + (r_tile_size * i) + (tx * d) + c]) \
            //         + (__expf(r_max - r_max_new) * pv));
            // }

            // for (int x = 0; x < d; x++) {
            //     float pv = 0;  // Pij * Vj
            //     for (int y = 0; y < Bc; y++) {
            //         pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
            //     }
            //     O[qkv_offset + (r_tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
            //         * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (r_tile_size * i) + (tx * d) + x]) \
            //         + (__expf(row_m - row_m_new) * pv));
            // }

            // // m[lm_offset + (Br * i) + tx] = r_max_new;
            // // l[lm_offset + (Br * i) + tx] = r_logits_new;
            // m[lm_offset + (Br * i) + tx] = row_m_new;
            // l[lm_offset + (Br * i) + tx] = row_l_new;
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor flash_attn_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: if I have time, try tune this configuration later.
    const int Br = 32;
    const int Bc = 32;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);

    const float softmax_scale = 1.0 / sqrt(D);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, H, N});
    auto m = torch::full({B, H, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int scalar_size = sizeof(float);
    const int sram_size = (Br * D + 2 * Bc * D + Bc * Br) * scalar_size;
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid(B, H);  // batch_size, num_heads
    dim3 block(Bc);  // 32 threads per block

    fwd_kernel<<<grid, block, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, D, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}