#include <torch/extension.h>

torch::Tensor flash_attn_fwd_v1(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attn_fwd_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd_v1", torch::wrap_pybind_function(flash_attn_fwd_v1), "flash_attn_fwd_v1");
    m.def("flash_attn_fwd_v2", torch::wrap_pybind_function(flash_attn_fwd_v2), "flash_attn_fwd_v2");
}