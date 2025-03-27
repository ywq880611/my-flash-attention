#include <torch/extension.h>

torch::Tensor flash_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd", torch::wrap_pybind_function(flash_attn_fwd), "flash_attn_fwd");
}