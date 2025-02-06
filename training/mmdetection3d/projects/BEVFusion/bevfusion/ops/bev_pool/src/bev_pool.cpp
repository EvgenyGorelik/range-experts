#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

// CUDA function declarations
void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_bfloat16(int b, int d, int h, int w, int n, int c, int n_intervals, const __nv_bfloat16* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, __nv_bfloat16* out);

void bev_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad);

void bev_pool_grad_bfloat16(int b, int d, int h, int w, int n, int c, int n_intervals, const __nv_bfloat16* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, __nv_bfloat16* x_grad);


/*
  Function: pillar pooling (forward, cuda)
  Args:
    x                : input features, FloatTensor[n, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    out              : output features, FloatTensor[b, d, h, w, c]
*/
at::Tensor bev_pool_forward(
  const at::Tensor _x,
  const at::Tensor _geom_feats,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _x.size(0);
  int c = _x.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_x));
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  at::Tensor _out = torch::zeros({b, d, h, w, c}, options);

  if (_x.scalar_type() == torch::kFloat) {
    const float* x = _x.data_ptr<float>();
    float* out = _out.data_ptr<float>();
    bev_pool(
      b, d, h, w, n, c, n_intervals, x,
      geom_feats, interval_starts, interval_lengths, out
    );
  } else if (_x.scalar_type() == torch::kBFloat16) {
    const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(_x.data_ptr<at::BFloat16>());
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(_out.data_ptr<at::BFloat16>());
    bev_pool_bfloat16(
      b, d, h, w, n, c, n_intervals, x,
      geom_feats, interval_starts, interval_lengths, out
    );
  } else {
    AT_ERROR("Unsupported input type");
  }
  return _out;
}


/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : input features, FloatTensor[b, d, h, w, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    x_grad           : output features, FloatTensor[n, 4]
*/
at::Tensor bev_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_feats,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _geom_feats.size(0);
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _x_grad = torch::zeros({n, c}, options);

  if (_out_grad.scalar_type() == torch::kFloat) {
    const float* out_grad = _out_grad.data_ptr<float>();
    float* x_grad = _x_grad.data_ptr<float>();
    bev_pool_grad(
      b, d, h, w, n, c, n_intervals, out_grad,
      geom_feats, interval_starts, interval_lengths, x_grad
    );
  } else if (_out_grad.scalar_type() == torch::kBFloat16) {
    const __nv_bfloat16* out_grad = reinterpret_cast<const __nv_bfloat16*>(_out_grad.data_ptr<at::BFloat16>());
    __nv_bfloat16* x_grad = reinterpret_cast<__nv_bfloat16*>(_x_grad.data_ptr<at::BFloat16>());
    bev_pool_grad_bfloat16(
      b, d, h, w, n, c, n_intervals, out_grad,
      geom_feats, interval_starts, interval_lengths, x_grad
    );
  } else {
    AT_ERROR("Unsupported input type");
  }

  return _x_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_forward", &bev_pool_forward,
        "bev_pool_forward");
  m.def("bev_pool_backward", &bev_pool_backward,
        "bev_pool_backward");
}
