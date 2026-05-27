#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kNumerator = 6;
constexpr int kDenominator = 4;
constexpr int kGradCount = kNumerator + kDenominator;
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__device__ __forceinline__ float sign0(float x) {
  return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

__global__ void rational_forward_kernel(const float* __restrict__ x,
                                        const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ y,
                                        int64_t n) {
  __shared__ float ca[kNumerator];
  __shared__ float cb[kDenominator];
  if (threadIdx.x < kNumerator) {
    ca[threadIdx.x] = a[threadIdx.x];
  }
  if (threadIdx.x < kDenominator) {
    cb[threadIdx.x] = b[threadIdx.x];
  }
  __syncthreads();

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    const float xv = x[i];

    const float p =
        fmaf(fmaf(fmaf(fmaf(fmaf(ca[5], xv, ca[4]), xv, ca[3]), xv, ca[2]),
                  xv,
                  ca[1]),
             xv,
             ca[0]);
    const float q =
        fmaf(fmaf(fmaf(fmaf(cb[3], xv, cb[2]), xv, cb[1]), xv, cb[0]), xv, 1.0f);
    y[i] = __fdividef(p, q);
  }
}

__global__ void rational_version_a_forward_kernel(const float* __restrict__ x,
                                                  const float* __restrict__ a,
                                                  const float* __restrict__ b,
                                                  float* __restrict__ y,
                                                  int64_t n) {
  __shared__ float ca[kNumerator];
  __shared__ float cab[kDenominator];
  if (threadIdx.x < kNumerator) {
    ca[threadIdx.x] = a[threadIdx.x];
  }
  if (threadIdx.x < kDenominator) {
    cab[threadIdx.x] = fabsf(b[threadIdx.x]);
  }
  __syncthreads();

  const float a0 = ca[0];
  const float a1 = ca[1];
  const float a2 = ca[2];
  const float a3 = ca[3];
  const float a4 = ca[4];
  const float a5 = ca[5];
  const float b0 = cab[0];
  const float b1 = cab[1];
  const float b2 = cab[2];
  const float b3 = cab[3];

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    const float xv = x[i];
    const float ax = fabsf(xv);
    const float x2 = xv * xv;
    const float x4 = x2 * x2;

    const float p =
        fmaf(fmaf(fmaf(fmaf(fmaf(a5, xv, a4), xv, a3), xv, a2),
                  xv,
                  a1),
             xv,
             a0);
    const float q = fmaf(fmaf(b2, x2, b0), ax, fmaf(b3, x4, fmaf(b1, x2, 1.0f)));
    y[i] = p * __frcp_rn(q);
  }
}

__global__ void rational_backward_stage1_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ grad_x,
    float* __restrict__ partials,
    int64_t n) {
  __shared__ float ca[kNumerator];
  __shared__ float cb[kDenominator];
  __shared__ float shared[kGradCount][kThreads];

  if (threadIdx.x < kNumerator) {
    ca[threadIdx.x] = a[threadIdx.x];
  }
  if (threadIdx.x < kDenominator) {
    cb[threadIdx.x] = b[threadIdx.x];
  }
  __syncthreads();

  float local[kGradCount];
#pragma unroll
  for (int j = 0; j < kGradCount; ++j) {
    local[j] = 0.0f;
  }

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    const float g = grad_output[i];
    const float xv = x[i];
    const float x2 = xv * xv;
    const float x3 = x2 * xv;
    const float x4 = x2 * x2;
    const float x5 = x4 * xv;

    const float p =
        fmaf(fmaf(fmaf(fmaf(fmaf(ca[5], xv, ca[4]), xv, ca[3]), xv, ca[2]),
                  xv,
                  ca[1]),
             xv,
             ca[0]);
    const float dp =
        fmaf(fmaf(fmaf(fmaf(5.0f * ca[5], xv, 4.0f * ca[4]), xv, 3.0f * ca[3]),
                  xv,
                  2.0f * ca[2]),
             xv,
             ca[1]);

    const float q =
        fmaf(fmaf(fmaf(fmaf(cb[3], xv, cb[2]), xv, cb[1]), xv, cb[0]), xv, 1.0f);
    const float dqdx = fmaf(fmaf(4.0f * cb[3], xv, 3.0f * cb[2]),
                            x2,
                            fmaf(2.0f * cb[1], xv, cb[0]));
    const float inv_q = __frcp_rn(q);
    const float inv_q2 = inv_q * inv_q;

    grad_x[i] = g * ((dp * q - p * dqdx) * inv_q2);

    local[0] += g * inv_q;
    local[1] += g * xv * inv_q;
    local[2] += g * x2 * inv_q;
    local[3] += g * x3 * inv_q;
    local[4] += g * x4 * inv_q;
    local[5] += g * x5 * inv_q;

    const float b_scale = -g * p * inv_q2;
    local[6] += b_scale * xv;
    local[7] += b_scale * x2;
    local[8] += b_scale * x3;
    local[9] += b_scale * x4;
  }

#pragma unroll
  for (int j = 0; j < kGradCount; ++j) {
    shared[j][threadIdx.x] = local[j];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
#pragma unroll
      for (int j = 0; j < kGradCount; ++j) {
        shared[j][threadIdx.x] += shared[j][threadIdx.x + offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < kGradCount) {
    partials[static_cast<int64_t>(threadIdx.x) * gridDim.x + blockIdx.x] =
        shared[threadIdx.x][0];
  }
}

__global__ void rational_version_a_backward_stage1_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ grad_x,
    float* __restrict__ partials,
    int64_t n) {
  __shared__ float ca[kNumerator];
  __shared__ float cab[kDenominator];
  __shared__ float csb[kDenominator];
  __shared__ float shared[kGradCount][kThreads];

  if (threadIdx.x < kNumerator) {
    ca[threadIdx.x] = a[threadIdx.x];
  }
  if (threadIdx.x < kDenominator) {
    const float bv = b[threadIdx.x];
    cab[threadIdx.x] = fabsf(bv);
    csb[threadIdx.x] = sign0(bv);
  }
  __syncthreads();

  const float a0 = ca[0];
  const float a1 = ca[1];
  const float a2 = ca[2];
  const float a3 = ca[3];
  const float a4 = ca[4];
  const float a5 = ca[5];
  const float b0 = cab[0];
  const float b1 = cab[1];
  const float b2 = cab[2];
  const float b3 = cab[3];
  const float sb0 = csb[0];
  const float sb1 = csb[1];
  const float sb2 = csb[2];
  const float sb3 = csb[3];

  float local[kGradCount];
#pragma unroll
  for (int j = 0; j < kGradCount; ++j) {
    local[j] = 0.0f;
  }

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += stride) {
    const float g = grad_output[i];
    const float xv = x[i];
    const float ax = fabsf(xv);
    const float x2 = xv * xv;
    const float x3 = x2 * xv;
    const float x4 = x2 * x2;
    const float x5 = x4 * xv;
    const float ax3 = ax * x2;

    const float p =
        fmaf(fmaf(fmaf(fmaf(fmaf(a5, xv, a4), xv, a3), xv, a2),
                  xv,
                  a1),
             xv,
             a0);
    const float dp =
        fmaf(fmaf(fmaf(fmaf(5.0f * a5, xv, 4.0f * a4), xv, 3.0f * a3),
                  xv,
                  2.0f * a2),
             xv,
             a1);

    const float q = fmaf(fmaf(b2, x2, b0), ax, fmaf(b3, x4, fmaf(b1, x2, 1.0f)));
    const float dqdx = b0 * sign0(xv) + 2.0f * b1 * xv +
                       3.0f * b2 * xv * ax + 4.0f * b3 * x3;
    const float inv_q = __frcp_rn(q);
    const float inv_q2 = inv_q * inv_q;

    grad_x[i] = g * ((dp * q - p * dqdx) * inv_q2);

    local[0] += g * inv_q;
    local[1] += g * xv * inv_q;
    local[2] += g * x2 * inv_q;
    local[3] += g * x3 * inv_q;
    local[4] += g * x4 * inv_q;
    local[5] += g * x5 * inv_q;

    const float b_scale = -g * p * inv_q2;
    local[6] += b_scale * sb0 * ax;
    local[7] += b_scale * sb1 * x2;
    local[8] += b_scale * sb2 * ax3;
    local[9] += b_scale * sb3 * x4;
  }

#pragma unroll
  for (int j = 0; j < kGradCount; ++j) {
    shared[j][threadIdx.x] = local[j];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
#pragma unroll
      for (int j = 0; j < kGradCount; ++j) {
        shared[j][threadIdx.x] += shared[j][threadIdx.x + offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < kGradCount) {
    partials[static_cast<int64_t>(threadIdx.x) * gridDim.x + blockIdx.x] =
        shared[threadIdx.x][0];
  }
}

__global__ void rational_backward_stage2_kernel(
    const float* __restrict__ partials,
    float* __restrict__ grad_numerator,
    float* __restrict__ grad_denominator,
    int blocks) {
  __shared__ float shared[kThreads];

  const int coeff = blockIdx.x;
  float local = 0.0f;
  for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
    local += partials[static_cast<int64_t>(coeff) * blocks + i];
  }
  shared[threadIdx.x] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (coeff < kNumerator) {
      grad_numerator[coeff] = shared[0];
    } else {
      grad_denominator[coeff - kNumerator] = shared[0];
    }
  }
}


constexpr int kFastRlbMaxBasis = 4;
constexpr int kFastRlbMaxGrad = kGradCount + 2 * kFastRlbMaxBasis;

__device__ __forceinline__ void fast_rlb_eval(float t,
                                               int group,
                                               int groups,
                                               int basis_count,
                                               float coeff_limit,
                                               const float* __restrict__ numerator,
                                               const float* __restrict__ denominator,
                                               const float* __restrict__ coeff_logits,
                                               const float* __restrict__ centers,
                                               const float* __restrict__ beta,
                                               float* f,
                                               float* dfdt,
                                               float* partials) {
  const float* a = numerator + static_cast<int64_t>(group) * kNumerator;
  const float* b_raw = denominator + static_cast<int64_t>(group) * kDenominator;
  const float* logits = coeff_logits + static_cast<int64_t>(group) * basis_count * 2;
  const float* ctr = centers + static_cast<int64_t>(group) * basis_count;
  const float* bet = beta + static_cast<int64_t>(group) * basis_count;

  const float ax = fabsf(t);
  const float t2 = t * t;
  const float t3 = t2 * t;
  const float t4 = t2 * t2;
  const float t5 = t4 * t;
  const float ax3 = ax * t2;

  const float p = fmaf(fmaf(fmaf(fmaf(fmaf(a[5], t, a[4]), t, a[3]), t, a[2]), t, a[1]), t, a[0]);
  const float dp = fmaf(fmaf(fmaf(fmaf(5.0f * a[5], t, 4.0f * a[4]), t, 3.0f * a[3]), t, 2.0f * a[2]), t, a[1]);

  const float b0 = fabsf(b_raw[0]);
  const float b1 = fabsf(b_raw[1]);
  const float b2 = fabsf(b_raw[2]);
  const float b3 = fabsf(b_raw[3]);
  const float q = 1.0f + b0 * ax + b1 * t2 + b2 * ax3 + b3 * t4;
  const float dqdt = b0 * sign0(t) + 2.0f * b1 * t + 3.0f * b2 * t * ax + 4.0f * b3 * t3;
  const float inv_q = __frcp_rn(q);
  const float inv_q2 = inv_q * inv_q;

  float out = p * inv_q;
  float deriv = (dp * q - p * dqdt) * inv_q2;

  partials[0] = inv_q;
  partials[1] = t * inv_q;
  partials[2] = t2 * inv_q;
  partials[3] = t3 * inv_q;
  partials[4] = t4 * inv_q;
  partials[5] = t5 * inv_q;
  const float b_scale = -p * inv_q2;
  partials[6] = b_scale * sign0(b_raw[0]) * ax;
  partials[7] = b_scale * sign0(b_raw[1]) * t2;
  partials[8] = b_scale * sign0(b_raw[2]) * ax3;
  partials[9] = b_scale * sign0(b_raw[3]) * t4;

#pragma unroll
  for (int idx = kGradCount; idx < kFastRlbMaxGrad; ++idx) {
    partials[idx] = 0.0f;
  }

  for (int k = 0; k < basis_count; ++k) {
    const float center = ctr[k];
    const float beta_k = bet[k];
    const float u = t - center;
    const float u2 = u * u;
    const float den = 1.0f + beta_k * u2;
    const float inv_den = __frcp_rn(den);
    const float inv_den2 = inv_den * inv_den;
    const float odd = u * inv_den;
    const float odd_dt = (1.0f - beta_k * u2) * inv_den2;
    const float zero = __frcp_rn(1.0f + beta_k * center * center);
    const float bump = inv_den - zero;
    const float bump_dt = -2.0f * beta_k * u * inv_den2;

    const float odd_logit = logits[2 * k];
    const float bump_logit = logits[2 * k + 1];
    const float odd_tanh = tanhf(odd_logit);
    const float bump_tanh = tanhf(bump_logit);
    const float odd_coeff = coeff_limit * odd_tanh;
    const float bump_coeff = coeff_limit * bump_tanh;

    out += odd_coeff * odd + bump_coeff * bump;
    deriv += odd_coeff * odd_dt + bump_coeff * bump_dt;

    partials[kGradCount + 2 * k] = coeff_limit * (1.0f - odd_tanh * odd_tanh) * odd;
    partials[kGradCount + 2 * k + 1] = coeff_limit * (1.0f - bump_tanh * bump_tanh) * bump;
  }

  *f = out;
  *dfdt = deriv;
}

__global__ void rational_local_basis_forward_kernel(const float* __restrict__ x,
                                                    const float* __restrict__ numerator,
                                                    const float* __restrict__ denominator,
                                                    const float* __restrict__ coeff_logits,
                                                    const float* __restrict__ centers,
                                                    const float* __restrict__ beta,
                                                    float* __restrict__ y,
                                                    int64_t rows,
                                                    int hidden_dim,
                                                    int groups,
                                                    int basis_count,
                                                    float coeff_limit,
                                                    float eps) {
  __shared__ float shared[kThreads];

  const int row_group = blockIdx.x;
  const int group = row_group % groups;
  const int row = row_group / groups;
  const int width = hidden_dim / groups;
  const int tid = threadIdx.x;
  const int64_t base_offset = static_cast<int64_t>(row) * hidden_dim + static_cast<int64_t>(group) * width;

  float value = 0.0f;
  if (tid < width) {
    value = x[base_offset + tid];
  }
  shared[tid] = (tid < width) ? value * value : 0.0f;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }
  const float rms = sqrtf(shared[0] / static_cast<float>(width) + eps);

  if (tid < width) {
    const float t = value / rms;
    float partials[kFastRlbMaxGrad];
    float f = 0.0f;
    float dfdt = 0.0f;
    fast_rlb_eval(t, group, groups, basis_count, coeff_limit, numerator, denominator, coeff_logits, centers, beta, &f, &dfdt, partials);
    y[base_offset + tid] = rms * f;
  }
}

__global__ void rational_local_basis_backward_kernel(const float* __restrict__ grad_output,
                                                     const float* __restrict__ x,
                                                     const float* __restrict__ numerator,
                                                     const float* __restrict__ denominator,
                                                     const float* __restrict__ coeff_logits,
                                                     const float* __restrict__ centers,
                                                     const float* __restrict__ beta,
                                                     float* __restrict__ grad_x,
                                                     float* __restrict__ grad_numerator,
                                                     float* __restrict__ grad_denominator,
                                                     float* __restrict__ grad_coeff_logits,
                                                     int64_t rows,
                                                     int hidden_dim,
                                                     int groups,
                                                     int basis_count,
                                                     float coeff_limit,
                                                     float eps) {
  __shared__ float shared[kThreads];
  __shared__ float grad_shared[kFastRlbMaxGrad][kThreads];

  const int row_group = blockIdx.x;
  const int group = row_group % groups;
  const int row = row_group / groups;
  const int width = hidden_dim / groups;
  const int tid = threadIdx.x;
  const int grad_count = kGradCount + 2 * basis_count;
  const int64_t base_offset = static_cast<int64_t>(row) * hidden_dim + static_cast<int64_t>(group) * width;

  float value = 0.0f;
  if (tid < width) {
    value = x[base_offset + tid];
  }
  shared[tid] = (tid < width) ? value * value : 0.0f;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }
  const float rms = sqrtf(shared[0] / static_cast<float>(width) + eps);

  float t = 0.0f;
  float gy = 0.0f;
  float f = 0.0f;
  float dfdt = 0.0f;
  float partials[kFastRlbMaxGrad];
#pragma unroll
  for (int idx = 0; idx < kFastRlbMaxGrad; ++idx) {
    partials[idx] = 0.0f;
  }

  if (tid < width) {
    t = value / rms;
    gy = grad_output[base_offset + tid];
    fast_rlb_eval(t, group, groups, basis_count, coeff_limit, numerator, denominator, coeff_logits, centers, beta, &f, &dfdt, partials);
  }

  shared[tid] = (tid < width) ? gy * (f - dfdt * t) : 0.0f;
  for (int idx = 0; idx < kFastRlbMaxGrad; ++idx) {
    grad_shared[idx][tid] = (tid < width && idx < grad_count) ? gy * rms * partials[idx] : 0.0f;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
#pragma unroll
      for (int idx = 0; idx < kFastRlbMaxGrad; ++idx) {
        grad_shared[idx][tid] += grad_shared[idx][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid < width) {
    grad_x[base_offset + tid] = gy * dfdt + (t / static_cast<float>(width)) * shared[0];
  }

  if (tid < grad_count) {
    const float gsum = grad_shared[tid][0];
    if (tid < kNumerator) {
      atomicAdd(grad_numerator + static_cast<int64_t>(group) * kNumerator + tid, gsum);
    } else if (tid < kGradCount) {
      atomicAdd(grad_denominator + static_cast<int64_t>(group) * kDenominator + (tid - kNumerator), gsum);
    } else {
      atomicAdd(grad_coeff_logits + static_cast<int64_t>(group) * basis_count * 2 + (tid - kGradCount), gsum);
    }
  }
}

int launch_blocks(int64_t n, int elements_per_thread = 1) {
  const int64_t work_per_block = static_cast<int64_t>(kThreads) * elements_per_thread;
  const int64_t blocks = (n + work_per_block - 1) / work_per_block;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(blocks, kMaxBlocks)));
}

}  // namespace

torch::Tensor rational_forward_cuda(torch::Tensor x,
                                    torch::Tensor numerator,
                                    torch::Tensor denominator) {
  const at::cuda::CUDAGuard device_guard(x.device());
  auto y = torch::empty_like(x);
  const int64_t n = x.numel();
  const int blocks = launch_blocks(n);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rational_forward_kernel<<<blocks, kThreads, 0, stream>>>(
      x.data_ptr<float>(),
      numerator.data_ptr<float>(),
      denominator.data_ptr<float>(),
      y.data_ptr<float>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor rational_version_a_forward_cuda(torch::Tensor x,
                                              torch::Tensor numerator,
                                              torch::Tensor denominator) {
  const at::cuda::CUDAGuard device_guard(x.device());
  auto y = torch::empty_like(x);
  const int64_t n = x.numel();
  const int blocks = launch_blocks(n);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rational_version_a_forward_kernel<<<blocks, kThreads, 0, stream>>>(
      x.data_ptr<float>(),
      numerator.data_ptr<float>(),
      denominator.data_ptr<float>(),
      y.data_ptr<float>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<torch::Tensor> rational_backward_cuda(torch::Tensor grad_output,
                                                  torch::Tensor x,
                                                  torch::Tensor numerator,
                                                  torch::Tensor denominator) {
  const at::cuda::CUDAGuard device_guard(x.device());
  auto grad_x = torch::empty_like(x);
  auto grad_numerator = torch::empty_like(numerator);
  auto grad_denominator = torch::empty_like(denominator);

  const int64_t n = x.numel();
  const int blocks = launch_blocks(n);
  auto partials = torch::empty({kGradCount, blocks}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rational_backward_stage1_kernel<<<blocks, kThreads, 0, stream>>>(
      grad_output.data_ptr<float>(),
      x.data_ptr<float>(),
      numerator.data_ptr<float>(),
      denominator.data_ptr<float>(),
      grad_x.data_ptr<float>(),
      partials.data_ptr<float>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  rational_backward_stage2_kernel<<<kGradCount, kThreads, 0, stream>>>(
      partials.data_ptr<float>(),
      grad_numerator.data_ptr<float>(),
      grad_denominator.data_ptr<float>(),
      blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_numerator, grad_denominator};
}

std::vector<torch::Tensor> rational_version_a_backward_cuda(torch::Tensor grad_output,
                                                            torch::Tensor x,
                                                            torch::Tensor numerator,
                                                            torch::Tensor denominator) {
  const at::cuda::CUDAGuard device_guard(x.device());
  auto grad_x = torch::empty_like(x);
  auto grad_numerator = torch::empty_like(numerator);
  auto grad_denominator = torch::empty_like(denominator);

  const int64_t n = x.numel();
  const int blocks = launch_blocks(n, 4);
  auto partials = torch::empty({kGradCount, blocks}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rational_version_a_backward_stage1_kernel<<<blocks, kThreads, 0, stream>>>(
      grad_output.data_ptr<float>(),
      x.data_ptr<float>(),
      numerator.data_ptr<float>(),
      denominator.data_ptr<float>(),
      grad_x.data_ptr<float>(),
      partials.data_ptr<float>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  rational_backward_stage2_kernel<<<kGradCount, kThreads, 0, stream>>>(
      partials.data_ptr<float>(),
      grad_numerator.data_ptr<float>(),
      grad_denominator.data_ptr<float>(),
      blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_numerator, grad_denominator};
}


torch::Tensor rational_local_basis_forward_cuda(torch::Tensor x,
                                                torch::Tensor numerator,
                                                torch::Tensor denominator,
                                                torch::Tensor coeff_logits,
                                                torch::Tensor centers,
                                                torch::Tensor beta,
                                                double coeff_limit,
                                                double eps,
                                                int64_t hidden_dim,
                                                int64_t groups) {
  const at::cuda::CUDAGuard device_guard(x.device());
  auto y = torch::empty_like(x);
  const int64_t rows = x.numel() / hidden_dim;
  const int64_t basis_count = centers.size(1);
  const int64_t blocks = rows * groups;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rational_local_basis_forward_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      x.data_ptr<float>(),
      numerator.data_ptr<float>(),
      denominator.data_ptr<float>(),
      coeff_logits.data_ptr<float>(),
      centers.data_ptr<float>(),
      beta.data_ptr<float>(),
      y.data_ptr<float>(),
      rows,
      static_cast<int>(hidden_dim),
      static_cast<int>(groups),
      static_cast<int>(basis_count),
      static_cast<float>(coeff_limit),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<torch::Tensor> rational_local_basis_backward_cuda(torch::Tensor grad_output,
                                                              torch::Tensor x,
                                                              torch::Tensor numerator,
                                                              torch::Tensor denominator,
                                                              torch::Tensor coeff_logits,
                                                              torch::Tensor centers,
                                                              torch::Tensor beta,
                                                              double coeff_limit,
                                                              double eps,
                                                              int64_t hidden_dim,
                                                              int64_t groups) {
  const at::cuda::CUDAGuard device_guard(x.device());
  auto grad_x = torch::empty_like(x);
  auto grad_numerator = torch::zeros_like(numerator);
  auto grad_denominator = torch::zeros_like(denominator);
  auto grad_coeff_logits = torch::zeros_like(coeff_logits);

  const int64_t rows = x.numel() / hidden_dim;
  const int64_t basis_count = centers.size(1);
  const int64_t blocks = rows * groups;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rational_local_basis_backward_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      grad_output.data_ptr<float>(),
      x.data_ptr<float>(),
      numerator.data_ptr<float>(),
      denominator.data_ptr<float>(),
      coeff_logits.data_ptr<float>(),
      centers.data_ptr<float>(),
      beta.data_ptr<float>(),
      grad_x.data_ptr<float>(),
      grad_numerator.data_ptr<float>(),
      grad_denominator.data_ptr<float>(),
      grad_coeff_logits.data_ptr<float>(),
      rows,
      static_cast<int>(hidden_dim),
      static_cast<int>(groups),
      static_cast<int>(basis_count),
      static_cast<float>(coeff_limit),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_numerator, grad_denominator, grad_coeff_logits};
}
