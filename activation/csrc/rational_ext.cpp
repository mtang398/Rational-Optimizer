#include <torch/extension.h>

#include <vector>

namespace {

constexpr int64_t kNumerator = 6;
constexpr int64_t kDenominator = 4;

void check_cuda_float_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_coeff(const torch::Tensor& tensor,
                 const char* name,
                 int64_t expected_numel,
                 const torch::Tensor& x) {
  check_cuda_float_contiguous(tensor, name);
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
  TORCH_CHECK(tensor.numel() == expected_numel,
              name,
              " must have ",
              expected_numel,
              " coefficients");
  TORCH_CHECK(tensor.get_device() == x.get_device(),
              name,
              " must be on the same CUDA device as x");
}

void check_matrix(const torch::Tensor& tensor,
                  const char* name,
                  int64_t dim0,
                  int64_t dim1,
                  const torch::Tensor& x) {
  check_cuda_float_contiguous(tensor, name);
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(tensor.size(0) == dim0 && tensor.size(1) == dim1,
              name,
              " must have shape [",
              dim0,
              ", ",
              dim1,
              "]");
  TORCH_CHECK(tensor.get_device() == x.get_device(),
              name,
              " must be on the same CUDA device as x");
}

void check_tensor3(const torch::Tensor& tensor,
                   const char* name,
                   int64_t dim0,
                   int64_t dim1,
                   int64_t dim2,
                   const torch::Tensor& x) {
  check_cuda_float_contiguous(tensor, name);
  TORCH_CHECK(tensor.dim() == 3, name, " must be 3-D");
  TORCH_CHECK(tensor.size(0) == dim0 && tensor.size(1) == dim1 && tensor.size(2) == dim2,
              name,
              " must have shape [",
              dim0,
              ", ",
              dim1,
              ", ",
              dim2,
              "]");
  TORCH_CHECK(tensor.get_device() == x.get_device(),
              name,
              " must be on the same CUDA device as x");
}

}  // namespace

torch::Tensor rational_forward_cuda(torch::Tensor x,
                                    torch::Tensor numerator,
                                    torch::Tensor denominator);

std::vector<torch::Tensor> rational_backward_cuda(torch::Tensor grad_output,
                                                  torch::Tensor x,
                                                  torch::Tensor numerator,
                                                  torch::Tensor denominator);

torch::Tensor rational_version_a_forward_cuda(torch::Tensor x,
                                              torch::Tensor numerator,
                                              torch::Tensor denominator);

std::vector<torch::Tensor> rational_version_a_backward_cuda(torch::Tensor grad_output,
                                                            torch::Tensor x,
                                                            torch::Tensor numerator,
                                                            torch::Tensor denominator);

torch::Tensor rational_local_basis_forward_cuda(torch::Tensor x,
                                                torch::Tensor numerator,
                                                torch::Tensor denominator,
                                                torch::Tensor coeff_logits,
                                                torch::Tensor centers,
                                                torch::Tensor beta,
                                                double coeff_limit,
                                                double eps,
                                                int64_t hidden_dim,
                                                int64_t groups);

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
                                                              int64_t groups);

torch::Tensor rational_forward(torch::Tensor x,
                               torch::Tensor numerator,
                               torch::Tensor denominator) {
  check_cuda_float_contiguous(x, "x");
  check_coeff(numerator, "numerator", kNumerator, x);
  check_coeff(denominator, "denominator", kDenominator, x);
  return rational_forward_cuda(x, numerator, denominator);
}

std::vector<torch::Tensor> rational_backward(torch::Tensor grad_output,
                                             torch::Tensor x,
                                             torch::Tensor numerator,
                                             torch::Tensor denominator) {
  check_cuda_float_contiguous(grad_output, "grad_output");
  check_cuda_float_contiguous(x, "x");
  TORCH_CHECK(grad_output.numel() == x.numel(),
              "grad_output and x must contain the same number of elements");
  check_coeff(numerator, "numerator", kNumerator, x);
  check_coeff(denominator, "denominator", kDenominator, x);
  return rational_backward_cuda(grad_output, x, numerator, denominator);
}

torch::Tensor rational_version_a_forward(torch::Tensor x,
                                         torch::Tensor numerator,
                                         torch::Tensor denominator) {
  check_cuda_float_contiguous(x, "x");
  check_coeff(numerator, "numerator", kNumerator, x);
  check_coeff(denominator, "denominator", kDenominator, x);
  return rational_version_a_forward_cuda(x, numerator, denominator);
}

std::vector<torch::Tensor> rational_version_a_backward(torch::Tensor grad_output,
                                                       torch::Tensor x,
                                                       torch::Tensor numerator,
                                                       torch::Tensor denominator) {
  check_cuda_float_contiguous(grad_output, "grad_output");
  check_cuda_float_contiguous(x, "x");
  TORCH_CHECK(grad_output.numel() == x.numel(),
              "grad_output and x must contain the same number of elements");
  check_coeff(numerator, "numerator", kNumerator, x);
  check_coeff(denominator, "denominator", kDenominator, x);
  return rational_version_a_backward_cuda(grad_output, x, numerator, denominator);
}

torch::Tensor rational_local_basis_forward(torch::Tensor x,
                                           torch::Tensor numerator,
                                           torch::Tensor denominator,
                                           torch::Tensor coeff_logits,
                                           torch::Tensor centers,
                                           torch::Tensor beta,
                                           double coeff_limit,
                                           double eps,
                                           int64_t hidden_dim,
                                           int64_t groups) {
  check_cuda_float_contiguous(x, "x");
  TORCH_CHECK(hidden_dim > 0, "hidden_dim must be positive");
  TORCH_CHECK(groups > 0, "groups must be positive");
  TORCH_CHECK(hidden_dim % groups == 0, "hidden_dim must be divisible by groups");
  TORCH_CHECK(x.numel() % hidden_dim == 0, "x.numel() must be divisible by hidden_dim");
  const int64_t width = hidden_dim / groups;
  TORCH_CHECK(width <= 256, "fused local-basis rational supports group width <= 256");
  TORCH_CHECK(x.size(-1) == hidden_dim, "x last dimension must equal hidden_dim");
  check_matrix(numerator, "numerator", groups, kNumerator, x);
  check_matrix(denominator, "denominator", groups, kDenominator, x);
  TORCH_CHECK(centers.dim() == 2, "centers must be 2-D");
  const int64_t basis_count = centers.size(1);
  TORCH_CHECK(basis_count > 0 && basis_count <= 4, "basis_count must be in [1, 4]");
  check_matrix(centers, "centers", groups, basis_count, x);
  check_matrix(beta, "beta", groups, basis_count, x);
  check_tensor3(coeff_logits, "coeff_logits", groups, basis_count, 2, x);
  return rational_local_basis_forward_cuda(
      x, numerator, denominator, coeff_logits, centers, beta, coeff_limit, eps, hidden_dim, groups);
}

std::vector<torch::Tensor> rational_local_basis_backward(torch::Tensor grad_output,
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
  check_cuda_float_contiguous(grad_output, "grad_output");
  check_cuda_float_contiguous(x, "x");
  TORCH_CHECK(grad_output.numel() == x.numel(),
              "grad_output and x must contain the same number of elements");
  TORCH_CHECK(hidden_dim > 0, "hidden_dim must be positive");
  TORCH_CHECK(groups > 0, "groups must be positive");
  TORCH_CHECK(hidden_dim % groups == 0, "hidden_dim must be divisible by groups");
  TORCH_CHECK(x.numel() % hidden_dim == 0, "x.numel() must be divisible by hidden_dim");
  const int64_t width = hidden_dim / groups;
  TORCH_CHECK(width <= 256, "fused local-basis rational supports group width <= 256");
  TORCH_CHECK(x.size(-1) == hidden_dim, "x last dimension must equal hidden_dim");
  check_matrix(numerator, "numerator", groups, kNumerator, x);
  check_matrix(denominator, "denominator", groups, kDenominator, x);
  TORCH_CHECK(centers.dim() == 2, "centers must be 2-D");
  const int64_t basis_count = centers.size(1);
  TORCH_CHECK(basis_count > 0 && basis_count <= 4, "basis_count must be in [1, 4]");
  check_matrix(centers, "centers", groups, basis_count, x);
  check_matrix(beta, "beta", groups, basis_count, x);
  check_tensor3(coeff_logits, "coeff_logits", groups, basis_count, 2, x);
  return rational_local_basis_backward_cuda(
      grad_output, x, numerator, denominator, coeff_logits, centers, beta, coeff_limit, eps, hidden_dim, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rational_forward, "Rational 5/4 CUDA forward");
  m.def("backward", &rational_backward, "Rational 5/4 CUDA backward");
  m.def("version_a_forward", &rational_version_a_forward, "Rational Version A 5/4 CUDA forward");
  m.def("version_a_backward", &rational_version_a_backward, "Rational Version A 5/4 CUDA backward");
  m.def("local_basis_forward", &rational_local_basis_forward, "Fused rational local-basis CUDA forward");
  m.def("local_basis_backward", &rational_local_basis_backward, "Fused rational local-basis CUDA backward");
}
