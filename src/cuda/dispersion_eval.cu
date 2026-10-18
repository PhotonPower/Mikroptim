#include "mikroptim/cuda/dispersion_eval.hpp"
#include "mikroptim/glass/dispersion.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

namespace mko::cuda {

namespace {

// Koeffizienten als Wert-Parameter des Kernels (landen im Konstantenspeicher).
struct DispersionParams {
    double B[3];
    double C[3];
    double n0;
    double nu[3];
};

__global__ void dispersion_kernel(DispersionParams p, const double* lambda_um, int n,
                                  double* n_sellmeier, double* n_buchdahl) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const double l = lambda_um[i];
    n_sellmeier[i] = sellmeier_index(p.B, p.C, l);
    n_buchdahl[i] = buchdahl_index(p.n0, p.nu[0], p.nu[1], p.nu[2], l);
}

struct DeviceBuffers {
    double* lambda = nullptr;
    double* ns = nullptr;
    double* nb = nullptr;
    ~DeviceBuffers() {
        cudaFree(lambda);
        cudaFree(ns);
        cudaFree(nb);
    }
};

} // namespace

bool evaluate_dispersion(const std::array<double, 3>& B, const std::array<double, 3>& C, double n0,
                         const std::array<double, 3>& nu, const std::vector<double>& lambda_um,
                         std::vector<double>& n_sellmeier, std::vector<double>& n_buchdahl) {
    const int n = static_cast<int>(lambda_um.size());
    n_sellmeier.assign(lambda_um.size(), 0.0);
    n_buchdahl.assign(lambda_um.size(), 0.0);
    if (n == 0) {
        return true;
    }
    DispersionParams p{};
    for (int k = 0; k < 3; ++k) {
        p.B[k] = B[k];
        p.C[k] = C[k];
        p.nu[k] = nu[k];
    }
    p.n0 = n0;

    const std::size_t bytes = lambda_um.size() * sizeof(double);
    DeviceBuffers b;
    MKO_CUDA_TRY(cudaMalloc(&b.lambda, bytes));
    MKO_CUDA_TRY(cudaMalloc(&b.ns, bytes));
    MKO_CUDA_TRY(cudaMalloc(&b.nb, bytes));
    MKO_CUDA_TRY(cudaMemcpy(b.lambda, lambda_um.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int kBlock = 128;
    const int grid = (n + kBlock - 1) / kBlock;
    dispersion_kernel<<<grid, kBlock>>>(p, b.lambda, n, b.ns, b.nb);
    MKO_CUDA_TRY(cudaGetLastError());
    MKO_CUDA_TRY(cudaDeviceSynchronize());

    MKO_CUDA_TRY(cudaMemcpy(n_sellmeier.data(), b.ns, bytes, cudaMemcpyDeviceToHost));
    MKO_CUDA_TRY(cudaMemcpy(n_buchdahl.data(), b.nb, bytes, cudaMemcpyDeviceToHost));
    return true;
}

} // namespace mko::cuda
