#include "mikroptim/cuda/quadrature_check.hpp"
#include "mikroptim/pupil/quadrature.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

namespace mko::cuda {

namespace {

__global__ void pupil_samples_kernel(PupilQuadrature<double> q, double* rho, double* phi,
                                     double* weight) {
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= sample_count(q)) {
        return;
    }
    const PupilSample<double> s = sample(q, k);
    rho[k] = s.rho;
    phi[k] = s.phi;
    weight[k] = s.weight;
}

struct DeviceBuffers {
    double* rho = nullptr;
    double* phi = nullptr;
    double* weight = nullptr;
    ~DeviceBuffers() {
        cudaFree(rho);
        cudaFree(phi);
        cudaFree(weight);
    }
};

} // namespace

bool pupil_samples_gpu(const PupilQuadrature<double>& q, std::vector<double>& rho,
                       std::vector<double>& phi, std::vector<double>& weight) {
    const int n = sample_count(q);
    const std::size_t count = n > 0 ? static_cast<std::size_t>(n) : 0;
    rho.assign(count, 0.0);
    phi.assign(count, 0.0);
    weight.assign(count, 0.0);
    if (count == 0) {
        return true;
    }
    const std::size_t bytes = count * sizeof(double);
    DeviceBuffers b;
    MKO_CUDA_TRY(cudaMalloc(&b.rho, bytes));
    MKO_CUDA_TRY(cudaMalloc(&b.phi, bytes));
    MKO_CUDA_TRY(cudaMalloc(&b.weight, bytes));

    constexpr int kBlock = 128;
    const int grid = (n + kBlock - 1) / kBlock;
    pupil_samples_kernel<<<grid, kBlock>>>(q, b.rho, b.phi, b.weight);
    MKO_CUDA_TRY(cudaGetLastError());
    MKO_CUDA_TRY(cudaDeviceSynchronize());

    MKO_CUDA_TRY(cudaMemcpy(rho.data(), b.rho, bytes, cudaMemcpyDeviceToHost));
    MKO_CUDA_TRY(cudaMemcpy(phi.data(), b.phi, bytes, cudaMemcpyDeviceToHost));
    MKO_CUDA_TRY(cudaMemcpy(weight.data(), b.weight, bytes, cudaMemcpyDeviceToHost));
    return true;
}

} // namespace mko::cuda
