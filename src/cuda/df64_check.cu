// Prüfkernel für den Zahlentyp df64 (M7-T1): dieselben Operationen wie auf dem Host
// (df64_check_ops), Vergleich im Test CudaDf64.DeviceMatchesHostBitwise.

#include "mikroptim/cuda/df64_check.hpp"
#include "mikroptim/math/df64.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace mko::cuda {

namespace {

__global__ void df64_check_kernel(const df64* a, const df64* b, const double* x, int n, df64* out,
                                  double* as_double) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    df64_check_ops(a[i], b[i], x[i], out + static_cast<std::size_t>(i) * kDf64CheckOps,
                   as_double + i);
}

struct DeviceBuffers {
    df64* a = nullptr;
    df64* b = nullptr;
    double* x = nullptr;
    df64* out = nullptr;
    double* as_double = nullptr;
    ~DeviceBuffers() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(x);
        cudaFree(out);
        cudaFree(as_double);
    }
};

} // namespace

bool df64_check_ops_gpu(const std::vector<df64>& a, const std::vector<df64>& b,
                        const std::vector<double>& x, std::vector<df64>& out,
                        std::vector<double>& as_double) {
    if (b.size() != a.size() || x.size() != a.size()) {
        return false;
    }
    const std::size_t n = a.size();
    out.assign(n * kDf64CheckOps, df64(0));
    as_double.assign(n, 0.0);
    if (n == 0) {
        return true;
    }
    DeviceBuffers d;
    MKO_CUDA_TRY(cudaMalloc(&d.a, n * sizeof(df64)));
    MKO_CUDA_TRY(cudaMalloc(&d.b, n * sizeof(df64)));
    MKO_CUDA_TRY(cudaMalloc(&d.x, n * sizeof(double)));
    MKO_CUDA_TRY(cudaMalloc(&d.out, out.size() * sizeof(df64)));
    MKO_CUDA_TRY(cudaMalloc(&d.as_double, n * sizeof(double)));
    MKO_CUDA_TRY(cudaMemcpy(d.a, a.data(), n * sizeof(df64), cudaMemcpyHostToDevice));
    MKO_CUDA_TRY(cudaMemcpy(d.b, b.data(), n * sizeof(df64), cudaMemcpyHostToDevice));
    MKO_CUDA_TRY(cudaMemcpy(d.x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int kBlock = 128;
    const int count = static_cast<int>(n);
    const int grid = (count + kBlock - 1) / kBlock;
    df64_check_kernel<<<grid, kBlock>>>(d.a, d.b, d.x, count, d.out, d.as_double);
    MKO_CUDA_TRY(cudaGetLastError());
    MKO_CUDA_TRY(cudaDeviceSynchronize());

    MKO_CUDA_TRY(cudaMemcpy(out.data(), d.out, out.size() * sizeof(df64), cudaMemcpyDeviceToHost));
    MKO_CUDA_TRY(
        cudaMemcpy(as_double.data(), d.as_double, n * sizeof(double), cudaMemcpyDeviceToHost));
    return true;
}

} // namespace mko::cuda
