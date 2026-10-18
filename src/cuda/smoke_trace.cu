#include "mikroptim/cuda/smoke_trace.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

namespace mko::cuda {

namespace {

__global__ void smoke_kernel(const double* heights, int n, double c, double mu,
                             double* axis_crossing, unsigned* status) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    Ray<double> r{{0.0, heights[i], 0.0}, {0.0, 0.0, 1.0}, 0.0};
    double s = 0.0;
    TraceStatus st = intersect_sphere(r, c, s);
    if (st == TraceStatus::Ok) {
        st = refract_sphere(r, c, mu);
    }
    status[i] = static_cast<unsigned>(st);
    axis_crossing[i] = (st == TraceStatus::Ok) ? meridional_axis_crossing(r) : 0.0;
}

struct DeviceBuffers {
    double* h = nullptr;
    double* z = nullptr;
    unsigned* s = nullptr;
    ~DeviceBuffers() {
        cudaFree(h);
        cudaFree(z);
        cudaFree(s);
    }
};

} // namespace

bool smoke_trace(const std::vector<double>& heights, double curvature, double mu,
                 std::vector<double>& axis_crossing, std::vector<unsigned>& status) {
    const int n = static_cast<int>(heights.size());
    axis_crossing.assign(heights.size(), 0.0);
    status.assign(heights.size(), 0u);
    if (n == 0) {
        return true;
    }
    DeviceBuffers b;
    MKO_CUDA_TRY(cudaMalloc(&b.h, heights.size() * sizeof(double)));
    MKO_CUDA_TRY(cudaMalloc(&b.z, heights.size() * sizeof(double)));
    MKO_CUDA_TRY(cudaMalloc(&b.s, heights.size() * sizeof(unsigned)));
    MKO_CUDA_TRY(cudaMemcpy(b.h, heights.data(), heights.size() * sizeof(double),
                            cudaMemcpyHostToDevice));

    constexpr int kBlock = 128;
    const int grid = (n + kBlock - 1) / kBlock;
    smoke_kernel<<<grid, kBlock>>>(b.h, n, curvature, mu, b.z, b.s);
    MKO_CUDA_TRY(cudaGetLastError());
    MKO_CUDA_TRY(cudaDeviceSynchronize());

    MKO_CUDA_TRY(cudaMemcpy(axis_crossing.data(), b.z, heights.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
    MKO_CUDA_TRY(cudaMemcpy(status.data(), b.s, heights.size() * sizeof(unsigned),
                            cudaMemcpyDeviceToHost));
    return true;
}

} // namespace mko::cuda
