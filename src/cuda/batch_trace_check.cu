#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/batch_trace_check.hpp"
#include "mikroptim/trace/trace.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace mko::cuda {

namespace {

// Ein Thread pro Strahl, Schleife mit Gitterschritt; Index ((v · F + f) · L + l) · N + k.
__global__ void batch_trace_kernel(SystemBatchView<double> batch, PupilQuadrature<double> pupil,
                                   long long total, Ray<double>* rays, unsigned* status,
                                   int* failed_surface) {
    const long long n = sample_count(pupil);
    const long long L = batch.shape.wavelengths;
    const long long F = batch.shape.fields;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const int k = static_cast<int>(i % n);
        const long long block = i / n; // (v · F + f) · L + l
        const int l = static_cast<int>(block % L);
        const int f = static_cast<int>((block / L) % F);
        const int v = static_cast<int>(block / (L * F));
        Ray<double> ray = batch_source_ray(batch, v, f, l, sample(pupil, k));
        int failed = -1;
        const TraceStatus st = trace_ray(batch.surface_table(v), l, ray, nullptr, &failed);
        rays[i] = ray;
        status[i] = static_cast<unsigned>(st);
        failed_surface[i] = failed;
    }
}

struct DeviceBuffers {
    Ray<double>* rays = nullptr;
    unsigned* status = nullptr;
    int* failed = nullptr;
    ~DeviceBuffers() {
        cudaFree(rays);
        cudaFree(status);
        cudaFree(failed);
    }
};

} // namespace

void trace_batch_gpu(const DeviceSystemBatch& batch, const PupilQuadrature<double>& pupil,
                     std::vector<Ray<double>>& rays, std::vector<unsigned>& status,
                     std::vector<int>& failed_surface) {
    const SystemBatchView<double> view = batch.view(); // wirft bei leerem Batch
    const std::size_t total = static_cast<std::size_t>(view.shape.variants) *
                              static_cast<std::size_t>(view.shape.fields) *
                              static_cast<std::size_t>(view.shape.wavelengths) *
                              static_cast<std::size_t>(sample_count(pupil));
    rays.assign(total, Ray<double>{});
    status.assign(total, 0u);
    failed_surface.assign(total, -1);
    if (total == 0) {
        return;
    }
    const detail::DeviceGuard guard(batch.device());
    DeviceBuffers b;
    MKO_CUDA_CHECK(cudaMalloc(&b.rays, total * sizeof(Ray<double>)));
    MKO_CUDA_CHECK(cudaMalloc(&b.status, total * sizeof(unsigned)));
    MKO_CUDA_CHECK(cudaMalloc(&b.failed, total * sizeof(int)));

    constexpr int kBlock = 128;
    const std::size_t blocks = (total + kBlock - 1) / kBlock;
    const int grid = static_cast<int>(std::min<std::size_t>(blocks, 65535));
    batch_trace_kernel<<<grid, kBlock>>>(view, pupil, static_cast<long long>(total), b.rays,
                                         b.status, b.failed);
    MKO_CUDA_CHECK(cudaGetLastError());
    MKO_CUDA_CHECK(cudaDeviceSynchronize());

    MKO_CUDA_CHECK(
        cudaMemcpy(rays.data(), b.rays, total * sizeof(Ray<double>), cudaMemcpyDeviceToHost));
    MKO_CUDA_CHECK(
        cudaMemcpy(status.data(), b.status, total * sizeof(unsigned), cudaMemcpyDeviceToHost));
    MKO_CUDA_CHECK(
        cudaMemcpy(failed_surface.data(), b.failed, total * sizeof(int), cudaMemcpyDeviceToHost));
}

} // namespace mko::cuda
