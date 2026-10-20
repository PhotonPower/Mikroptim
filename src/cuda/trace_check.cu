#include "mikroptim/cuda/trace_check.hpp"
#include "mikroptim/trace/trace.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

namespace mko::cuda {

namespace {

// Bytes der Flächentabelle im Shared Memory: curvature[S], thickness[S], index[S·L] (FP64),
// danach type[S] (1 Byte).
std::size_t shared_bytes(int surfaces, int wavelengths) {
    const std::size_t s = static_cast<std::size_t>(surfaces);
    const std::size_t l = static_cast<std::size_t>(wavelengths);
    return (2 * s + s * l) * sizeof(double) + s * sizeof(SurfaceKind);
}

__global__ void trace_kernel(SurfaceTableView<double> global, int wavelength_index,
                             const Ray<double>* rays_in, int n, Ray<double>* rays_out,
                             unsigned* status, int* failed_surface) {
    extern __shared__ double smem[];
    const int S = global.surfaces;
    const int L = global.wavelengths;
    double* curvature = smem;
    double* thickness = curvature + S;
    double* index = thickness + S;
    SurfaceKind* type = reinterpret_cast<SurfaceKind*>(index + S * L);
    for (int i = static_cast<int>(threadIdx.x); i < S; i += static_cast<int>(blockDim.x)) {
        curvature[i] = global.curvature[i];
        thickness[i] = global.thickness[i];
        type[i] = global.type[i];
    }
    for (int i = static_cast<int>(threadIdx.x); i < S * L; i += static_cast<int>(blockDim.x)) {
        index[i] = global.index[i];
    }
    __syncthreads();

    SurfaceTableView<double> sys{S, L, type, curvature, thickness, index};
    sys.decenter_tilt = global.decenter_tilt; // M8-T3: aus dem globalen Speicher
    sys.tilt_trig = global.tilt_trig;
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k < n) {
        Ray<double> ray = rays_in[k];
        int failed = -1;
        const TraceStatus st = trace_ray(sys, wavelength_index, ray, nullptr, &failed);
        rays_out[k] = ray;
        status[k] = static_cast<unsigned>(st);
        failed_surface[k] = failed;
    }
}

struct DeviceBuffers {
    SurfaceKind* type = nullptr;
    double* curvature = nullptr;
    double* thickness = nullptr;
    double* index = nullptr;
    Ray<double>* in = nullptr;
    Ray<double>* out = nullptr;
    unsigned* status = nullptr;
    int* failed = nullptr;
    double* decenter_tilt = nullptr;
    double* tilt_trig = nullptr;
    ~DeviceBuffers() {
        cudaFree(decenter_tilt);
        cudaFree(tilt_trig);
        cudaFree(type);
        cudaFree(curvature);
        cudaFree(thickness);
        cudaFree(index);
        cudaFree(in);
        cudaFree(out);
        cudaFree(status);
        cudaFree(failed);
    }
};

} // namespace

bool trace_rays_gpu(const SurfaceTableView<double>& table, int wavelength_index,
                    const std::vector<Ray<double>>& rays_in, std::vector<Ray<double>>& rays_out,
                    std::vector<unsigned>& status, std::vector<int>& failed_surface) {
    const std::size_t count = rays_in.size();
    rays_out.assign(count, Ray<double>{});
    status.assign(count, 0u);
    failed_surface.assign(count, -1);
    if (count == 0) {
        return true;
    }
    const std::size_t smem = shared_bytes(table.surfaces, table.wavelengths);
    if (table.surfaces < 2 || table.wavelengths < 1 || smem > 48 * 1024) {
        std::fprintf(stderr, "trace_rays_gpu: Flächentabelle ungültig oder zu groß\n");
        return false;
    }
    const std::size_t S = static_cast<std::size_t>(table.surfaces);
    const std::size_t SL = S * static_cast<std::size_t>(table.wavelengths);

    DeviceBuffers b;
    MKO_CUDA_TRY(cudaMalloc(&b.type, S * sizeof(SurfaceKind)));
    MKO_CUDA_TRY(cudaMalloc(&b.curvature, S * sizeof(double)));
    MKO_CUDA_TRY(cudaMalloc(&b.thickness, S * sizeof(double)));
    MKO_CUDA_TRY(cudaMalloc(&b.index, SL * sizeof(double)));
    MKO_CUDA_TRY(cudaMalloc(&b.in, count * sizeof(Ray<double>)));
    MKO_CUDA_TRY(cudaMalloc(&b.out, count * sizeof(Ray<double>)));
    MKO_CUDA_TRY(cudaMalloc(&b.status, count * sizeof(unsigned)));
    MKO_CUDA_TRY(cudaMalloc(&b.failed, count * sizeof(int)));
    MKO_CUDA_TRY(cudaMemcpy(b.type, table.type, S * sizeof(SurfaceKind), cudaMemcpyHostToDevice));
    MKO_CUDA_TRY(
        cudaMemcpy(b.curvature, table.curvature, S * sizeof(double), cudaMemcpyHostToDevice));
    MKO_CUDA_TRY(
        cudaMemcpy(b.thickness, table.thickness, S * sizeof(double), cudaMemcpyHostToDevice));
    MKO_CUDA_TRY(cudaMemcpy(b.index, table.index, SL * sizeof(double), cudaMemcpyHostToDevice));
    MKO_CUDA_TRY(
        cudaMemcpy(b.in, rays_in.data(), count * sizeof(Ray<double>), cudaMemcpyHostToDevice));

    SurfaceTableView<double> device_table{table.surfaces, table.wavelengths, b.type,
                                          b.curvature,    b.thickness,       b.index};
    // Dezentrierung und Verkippung (M8-T3)
    const std::size_t X = S * static_cast<std::size_t>(kSurfaceTransformValues);
    if (table.decenter_tilt != nullptr) {
        MKO_CUDA_TRY(cudaMalloc(&b.decenter_tilt, X * sizeof(double)));
        MKO_CUDA_TRY(cudaMemcpy(b.decenter_tilt, table.decenter_tilt, X * sizeof(double),
                                cudaMemcpyHostToDevice));
        device_table.decenter_tilt = b.decenter_tilt;
        if (table.tilt_trig != nullptr) {
            MKO_CUDA_TRY(cudaMalloc(&b.tilt_trig, X * sizeof(double)));
            MKO_CUDA_TRY(cudaMemcpy(b.tilt_trig, table.tilt_trig, X * sizeof(double),
                                    cudaMemcpyHostToDevice));
            device_table.tilt_trig = b.tilt_trig;
        }
    }
    const int n = static_cast<int>(count);
    constexpr int kBlock = 128;
    const int grid = (n + kBlock - 1) / kBlock;
    trace_kernel<<<grid, kBlock, smem>>>(device_table, wavelength_index, b.in, n, b.out, b.status,
                                         b.failed);
    MKO_CUDA_TRY(cudaGetLastError());
    MKO_CUDA_TRY(cudaDeviceSynchronize());

    MKO_CUDA_TRY(
        cudaMemcpy(rays_out.data(), b.out, count * sizeof(Ray<double>), cudaMemcpyDeviceToHost));
    MKO_CUDA_TRY(
        cudaMemcpy(status.data(), b.status, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    MKO_CUDA_TRY(
        cudaMemcpy(failed_surface.data(), b.failed, count * sizeof(int), cudaMemcpyDeviceToHost));
    return true;
}

} // namespace mko::cuda
