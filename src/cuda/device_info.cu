#include "mikroptim/cuda/device_info.hpp"

#include "device_guard.hpp"

#include <cuda_runtime.h>

#include <cstdio>

namespace mko::cuda {

int device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError(); // Fehlerzustand zurücksetzen
        return 0;
    }
    return n;
}

bool supported_architecture(int major, int minor) {
    // Kernel liegen für sm_86 und sm_89 vor, PTX für compute_89 (ADR-0015): lauffähig ab sm_86
    return major * 10 + minor >= kMinComputeCapability;
}

int print_devices() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
        std::printf("CUDA nicht verfügbar: %s\n", cudaGetErrorString(err));
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) {
            continue;
        }
        // cudaDeviceProp::singleToDoublePrecisionPerfRatio gibt es seit CUDA 13 nicht mehr
        int ratio = 0;
        (void)cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, i);
        std::printf("GPU %d: %s, sm_%d%d, %.1f GiB, %d SMs, FP32:FP64 = %d:1\n", i, p.name, p.major,
                    p.minor, static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                    p.multiProcessorCount, ratio);
    }
    return n;
}

std::vector<int> usable_devices() {
    std::vector<int> devices;
    const int n = device_count();
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) == cudaSuccess &&
            supported_architecture(p.major, p.minor)) {
            devices.push_back(i);
        }
    }
    return devices;
}

DeviceMemoryInfo device_memory_info(int device) {
    const detail::DeviceGuard guard(device);
    DeviceMemoryInfo info;
    MKO_CUDA_CHECK(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
    return info;
}

void copy_device_to_host(void* host, const void* device, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    MKO_CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}

} // namespace mko::cuda
