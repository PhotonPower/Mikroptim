#include "mikroptim/cuda/device_info.hpp"

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
        std::printf("GPU %d: %s, sm_%d%d, %.1f GiB, %d SMs, FP32:FP64 = %d:1\n", i, p.name,
                    p.major, p.minor,
                    static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                    p.multiProcessorCount, p.singleToDoublePrecisionPerfRatio);
    }
    return n;
}

} // namespace mko::cuda
