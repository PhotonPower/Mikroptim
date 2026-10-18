#pragma once

// Setzt für die Lebensdauer des Objekts das aktuelle CUDA-Gerät und stellt danach das vorige
// wieder her (mehrere GPUs, M7: Aufrufer sollen ihr aktuelles Gerät nicht verlieren).

#include "cuda_check.hpp"

#include <cuda_runtime.h>

namespace mko::cuda::detail {

class DeviceGuard {
public:
    // device < 0: aktuelles Gerät beibehalten. Wirft CudaError.
    explicit DeviceGuard(int device) {
        MKO_CUDA_CHECK(cudaGetDevice(&previous_));
        if (device >= 0 && device != previous_) {
            MKO_CUDA_CHECK(cudaSetDevice(device));
            changed_ = true;
        }
    }
    ~DeviceGuard() {
        if (changed_) {
            cudaSetDevice(previous_); // im Destruktor nicht werfen
        }
    }
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_ = 0;
    bool changed_ = false;
};

// Aktuelles Gerät (wirft CudaError).
inline int current_device() {
    int device = 0;
    MKO_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

} // namespace mko::cuda::detail
