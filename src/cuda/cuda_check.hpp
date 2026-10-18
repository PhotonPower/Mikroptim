#pragma once

#include "mikroptim/cuda/cuda_error.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

// Jeder CUDA-Aufruf wird geprüft (Regel in AGENTS.md). Gibt bei Fehler false zurück.
#define MKO_CUDA_TRY(call)                                                                         \
    do {                                                                                           \
        const cudaError_t mko_err_ = (call);                                                       \
        if (mko_err_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "CUDA-Fehler %s:%d: %s\n", __FILE__, __LINE__,                    \
                         cudaGetErrorString(mko_err_));                                            \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

namespace mko::cuda::detail {

[[noreturn]] inline void throw_cuda_error(cudaError_t err, const char* call, const char* file,
                                          int line) {
    cudaGetLastError(); // nicht dauerhafte Fehler zurücksetzen (sonst meldet sie der nächste
                        // Aufruf)
    throw CudaError(std::string("CUDA-Fehler ") + file + ":" + std::to_string(line) + ": " + call +
                    ": " + cudaGetErrorString(err));
}

} // namespace mko::cuda::detail

// Wie MKO_CUDA_TRY, wirft aber mko::cuda::CudaError (Host-Fehler beim Konfigurieren).
#define MKO_CUDA_CHECK(call)                                                                       \
    do {                                                                                           \
        const cudaError_t mko_err_ = (call);                                                       \
        if (mko_err_ != cudaSuccess) {                                                             \
            ::mko::cuda::detail::throw_cuda_error(mko_err_, #call, __FILE__, __LINE__);            \
        }                                                                                          \
    } while (0)
