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

// Wartet nach den Kernelstarts eines Laufs auf das Gerät (cudaDeviceSynchronize) und wirft bei
// einem Fehler CudaError mit dem Namen des Laufs `what` und seinen Kenngrößen `context()` (nur im
// Fehlerfall aufgerufen, liefert std::string), damit ein Fehler, der erst hier gemeldet wird,
// eingeordnet werden kann. Fehler wie "an illegal memory access" sind dauerhaft (der
// CUDA-Kontext des Prozesses ist danach unbrauchbar) und können auch aus einem früheren
// asynchronen Kernelstart stammen; die Meldung sagt das dazu.
template <class Context>
void synchronize_checked(const char* what, const Context& context, const char* file, int line) {
    const cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        return;
    }
    // Dauerhaft ist der Fehler, wenn er nach dem Zurücksetzen erneut gemeldet wird.
    cudaGetLastError();
    const bool sticky =
        cudaPeekAtLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess;
    std::string message = std::string("CUDA-Fehler ") + file + ":" + std::to_string(line) +
                          ": cudaDeviceSynchronize() nach " + what + " (" + context() +
                          "): " + cudaGetErrorString(err);
    if (sticky) {
        message += " (dauerhafter Fehler: der CUDA-Kontext ist unbrauchbar, der Prozess muss neu "
                   "starten; Ursache kann auch ein früherer asynchroner Kernel sein)";
    }
    throw CudaError(message);
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
