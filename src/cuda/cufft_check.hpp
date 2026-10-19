#pragma once

// Prüfung der cuFFT-Aufrufe (intern, nur in .cu-Dateien): cufftResult → CudaError mit dem Namen
// des Fehlers (wie MKO_CUDA_CHECK). Gemeinsam für PsfEngine (M8-T2) und VectorPsfEngine (M8-T5).

#include "mikroptim/cuda/cuda_error.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <string>

namespace mko::cuda::detail {

inline const char* cufft_error_name(cufftResult r) {
    switch (r) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED";
    default:
        return "unbekannter cuFFT-Fehler";
    }
}

[[noreturn]] inline void throw_cufft_error(cufftResult r, const char* call, const char* file,
                                           int line) {
    cudaGetLastError(); // nicht dauerhafte CUDA-Fehler zurücksetzen
    throw CudaError(std::string("cuFFT-Fehler ") + file + ":" + std::to_string(line) + ": " + call +
                    ": " + cufft_error_name(r) + " (" + std::to_string(static_cast<int>(r)) + ")");
}

} // namespace mko::cuda::detail

// Jeder cuFFT-Aufruf wird geprüft (wie MKO_CUDA_CHECK, wirft CudaError).
#define MKO_CUFFT_CHECK(call)                                                                      \
    do {                                                                                           \
        const cufftResult mko_fft_err_ = (call);                                                   \
        if (mko_fft_err_ != CUFFT_SUCCESS) {                                                       \
            ::mko::cuda::detail::throw_cufft_error(mko_fft_err_, #call, __FILE__, __LINE__);       \
        }                                                                                          \
    } while (0)
