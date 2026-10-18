#pragma once

#include <cuda_runtime.h>

#include <cstdio>

// Jeder CUDA-Aufruf wird geprüft (Regel in AGENTS.md). Gibt bei Fehler false zurück.
#define MKO_CUDA_TRY(call)                                                                    \
    do {                                                                                       \
        const cudaError_t mko_err_ = (call);                                                   \
        if (mko_err_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA-Fehler %s:%d: %s\n", __FILE__, __LINE__,                \
                         cudaGetErrorString(mko_err_));                                        \
            return false;                                                                      \
        }                                                                                      \
    } while (0)
