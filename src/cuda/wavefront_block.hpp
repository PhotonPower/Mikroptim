#pragma once

// Gemeinsame Bausteine der Kernel RESIDUALS (M2-T2) und MERIT (M2-T3): Block-Abbildung,
// Flächentabelle im Shared Memory, deterministische Reduktionen (ADR-0007) und die zweite
// Stufe der Max-Reduktionen. Nur aus .cu-Dateien einbinden. Beide Kernel rufen dieselben
// Funktionen in derselben Reihenfolge auf; deshalb sind Kolben, OPD und Residuen in beiden
// Modi bitgleich (Test CudaMerit.MatchesSumOfResiduals). docs/architektur.md, Abschnitte
// "Kernel RESIDUALS" und "Kernel MERIT".

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mko::cuda::detail {

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = kResidualThreads;
constexpr int kBlockWarps = kBlockThreads / kWarpSize;
static_assert(kBlockThreads % kWarpSize == 0, "Blockgröße muss ein Vielfaches von 32 sein");
static_assert(kMaxPupilRings <= kBlockThreads, "Knoten werden von je einem Thread geladen");

// Threads der zweiten Reduktionsstufe (ein Thread pro Variante bzw. pro Variante und Fläche).
constexpr int kFinalizeThreads = 256;

// Summe über die 32 Lanes eines Warps als feste Baumreduktion (Abstände 16, 8, 4, 2, 1);
// das Ergebnis steht in Lane 0. Alle Lanes müssen teilnehmen.
template <class T>
__device__ inline T warp_sum(T x) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

// Maximum bzw. Minimum über die Lanes eines Warps (Ergebnis in Lane 0), gleiche Struktur.
template <class R>
__device__ inline R warp_max(R x) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        x = extremum_max(x, __shfl_down_sync(0xffffffffu, x, offset));
    }
    return x;
}
template <class R>
__device__ inline R warp_min(R x) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        x = extremum_min(x, __shfl_down_sync(0xffffffffu, x, offset));
    }
    return x;
}

// Tupel des Blocks b = (v · F + f) · L + l (Indexrechnung in 64 Bit).
struct BlockTuple {
    int v;
    int f;
    int l;
    std::int64_t block;
};

__device__ inline BlockTuple block_tuple(int fields, int wavelengths) {
    const std::int64_t block = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t vf = block / wavelengths;
    BlockTuple t;
    t.l = static_cast<int>(block % wavelengths);
    t.f = static_cast<int>(vf % fields);
    t.v = static_cast<int>(vf / fields);
    t.block = block;
    return t;
}

// Flächentabelle der Variante mit den Brechzahlen nur der Wellenlänge l des Blocks und die
// Quadratur des Feldes (Shared Memory, statisch bis kMaxSurfaces).
template <class R>
struct BlockTable {
    SurfaceKind type[kMaxSurfaces];
    R curvature[kMaxSurfaces];
    R thickness[kMaxSurfaces];
    R index[kMaxSurfaces];
    PupilQuadrature<R> pupil;
};

// Lädt Tabelle und Quadratur (koaleszierend über s; Quadratur wie
// make_pupil_quadrature(N, M, plan.mode)). Der Aufrufer synchronisiert danach.
template <class R>
__device__ inline void
load_block_table(BlockTable<R>& t, const SystemBatchView<R>& b, const PupilQuadrature<R>& pupil,
                 const ResidualFieldPlan& plan, const BlockTuple& tu, int tid) {
    const int S = b.shape.surfaces;
    for (int s = tid; s < S; s += kBlockThreads) {
        t.type[s] = b.type[s];
        t.curvature[s] = b.curvature[b.shape.surface_at(tu.v, s)];
        t.thickness[s] = b.thickness[b.shape.surface_at(tu.v, s)];
        t.index[s] = b.index[b.shape.index_at(tu.v, s, tu.l)];
    }
    if (tid < pupil.rings) {
        t.pupil.u[tid] = pupil.u[tid];
        t.pupil.w[tid] = pupil.w[tid];
    }
    if (tid == 0) {
        t.pupil.rings = pupil.rings;
        t.pupil.spokes = plan.mode == PupilMode::Axis ? 1 : pupil.spokes;
        t.pupil.mode = plan.mode;
    }
}

// Geisterplatten der Variante (M4-T5) im dynamischen Shared Memory: Koeffizienten [G·K],
// Brechzahlen der Gläser bei der Wellenlänge des Blocks [G], Normierungsradien [G]. Höchstens
// (kMaxSurfaces · kMaxGhostTerms + 2 · kMaxSurfaces) · 8 B = 17 KiB.
constexpr std::size_t kMaxGhostSharedBytes =
    (static_cast<std::size_t>(kMaxSurfaces) * kMaxGhostTerms + 2 * kMaxSurfaces) * sizeof(double);
static_assert(kMaxGhostSharedBytes + 8192 <= 48 * 1024, "Shared Memory pro Block über 48 KB");

// Bytes der Geisterdaten im dynamischen Shared Memory (R = double; Vielfaches von 8).
inline std::size_t ghost_shared_bytes(const BatchShape& shape) {
    const auto G = static_cast<std::size_t>(shape.ghost_plates);
    const auto K = static_cast<std::size_t>(shape.ghost_terms);
    return (G * K + 2 * G) * sizeof(double);
}

template <class R>
struct BlockGhosts {
    int plates;
    int terms;
    R* coeff;  // [G · K] oder nullptr
    R* index;  // [G]
    R* radius; // [G]
};

// Lädt die Geisterdaten der Variante (Brechzahlen der Wellenlänge l) an den Anfang `dynamic`
// des dynamischen Shared Memory; belegt ghost_shared_bytes. Der Aufrufer synchronisiert danach.
template <class R>
__device__ inline BlockGhosts<R> load_block_ghosts(unsigned char* dynamic,
                                                   const SystemBatchView<R>& b,
                                                   const BlockTuple& tu, int tid) {
    BlockGhosts<R> g{b.shape.ghost_plates, b.shape.ghost_terms, nullptr, nullptr, nullptr};
    const int G = g.plates;
    const int K = g.terms;
    if (G > 0) {
        g.coeff = reinterpret_cast<R*>(dynamic);
        g.index = g.coeff + G * K;
        g.radius = g.index + G;
        const R* const coeff = b.ghost_coeff + b.shape.ghost_coeff_at(tu.v, 0, 0);
        for (int i = tid; i < G * K; i += kBlockThreads) {
            g.coeff[i] = coeff[i];
        }
        for (int p = tid; p < G; p += kBlockThreads) {
            g.index[p] = b.ghost_index[b.shape.ghost_index_at(tu.v, p, tu.l)];
            g.radius[p] = b.ghost_norm_radius[p];
        }
    }
    return g;
}

// Sicht mit nur einer Wellenlänge (table_l = 0); dieselben Zahlen wie surface_table(v),
// einschließlich der Geisterplatten.
template <class R>
__device__ inline SurfaceTableView<R> block_table_view(const BlockTable<R>& t, int surfaces,
                                                       const BlockGhosts<R>& g) {
    return SurfaceTableView<R>{surfaces, 1,       t.type,  t.curvature, t.thickness, t.index,
                               g.plates, g.terms, g.coeff, g.index,     g.radius};
}

// Shared Memory des Kolbens.
template <class R>
struct PistonShared {
    R weight[kBlockWarps];
    R weighted_opl[kBlockWarps];
    R piston;
    int valid;
};

// Kolben OPL̄ = Σ W·OPL / W_ok aus den Teilsummen (W_ok, W·OPL) der Threads: Baumreduktion im
// Warp, Thread 0 summiert die Warps in fester Reihenfolge. Ohne gültigen Strahl (W_ok = 0) ist
// valid = false und der Kolben 0. Alle Threads müssen eintreten (zwei __syncthreads) und
// erhalten das Ergebnis.
template <class R>
__device__ inline void block_piston(PistonShared<R>& sh, R weight_ok, R weighted_opl, int tid,
                                    R& piston, bool& valid) {
    weight_ok = warp_sum(weight_ok);
    weighted_opl = warp_sum(weighted_opl);
    if (tid % kWarpSize == 0) {
        sh.weight[tid / kWarpSize] = weight_ok;
        sh.weighted_opl[tid / kWarpSize] = weighted_opl;
    }
    __syncthreads();
    if (tid == 0) {
        R w = R(0);
        R wo = R(0);
        for (int i = 0; i < kBlockWarps; ++i) {
            w += sh.weight[i];
            wo += sh.weighted_opl[i];
        }
        const bool ok = w > R(0);
        sh.valid = ok ? 1 : 0;
        sh.piston = ok ? wo / w : R(0);
    }
    __syncthreads();
    valid = sh.valid != 0;
    piston = sh.piston;
}

// Summe über den Block: Baumreduktion im Warp, dann die Warps in fester Reihenfolge. Das
// Ergebnis ist nur in Thread 0 gültig. `sh` hat kBlockWarps Einträge und wird nur von diesem
// Aufruf benutzt; alle Threads müssen eintreten (ein __syncthreads).
template <class T>
__device__ inline T block_sum(T* sh, T x, int tid) {
    x = warp_sum(x);
    if (tid % kWarpSize == 0) {
        sh[tid / kWarpSize] = x;
    }
    __syncthreads();
    T total = T(0);
    if (tid == 0) {
        for (int i = 0; i < kBlockWarps; ++i) {
            total += sh[i];
        }
    }
    return total;
}

// Shared Memory der Max-Reduktionen: [Warp][Fläche][2] (für R = double 4 KB).
template <class R>
struct ExtremaShared {
    R value[kBlockWarps * kMaxSurfaces * 2];
};

// Platzhalter für abgeschaltete Max-Reduktionen (Kernel-Variante ohne Extrema: kein Shared und
// kein Local Memory dafür).
struct NoExtrema {};

template <class R, bool kEnabled>
using ExtremaSharedIf = std::conditional_t<kEnabled, ExtremaShared<R>, NoExtrema>;
template <class R, bool kEnabled>
using ExtremaIf = std::conditional_t<kEnabled, SurfaceExtrema<R>, NoExtrema>;
template <class R, bool kEnabled>
using RayRecordIf = std::conditional_t<kEnabled, RaySurfaceRecord<R>, NoExtrema>;

// Max-Reduktionen des Blocks aus den Akkumulatoren der Threads: pro Fläche Maximum der
// Höhenquadrate und Minimum von cos I im Warp, dann über die Warps in fester Reihenfolge.
// Schreibt out[2 s] = max x² + y², out[2 s + 1] = min cos I für s < S. Alle Threads müssen
// eintreten (ein __syncthreads).
template <class R>
__device__ inline void block_extrema(ExtremaShared<R>& sh, const SurfaceExtrema<R>& acc,
                                     int surfaces, int tid, R* out) {
    const int lane = tid % kWarpSize;
    const int warp = tid / kWarpSize;
    for (int s = 0; s < surfaces; ++s) {
        const R h = warp_max(acc.max_height_sq[s]);
        const R c = warp_min(acc.min_cos_incidence[s]);
        if (lane == 0) {
            sh.value[(warp * kMaxSurfaces + s) * 2] = h;
            sh.value[(warp * kMaxSurfaces + s) * 2 + 1] = c;
        }
    }
    __syncthreads();
    for (int s = tid; s < surfaces; s += kBlockThreads) {
        R h = neutral_height_sq<R>();
        R c = neutral_cos_incidence<R>();
        for (int w = 0; w < kBlockWarps; ++w) {
            h = extremum_max(h, sh.value[(w * kMaxSurfaces + s) * 2]);
            c = extremum_min(c, sh.value[(w * kMaxSurfaces + s) * 2 + 1]);
        }
        out[2 * s] = h;
        out[2 * s + 1] = c;
    }
}

// Zweite Stufe der Max-Reduktionen: ein Thread pro (v, s) über die B = F · L Blöcke der
// Variante in fester Reihenfolge b = f · L + l, danach √ der Höhe.
// partial[(v · B + b) · S + s][2], out[(v · S + s)][2].
template <class R>
struct ExtremaFinalizeArgs {
    std::int64_t variants;
    int surfaces;
    int blocks_per_variant;
    const R* partial;
    R* out;
};

template <class R>
__global__ void extrema_finalize_kernel(const ExtremaFinalizeArgs<R> a) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
    const std::int64_t S = a.surfaces;
    if (i >= a.variants * S) {
        return;
    }
    const std::int64_t v = i / S;
    const std::int64_t s = i % S;
    R h = neutral_height_sq<R>();
    R c = neutral_cos_incidence<R>();
    for (int b = 0; b < a.blocks_per_variant; ++b) {
        const R* p = a.partial + ((v * a.blocks_per_variant + b) * S + s) * 2;
        h = extremum_max(h, p[0]);
        c = extremum_min(c, p[1]);
    }
    a.out[i * 2] = msqrt(h);
    a.out[i * 2 + 1] = c;
}

// Startet extrema_finalize_kernel (Gerät ist gesetzt) und prüft den Start.
template <class R>
inline void launch_extrema_finalize(const ExtremaFinalizeArgs<R>& a) {
    const std::int64_t n = a.variants * a.surfaces;
    if (n == 0) {
        return;
    }
    const auto grid = static_cast<unsigned>((n + kFinalizeThreads - 1) / kFinalizeThreads);
    extrema_finalize_kernel<R><<<grid, kFinalizeThreads>>>(a);
    MKO_CUDA_CHECK(cudaGetLastError());
}

// Zeiger auf einen Bereich des Ausgabepuffers (nullptr für leere Bereiche).
template <class T>
T* region_ptr(void* buffer, const BufferRegion& region) {
    return region.bytes == 0 ? nullptr
                             : reinterpret_cast<T*>(static_cast<char*>(buffer) + region.offset);
}

template <class T>
const T* region_ptr(const void* buffer, const BufferRegion& region) {
    return region.bytes == 0
               ? nullptr
               : reinterpret_cast<const T*>(static_cast<const char*>(buffer) + region.offset);
}

} // namespace mko::cuda::detail
