#pragma once

// Gemeinsame Bausteine der Kernel RESIDUALS (M2-T2) und MERIT (M2-T3): Block-Abbildung,
// Flächentabelle im Shared Memory, deterministische Reduktionen (ADR-0007) und die zweite
// Stufe der Max-Reduktionen. Nur aus .cu-Dateien einbinden. Beide Kernel rufen dieselben
// Funktionen in derselben Reihenfolge auf; deshalb sind Kolben, OPD und Residuen in beiden
// Modi bitgleich (Test CudaMerit.MatchesSumOfResiduals). docs/architektur.md, Abschnitte
// "Kernel RESIDUALS" und "Kernel MERIT".
//
// Zahlentypen (M7-T1): Der Batch im Gerätespeicher und alle Ausgaben sind FP64. Die Kernel sind
// über den Rechentyp R templatisiert (double oder df64); load_block_table und
// load_block_ghosts wandeln die Daten des Blocks beim Laden in R um, die Ausgaben werden beim
// Schreiben nach double gewandelt. Für R = double sind beide Umwandlungen ohne Wirkung.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/cuda/tuple_packs.hpp"
#include "mikroptim/math/df64.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/merit/ray_residual.hpp"
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

// Rechen- und Speichertyp teilen sich die Puffer im Shared und globalen Speicher (Geisterdaten,
// OPL im Kernel MERIT, vorläufige OPL auf dem Residuenplatz im Kernel RESIDUALS).
static_assert(sizeof(df64) == sizeof(double), "df64 und double müssen gleich groß sein");

// Warp-Shuffle für eingebaute Zahlentypen; df64 hat eine eigene Überladung in df64.hpp (zwei
// float-Shuffles), die über ADL gefunden wird.
template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
__device__ inline T shfl_down_sync(unsigned mask, T x, unsigned delta) {
    return __shfl_down_sync(mask, x, delta);
}

// Summe über die 32 Lanes eines Warps als feste Baumreduktion (Abstände 16, 8, 4, 2, 1);
// das Ergebnis steht in Lane 0. Alle Lanes müssen teilnehmen.
template <class T>
__device__ inline T warp_sum(T x) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset /= 2) {
        x += shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

// Maximum bzw. Minimum über die Lanes eines Warps (Ergebnis in Lane 0), gleiche Struktur.
template <class R>
__device__ inline R warp_max(R x) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset /= 2) {
        x = extremum_max(x, shfl_down_sync(0xffffffffu, x, offset));
    }
    return x;
}
template <class R>
__device__ inline R warp_min(R x) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset /= 2) {
        x = extremum_min(x, shfl_down_sync(0xffffffffu, x, offset));
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
// make_pupil_quadrature(N, M, plan.mode)) und wandelt sie in den Rechentyp R um. Der Aufrufer
// synchronisiert danach.
template <class R>
__device__ inline void load_block_table(BlockTable<R>& t, const SystemBatchView<double>& b,
                                        const PupilQuadrature<double>& pupil,
                                        const ResidualFieldPlan& plan, const BlockTuple& tu,
                                        int tid) {
    const int S = b.shape.surfaces;
    for (int s = tid; s < S; s += kBlockThreads) {
        t.type[s] = b.type[s];
        t.curvature[s] = R(b.curvature[b.shape.surface_at(tu.v, s)]);
        t.thickness[s] = R(b.thickness[b.shape.surface_at(tu.v, s)]);
        t.index[s] = R(b.index[b.shape.index_at(tu.v, s, tu.l)]);
    }
    if (tid < pupil.rings) {
        t.pupil.u[tid] = R(pupil.u[tid]);
        t.pupil.w[tid] = R(pupil.w[tid]);
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

// Bytes der Geisterdaten im dynamischen Shared Memory (R = double oder df64, beide 8 Byte;
// Vielfaches von 8).
__host__ __device__ inline std::size_t ghost_shared_bytes(const BatchShape& shape) {
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

// Lädt die Geisterdaten der Variante (Brechzahlen der Wellenlänge l) im Rechentyp R an den
// Anfang `dynamic` des dynamischen Shared Memory; belegt ghost_shared_bytes. Der Aufrufer
// synchronisiert danach.
template <class R>
__device__ inline BlockGhosts<R> load_block_ghosts(unsigned char* dynamic,
                                                   const SystemBatchView<double>& b,
                                                   const BlockTuple& tu, int tid) {
    BlockGhosts<R> g{b.shape.ghost_plates, b.shape.ghost_terms, nullptr, nullptr, nullptr};
    const int G = g.plates;
    const int K = g.terms;
    if (G > 0) {
        g.coeff = reinterpret_cast<R*>(dynamic);
        g.index = g.coeff + G * K;
        g.radius = g.index + G;
        const double* const coeff = b.ghost_coeff + b.shape.ghost_coeff_at(tu.v, 0, 0);
        for (int i = tid; i < G * K; i += kBlockThreads) {
            g.coeff[i] = R(coeff[i]);
        }
        for (int p = tid; p < G; p += kBlockThreads) {
            g.index[p] = R(b.ghost_index[b.shape.ghost_index_at(tu.v, p, tu.l)]);
            g.radius[p] = R(b.ghost_norm_radius[p]);
        }
    }
    return g;
}

// Dezentrierung und Verkippung der Variante (M8-T3, ADR-0011) im dynamischen Shared Memory,
// nur bei nicht koaxialem Batch (b.decenter_tilt != nullptr; sonst 0 Byte und keine
// Ladebefehle): decenter_tilt [S · 4] und tilt_trig [S · 4] (cos α, sin α, cos β, sin β, in FP64
// berechnet und in den Rechentyp R gewandelt), höchstens 8 Byte pro Wert.
__host__ __device__ inline std::size_t transform_shared_bytes(const SystemBatchView<double>& b) {
    return b.decenter_tilt == nullptr
               ? 0
               : 2 * static_cast<std::size_t>(b.shape.surfaces) *
                     static_cast<std::size_t>(kSurfaceTransformValues) * sizeof(double);
}

// Lädt die Transformationsdaten der Variante v nach `dst` (2 · S · 4 Werte vom Typ R) und
// trägt sie in die Sicht `t` ein; ohne Transformationsdaten im Batch ohne Wirkung. Der
// Aufrufer synchronisiert danach.
template <class R>
__device__ inline void load_block_transforms(R* dst, const SystemBatchView<double>& b, int v,
                                             int tid, SurfaceTableView<R>& t) {
    if (b.decenter_tilt == nullptr) {
        return;
    }
    const int S = b.shape.surfaces;
    const double* const src = b.decenter_tilt + b.shape.transform_at(v, 0, 0);
    R* const trig = dst + S * kSurfaceTransformValues;
    for (int s = tid; s < S; s += kBlockThreads) {
        const double* const x = src + s * kSurfaceTransformValues;
        R* const o = dst + s * kSurfaceTransformValues;
        R* const c = trig + s * kSurfaceTransformValues;
        o[0] = R(x[0]);
        o[1] = R(x[1]);
        o[2] = R(x[2]);
        o[3] = R(x[3]);
        c[0] = R(cos(x[2]));
        c[1] = R(sin(x[2]));
        c[2] = R(cos(x[3]));
        c[3] = R(sin(x[3]));
    }
    t.decenter_tilt = dst;
    t.tilt_trig = trig;
}

// Tabellentyp der Kernel (M8-T3): Jeder Trace-Kernel ist für koaxiale Batches
// (kTransforms = false, CoaxialTableView: der Tracer übersetzt ohne Transformationen, genau der
// Code von vor M8-T3) und für nicht koaxiale (SurfaceTableView mit Transformationsdaten im
// Shared Memory) instanziiert; der Host wählt nach batch.decenter_tilt.
template <class R, bool kTransforms>
using KernelTable = std::conditional_t<kTransforms, SurfaceTableView<R>, CoaxialTableView<R>>;

// Sicht mit nur einer Wellenlänge (table_l = 0); dieselben Zahlen wie surface_table(v),
// einschließlich der Geisterplatten (Transformationsdaten: load_block_transforms).
template <class R>
__device__ inline SurfaceTableView<R> block_table_view(const BlockTable<R>& t, int surfaces,
                                                       const BlockGhosts<R>& g) {
    return SurfaceTableView<R>{surfaces, 1,       t.type,  t.curvature, t.thickness, t.index,
                               g.plates, g.terms, g.coeff, g.index,     g.radius};
}

// block_table_view im Tabellentyp der Kernel-Instanz (KernelTable); mit kTransforms lädt sie
// zusätzlich die Transformationsdaten der Variante v nach `transform_dst` (M8-T3). Der Aufrufer
// synchronisiert danach.
template <class R, bool kTransforms>
__device__ inline KernelTable<R, kTransforms>
block_kernel_table(const BlockTable<R>& t, int surfaces, const BlockGhosts<R>& g, R* transform_dst,
                   const SystemBatchView<double>& b, int v, int tid) {
    SurfaceTableView<R> view = block_table_view(t, surfaces, g);
    if constexpr (kTransforms) {
        load_block_transforms(transform_dst, b, v, tid, view);
    }
    return KernelTable<R, kTransforms>{view};
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

// Shared Memory der Max-Reduktionen: [Warp][Fläche][2] (für R = double und df64 4 KB).
template <class R>
struct ExtremaShared {
    R value[kBlockWarps * kMaxSurfaces * 2];
};

// Platzhalter für abgeschaltete Max-Reduktionen (Kernel-Variante ohne Extrema: kein Shared und
// kein Local Memory dafür).
struct NoExtrema {};

template <class R, bool kEnabled>
using ExtremaSharedIf = std::conditional_t<kEnabled, ExtremaShared<R>, NoExtrema>;

// Max-Reduktionen des Blocks aus den Akkumulatoren der Threads: pro Fläche Maximum der
// Höhenquadrate und Minimum von cos I im Warp, dann über die Warps in fester Reihenfolge.
// Schreibt out[2 s] = max x² + y², out[2 s + 1] = min cos I für s < S (nach double gewandelt).
// height(s) und cos_incidence(s) liefern den Akkumulator des Threads. Alle Threads müssen
// eintreten (ein __syncthreads).
template <class R, class Height, class CosIncidence>
__device__ inline void block_extrema_of(ExtremaShared<R>& sh, Height height,
                                        CosIncidence cos_incidence, int surfaces, int tid,
                                        double* out) {
    const int lane = tid % kWarpSize;
    const int warp = tid / kWarpSize;
    for (int s = 0; s < surfaces; ++s) {
        const R h = warp_max(height(s));
        const R c = warp_min(cos_incidence(s));
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
        out[2 * s] = static_cast<double>(h);
        out[2 * s + 1] = static_cast<double>(c);
    }
}

// Akkumulator der Max-Reduktionen mit Doppelpuffer (M7-T4), zugleich Beobachter des Traces:
// surface() schreibt max(Akkumulator, Höhenquadrat) bzw. min(Akkumulator, cos I) des Strahls
// sofort in den zweiten Puffer, commit() übernimmt ihn nach einem gültigen Strahl (Status Ok)
// durch Umschalten. Ein versagter Strahl hinterlässt nur den verworfenen Puffer. Gleiche Werte
// wie RaySurfaceRecord plus SurfaceExtrema::add_ray (dieselben Formeln und Vergleiche), aber
// halb so viel Local-Memory-Verkehr pro Strahl und Fläche (2 Lade- und 2 Schreibzugriffe statt
// 4 und 4). 4 · kMaxSurfaces · sizeof(R) Byte (wie Aufzeichnung und Akkumulator zusammen).
template <class R>
struct PendingExtrema {
    R height_sq[2][kMaxSurfaces];
    R cos_incidence[2][kMaxSurfaces];
    int committed;

    __device__ void reset(int surfaces) {
        committed = 0;
        for (int s = 0; s < surfaces; ++s) {
            height_sq[0][s] = neutral_height_sq<R>();
            cos_incidence[0][s] = neutral_cos_incidence<R>();
        }
    }

    // Beobachter (Formeln wie RaySurfaceRecord::surface)
    __device__ void surface(int s, R c, const Vec3<R>& d_before, const Ray<R>& ray) {
        const Vec3<R>& p = ray.p;
        const R h = p.x * p.x + p.y * p.y;
        const Vec3<R> normal{-c * p.x, -c * p.y, R(1) - c * p.z};
        const R ci = dot(d_before, normal);
        const int a = committed;
        height_sq[1 - a][s] = extremum_max(height_sq[a][s], h);
        cos_incidence[1 - a][s] = extremum_min(cos_incidence[a][s], ci);
    }

    // Übernimmt den Strahl (nur bei Status Ok: dann hat er alle Flächen 0 … S − 1 passiert).
    __device__ void commit() { committed = 1 - committed; }
};

template <class R>
__device__ inline void block_extrema(ExtremaShared<R>& sh, const PendingExtrema<R>& acc,
                                     int surfaces, int tid, double* out) {
    const int a = acc.committed;
    block_extrema_of<R>(
        sh, [&acc, a](int s) { return acc.height_sq[a][s]; },
        [&acc, a](int s) { return acc.cos_incidence[a][s]; }, surfaces, tid, out);
}

template <class R, bool kEnabled>
using PendingExtremaIf = std::conditional_t<kEnabled, PendingExtrema<R>, NoExtrema>;

// Zweite Stufe der Max-Reduktionen: ein Thread pro (v, s) über die B Blöcke der Variante
// (seit M7-T4 Tupelpakete, b = v · B + p) in fester Reihenfolge, danach √ der Höhe.
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

// ---- Tupelpakete (M7-T4) ------------------------------------------------------------------
//
// Die Kernel RESIDUALS und MERIT rechnen seit M7-T4 nicht mehr einen Block pro Tupel (v, f, l),
// sondern einen Block pro Paket aufeinanderfolgender Tupel t = f · L + l derselben Variante
// (docs/architektur.md, "Kernel-Abbildung"). Die Strahlen eines Pakets liegen in der
// Residuenablage hintereinander (offset(t + 1) = offset(t) + n_f); Durchlauf 1 (Trace) verteilt
// sie dicht auf die Threads (Strahl r = t, t + 128, …), sodass nur der letzte Warp eines Pakets
// leere Lanes hat. Kolben und Quadratsumme bildet jedes Tupel dagegen unabhängig von Paket und
// Blockgröße in der Reihenfolge von M2: 128 virtuelle Lanes c summieren die Stützstellen
// k = c, c + 128, … aufsteigend, Baum über je 32 Lanes (Gruppe g = c / 32, ein Warp pro
// (Tupel, Gruppe)), dann die 4 Gruppen in fester Reihenfolge. Die Ergebnisse sind damit
// bitgleich zur Abbildung "ein Block à 128 Threads pro Tupel" (main vor M7-T4).

// Virtuelle Lanes der Reduktion eines Tupels (die Blockgröße von M2-T2 … M7-T3).
constexpr int kReductionLanes = 128;
constexpr int kReductionGroups = kReductionLanes / kWarpSize;
static_assert(kReductionLanes % kWarpSize == 0);

// Startgrenzen der Kernel RESIDUALS und MERIT. Standard: nur die Blockgröße (ptxas wählt die
// Register frei). Messschalter für die P100 (M7-T4, docs/hardware.md): -DMKO_KERNEL_MIN_BLOCKS=5
// in CMAKE_CUDA_FLAGS verlangt 5 residente Blöcke pro SM, also höchstens 96 Register; auf der
// RTX 3090 ohne messbaren Gewinn. (Ein ausdrückliches Minimum von 1 ist nicht dasselbe wie
// keines: ptxas teilt dann anders zu, z. B. 124 statt 96 Register für RESIDUALS auf sm_86.)
#ifdef MKO_KERNEL_MIN_BLOCKS
#define MKO_TRACE_LAUNCH_BOUNDS __launch_bounds__(kBlockThreads, MKO_KERNEL_MIN_BLOCKS)
#else
#define MKO_TRACE_LAUNCH_BOUNDS __launch_bounds__(kBlockThreads)
#endif

// Shared Memory pro Block, das die Kernel höchstens belegen (ohne Opt-in auf beiden Karten).
constexpr std::size_t kBlockMaxSharedBytes = 48 * 1024;

// Shared Memory eines Pakets (statisch): Flächentabelle ohne Brechzahlen, Quadratur (Knoten,
// Gewichte, Ringzahl; Speichen und Pupillenart pro Tupel), Angaben pro Tupel, Kolben und
// Teilsummen der Gruppen.
template <class R>
struct PackShared {
    SurfaceKind type[kMaxSurfaces];
    R curvature[kMaxSurfaces];
    R thickness[kMaxSurfaces];
    PupilQuadrature<R> pupil;
    int field[kMaxPackTuples];
    int wavelength[kMaxPackTuples];
    int spokes[kMaxPackTuples];
    PupilMode mode[kMaxPackTuples];
    int start[kMaxPackTuples + 1]; // erster Strahl des Tupels im Paket; start[tuples] = rays
    R scale[kMaxPackTuples];       // residual_scale(f, l)
    R pscale[kMaxPackTuples];      // penalty_scale(f, l, w_fail) im weichen Modus, sonst 0
    R piston[kMaxPackTuples];
    int valid[kMaxPackTuples];
    R sum_a[kMaxPackTuples * kReductionGroups]; // Teilsummen pro (Tupel, Gruppe)
    R sum_b[kMaxPackTuples * kReductionGroups];
    std::int32_t count_a[kMaxPackTuples * kReductionGroups];
    std::int32_t count_b[kMaxPackTuples * kReductionGroups];
};

// Lädt Flächentabelle (Brechzahlen des Wellenlängenfensters), Geisterdaten, Quadratur und die
// Angaben pro Tupel des Pakets in den Shared Memory (Tabellen an den Anfang von `dynamic`,
// pack_table_bytes) und liefert die Sicht darauf (wavelengths = Fensterbreite, Index
// l − first_wavelength). Dieselben Zahlen wie surface_table(v) auf der CPU. Mit kTransforms
// (Kernel-Instanz für nicht koaxiale Batches, M8-T3) zusätzlich die Transformationsdaten hinter
// den Brechzahlen, sonst die zur Übersetzungszeit koaxiale Sicht (KernelTable). Der Aufrufer
// synchronisiert danach.
template <class R, bool kTransforms>
__device__ inline KernelTable<R, kTransforms>
load_pack(PackShared<R>& sh, unsigned char* dynamic, const SystemBatchView<double>& b,
          const PupilQuadrature<double>& pupil, const ResidualFieldPlan* plan,
          const TuplePack& pack, int v, bool soft, double fail_weight, int tid) {
    const int S = b.shape.surfaces;
    const int L = b.shape.wavelengths;
    const int G = b.shape.ghost_plates;
    const int K = b.shape.ghost_terms;
    const int lo = pack.first_wavelength;
    const int W = pack.wavelengths;
    R* const coeff = reinterpret_cast<R*>(dynamic);
    R* const ghost_index = coeff + G * K;
    R* const radius = ghost_index + G * W;
    R* const index = radius + G;
    for (int s = tid; s < S; s += kBlockThreads) {
        sh.type[s] = b.type[s];
        sh.curvature[s] = R(b.curvature[b.shape.surface_at(v, s)]);
        sh.thickness[s] = R(b.thickness[b.shape.surface_at(v, s)]);
    }
    for (int i = tid; i < S * W; i += kBlockThreads) {
        const int s = i / W;
        index[i] = R(b.index[b.shape.index_at(v, s, lo + i - s * W)]);
    }
    if (G > 0) {
        const double* const c = b.ghost_coeff + b.shape.ghost_coeff_at(v, 0, 0);
        for (int i = tid; i < G * K; i += kBlockThreads) {
            coeff[i] = R(c[i]);
        }
        for (int i = tid; i < G * W; i += kBlockThreads) {
            const int p = i / W;
            ghost_index[i] = R(b.ghost_index[b.shape.ghost_index_at(v, p, lo + i - p * W)]);
        }
        for (int p = tid; p < G; p += kBlockThreads) {
            radius[p] = R(b.ghost_norm_radius[p]);
        }
    }
    if (tid < pupil.rings) {
        sh.pupil.u[tid] = R(pupil.u[tid]);
        sh.pupil.w[tid] = R(pupil.w[tid]);
    }
    if (tid < pack.tuples) {
        const int t = pack.first_tuple + tid;
        const int f = t / L;
        const int l = t % L;
        const ResidualFieldPlan p = plan[f];
        sh.field[tid] = f;
        sh.wavelength[tid] = l;
        sh.spokes[tid] = p.mode == PupilMode::Axis ? 1 : pupil.spokes;
        sh.mode[tid] = p.mode;
        sh.scale[tid] = R(residual_scale(b, f, l));
        sh.pscale[tid] = soft ? R(penalty_scale(b, f, l, fail_weight)) : R(0);
    }
    if (tid == 0) {
        sh.pupil.rings = pupil.rings;
        sh.pupil.spokes = pupil.spokes;
        sh.pupil.mode = PupilMode::Half; // pro Tupel in sh.mode
        int start = 0;
        for (int j = 0; j < pack.tuples; ++j) {
            sh.start[j] = start;
            start += plan[(pack.first_tuple + j) / L].samples;
        }
        sh.start[pack.tuples] = start;
    }
    SurfaceTableView<R> view{S,
                             W,
                             sh.type,
                             sh.curvature,
                             sh.thickness,
                             index,
                             G,
                             K,
                             G > 0 ? coeff : nullptr,
                             G > 0 ? ghost_index : nullptr,
                             G > 0 ? radius : nullptr};
    if constexpr (kTransforms) {
        // Transformationsdaten hinter den Brechzahlen (pack_table_bytes mit transforms)
        load_block_transforms(index + S * W, b, v, tid, view);
    }
    return KernelTable<R, kTransforms>{view};
}

// Tupel j des Strahls r im Paket (lineare Suche über start, höchstens kMaxPackTuples).
template <class R>
__device__ inline int pack_tuple_of(const PackShared<R>& sh, int tuples, int r) {
    int j = 0;
    while (j + 1 < tuples && r >= sh.start[j + 1]) {
        ++j;
    }
    return j;
}

// Stützstelle k des Tupels j.
template <class R>
__device__ inline PupilSample<R> pack_sample(const PackShared<R>& sh, int j, int k) {
    return sample(sh.pupil, k, sh.spokes[j], sh.mode[j]);
}

// Kolben aller Tupel des Pakets aus OPL und Status pro Strahl (opl[r], status[r], r = Strahl
// im Paket, vom Block vorher geschrieben und synchronisiert): ein Warp pro (Tupel j, Gruppe g),
// Lane c = 32 g + lane summiert (W_k, W_k · OPL_k) der Stützstellen k = c, c + 128, … im Kolben
// aufsteigend, Baumreduktion im Warp, danach summiert Thread j die 4 Gruppen in fester
// Reihenfolge: OPL̄_j = Σ W·OPL / W_ok in sh.piston[j], sh.valid[j] = (W_ok > 0). Dieselbe
// Reihenfolge wie block_piston mit 128 Threads pro Tupel. Alle Threads müssen eintreten (zwei
// __syncthreads); danach sind piston und valid für alle sichtbar.
template <class R>
__device__ inline void pack_piston(PackShared<R>& sh, int tuples, const R* opl,
                                   const std::uint8_t* status, int tid) {
    const int lane = tid % kWarpSize;
    for (int q = tid / kWarpSize; q < tuples * kReductionGroups; q += kBlockWarps) {
        const int j = q / kReductionGroups;
        const int first = sh.start[j];
        const int n = sh.start[j + 1] - first;
        R weight_ok = R(0);
        R weighted_opl = R(0);
        for (int k = (q % kReductionGroups) * kWarpSize + lane; k < n; k += kReductionLanes) {
            if (in_piston(static_cast<TraceStatus>(status[first + k]))) {
                const PupilSample<R> s = pack_sample(sh, j, k);
                weight_ok += s.weight;
                weighted_opl += s.weight * opl[first + k];
            }
        }
        weight_ok = warp_sum(weight_ok);
        weighted_opl = warp_sum(weighted_opl);
        if (lane == 0) {
            sh.sum_a[q] = weight_ok;
            sh.sum_b[q] = weighted_opl;
        }
    }
    __syncthreads();
    if (tid < tuples) {
        R w = R(0);
        R wo = R(0);
        for (int g = 0; g < kReductionGroups; ++g) {
            w += sh.sum_a[tid * kReductionGroups + g];
            wo += sh.sum_b[tid * kReductionGroups + g];
        }
        const bool ok = w > R(0);
        sh.valid[tid] = ok ? 1 : 0;
        sh.piston[tid] = ok ? wo / w : R(0);
    }
    __syncthreads();
}

} // namespace mko::cuda::detail
