// Kernel MERIT (M2-T3): Merit F_v = Σ r_k² pro Variante mit deterministischer zweistufiger
// Reduktion (ADR-0007): Teilsumme pro Tupel (v, f, l) als Baumreduktion (128 virtuelle Lanes,
// seit M7-T4 ein Block pro Tupelpaket), danach Summe in fester Reihenfolge (f, l) pro Variante.
// Die Residuen entstehen im Block wie im Kernel
// RESIDUALS (gemeinsame Bausteine in wavefront_block.hpp), die OPL liegen dazwischen im Shared
// Memory statt im globalen Speicher. Optional Max-Reduktionen pro Fläche.
// docs/architektur.md, Abschnitt "Kernel MERIT". Rechentyp double oder df64 (M7-T1,
// MeritRunOptions::precision); Batch, Teilsummen und Ergebnisse sind FP64.

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/math/df64.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"
#include "wavefront_block.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::cuda {

namespace {

using detail::kBlockThreads;
using detail::kBlockWarps;
using detail::region_ptr;

static_assert(kMeritThreads == kBlockThreads, "MERIT und RESIDUALS brauchen dieselbe Blockgröße");

// Parameter des Kernels (per Wert, < 1 KB Parameterraum). Batch und Ausgaben in FP64,
// unabhängig vom Rechentyp.
struct MeritKernelArgs {
    SystemBatchView<double> batch;
    PupilQuadrature<double> pupil; // Knoten und Gewichte (N, M); Pupillenart pro Feld aus plan
    const ResidualFieldPlan* plan; // [F]
    const TuplePack* packs;        // [P], gleich für alle Varianten (M7-T4)
    int packs_per_variant;         // P
    int soft;                      // 1: weiches Versagen mit Strafresiduen (M4-T2)
    double fail_weight;            // w_fail [Wellen²]
    double* partial;               // [V · F · L] Σ r_k² (+ Σ p_k²) pro Tupel
    std::int32_t* failed_partial;  // [V · F · L] versagte Strahlen pro Tupel
    std::int32_t* hard_partial;    // [V · F · L] hart versagte Strahlen pro Tupel
    double* extrema_partial;       // [V · P][S][2] (nur mit kExtrema)
};

// Block b = v · P + p: Paket p der Variante v (M7-T4). Ablauf wie residuals_kernel (Laden,
// Durchlauf 1 dicht über die Strahlen des Pakets, Kolben pro Tupel), aber OPL, Status und
// (weich) Strafresiduen im dynamischen Shared Memory und statt der Ablage der Residuen die
// Quadratsumme pro Tupel in der Reihenfolge von M2 (128 virtuelle Lanes, Baum pro Gruppe,
// Gruppen in fester Reihenfolge; im Rechentyp R, Ablage als double). Alle Threads durchlaufen
// jede Schleife und jedes __syncthreads.
template <class R, bool kExtrema, bool kTransforms>
__global__ void MKO_TRACE_LAUNCH_BOUNDS merit_kernel(const MeritKernelArgs a) {
    __shared__ detail::PackShared<R> sh;
    __shared__ detail::ExtremaSharedIf<R, kExtrema> sh_extrema;
    // Dynamisch: Tabellen des Pakets (pack_table_bytes), OPL [rays] (R), im weichen Modus
    // Strafresiduen [rays] (R), danach Status [rays] (1 Byte)
    extern __shared__ __align__(16) unsigned char sh_dynamic[];

    const int S = a.batch.shape.surfaces;
    const auto block = static_cast<std::int64_t>(blockIdx.x);
    const int v = static_cast<int>(block / a.packs_per_variant);
    const TuplePack pack = a.packs[block % a.packs_per_variant];
    const int tid = static_cast<int>(threadIdx.x);

    const detail::KernelTable<R, kTransforms> table = detail::load_pack<R, kTransforms>(
        sh, sh_dynamic, a.batch, a.pupil, a.plan, pack, v, a.soft != 0, a.fail_weight, tid);
    R* const sh_opl = reinterpret_cast<R*>(
        sh_dynamic + pack_table_bytes(S, a.batch.shape.ghost_plates, a.batch.shape.ghost_terms,
                                      pack.wavelengths, a.batch.decenter_tilt != nullptr));
    R* const sh_penalty = a.soft ? sh_opl + pack.rays : nullptr;
    std::uint8_t* const sh_status =
        reinterpret_cast<std::uint8_t*>(sh_opl + (a.soft ? 2 : 1) * pack.rays);
    __syncthreads();

    // Durchlauf 1 (wie RESIDUALS): Trace, OPL, Status und Strafresiduum in den Shared Memory.
    // Max-Reduktionen: Akkumulator mit Doppelpuffer, zugleich Beobachter (PendingExtrema)
    [[maybe_unused]] detail::PendingExtremaIf<R, kExtrema> extrema;
    if constexpr (kExtrema) {
        extrema.reset(S);
    }
    for (int r = tid; r < pack.rays; r += kBlockThreads) {
        const int j = detail::pack_tuple_of(sh, pack.tuples, r);
        const int k = r - sh.start[j];
        const int f = sh.field[j];
        const int l = sh.wavelength[j];
        const PupilSample<R> s = detail::pack_sample(sh, j, k);
        const Vec3<R> center = batch_image_point(a.batch, table, v, f);
        R opl = R(0);
        SoftFailure<R> soft{R(0), TraceStatus::Ok};
        SoftFailure<R>* const soft_ptr = a.soft ? &soft : nullptr;
        const int table_l = l - pack.first_wavelength;
        TraceStatus st;
        if constexpr (kExtrema) {
            st = batch_ray_opl(a.batch, table, table_l, v, f, l, s, center, &opl, &extrema,
                               soft_ptr);
        } else {
            NoSurfaceObserver* const none = nullptr;
            st = batch_ray_opl(a.batch, table, table_l, v, f, l, s, center, &opl, none, soft_ptr);
        }
        if constexpr (kExtrema) {
            if (st == TraceStatus::Ok) {
                extrema.commit();
            }
        }
        if (sh_penalty != nullptr) {
            sh_penalty[r] = ray_penalty(sh.pscale[j], s, st, soft.penalty);
        }
        sh_opl[r] = opl;
        sh_status[r] = static_cast<std::uint8_t>(st);
    }
    __syncthreads();

    detail::pack_piston(sh, pack.tuples, sh_opl, sh_status, tid);

    // Quadratsumme pro Tupel: ein Warp pro (Tupel j, Gruppe g), Lane c = 32 g + lane bildet wie
    // ein Thread in M2 Σ r_k² über k = c, c + 128, … aufsteigend, danach Σ p_k² derselben
    // Stützstellen dazu; Baum im Warp, Zähler versagter und hart versagter Strahlen.
    const int lane = tid % detail::kWarpSize;
    for (int q = tid / detail::kWarpSize; q < pack.tuples * detail::kReductionGroups;
         q += kBlockWarps) {
        const int j = q / detail::kReductionGroups;
        const int first = sh.start[j];
        const int n = sh.start[j + 1] - first;
        const bool valid = sh.valid[j] != 0;
        const R piston = sh.piston[j];
        const R scale = sh.scale[j];
        R sum = R(0);
        R penalty_sum = R(0);
        std::int32_t failed = 0;
        std::int32_t hard = 0;
        for (int k = (q % detail::kReductionGroups) * detail::kWarpSize + lane; k < n;
             k += detail::kReductionLanes) {
            const auto st = static_cast<TraceStatus>(sh_status[first + k]);
            R e = R(0);
            if (valid && in_piston(st)) {
                e = sh_opl[first + k] - piston;
            }
            const R r = ray_residual(scale, detail::pack_sample(sh, j, k), e);
            sum += r * r;
            failed += st == TraceStatus::Ok ? 0 : 1;
            hard += is_hard_failure(st) ? 1 : 0;
            if (sh_penalty != nullptr) {
                const R p = sh_penalty[first + k];
                penalty_sum += p * p;
            }
        }
        sum += penalty_sum;
        sum = detail::warp_sum(sum);
        failed = detail::warp_sum(failed);
        hard = detail::warp_sum(hard);
        if (lane == 0) {
            sh.sum_a[q] = sum;
            sh.count_a[q] = failed;
            sh.count_b[q] = hard;
        }
    }
    __syncthreads();
    if (tid < pack.tuples) {
        R total = R(0);
        std::int32_t total_failed = 0;
        std::int32_t total_hard = 0;
        for (int g = 0; g < detail::kReductionGroups; ++g) {
            total += sh.sum_a[tid * detail::kReductionGroups + g];
            total_failed += sh.count_a[tid * detail::kReductionGroups + g];
            total_hard += sh.count_b[tid * detail::kReductionGroups + g];
        }
        // Tupel t = f · L + l der Variante: Ablage wie bisher pro (v, f, l)
        const std::int64_t i =
            static_cast<std::int64_t>(v) * a.batch.shape.fields * a.batch.shape.wavelengths +
            pack.first_tuple + tid;
        a.partial[i] = static_cast<double>(total);
        a.failed_partial[i] = total_failed;
        a.hard_partial[i] = total_hard;
    }
    if constexpr (kExtrema) {
        detail::block_extrema(sh_extrema, extrema, S, tid,
                              a.extrema_partial + block * static_cast<std::int64_t>(S) * 2);
    }
}

// Zweite Stufe: ein Thread pro Variante summiert die B = F · L Teilsummen in fester
// Reihenfolge b = f · L + l und addiert zuletzt den Anteil weiterer Residuen.
template <class R>
struct MeritFinalizeArgs {
    std::int64_t variants;
    int blocks_per_variant;
    const R* partial;
    const std::int32_t* failed_partial;
    const std::int32_t* hard_partial;
    const R* extra; // [V] oder nullptr
    R* merit;
    std::int64_t* failed;
    std::int64_t* hard_failed;
};

template <class R>
__global__ void merit_finalize_kernel(const MeritFinalizeArgs<R> a) {
    const std::int64_t v =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
    if (v >= a.variants) {
        return;
    }
    const std::int64_t first = v * a.blocks_per_variant;
    R sum = R(0);
    std::int64_t failed = 0;
    std::int64_t hard = 0;
    for (int b = 0; b < a.blocks_per_variant; ++b) {
        sum += a.partial[first + b];
        failed += a.failed_partial[first + b];
        hard += a.hard_partial[first + b];
    }
    if (a.extra != nullptr) {
        sum += a.extra[v];
    }
    a.merit[v] = sum;
    a.failed[v] = failed;
    a.hard_failed[v] = hard;
}

// Statisches Shared Memory der Kernel-Variante (Kernelattribut).
template <class R>
std::size_t static_shared_bytes_of(bool extrema) {
    cudaFuncAttributes attr{};
    if (extrema) {
        MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, merit_kernel<R, true, false>));
    } else {
        MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, merit_kernel<R, false, false>));
    }
    return attr.sharedSizeBytes;
}

// Größeres statisches Shared Memory der beiden Rechentypen (PackShared<df64> ist wegen der
// Ausrichtung von float-Paaren einige Byte kleiner): merit_max_samples liefert damit für FP64
// und df64 dieselbe Grenze. `precision` ist geprüft.
std::size_t static_shared_bytes(bool extrema, Precision /*precision*/) {
    return std::max(static_shared_bytes_of<df64>(extrema), static_shared_bytes_of<double>(extrema));
}

// Pro Strahl im dynamischen Shared Memory: OPL im Rechentyp (df64 und double: 8 Byte), im
// weichen Modus das Strafresiduum (8 Byte), Status (1 Byte).
static_assert(sizeof(df64) == sizeof(double));
constexpr std::size_t bytes_per_ray(bool soft) {
    return (soft ? 2 : 1) * sizeof(double) + sizeof(std::uint8_t);
}

// Startet den Block-Kernel mit Rechentyp R (Gerät ist gesetzt) und prüft den Start.
template <class R>
void launch_merit(const MeritKernelArgs& args, bool extrema, unsigned grid, std::size_t dynamic) {
    if (extrema) {
        if (args.batch.decenter_tilt != nullptr) { // nicht koaxial (M8-T3)
            merit_kernel<R, true, true><<<grid, kBlockThreads, dynamic>>>(args);
        } else {
            merit_kernel<R, true, false><<<grid, kBlockThreads, dynamic>>>(args);
        }
    } else if (args.batch.decenter_tilt != nullptr) {
        merit_kernel<R, false, true><<<grid, kBlockThreads, dynamic>>>(args);
    } else {
        merit_kernel<R, false, false><<<grid, kBlockThreads, dynamic>>>(args);
    }
    MKO_CUDA_CHECK(cudaGetLastError());
}

} // namespace

int merit_max_samples(bool extrema, int ghost_plates, int ghost_terms, Precision precision,
                      FailureMode failure, bool non_coaxial) {
    if (ghost_plates < 0 || ghost_terms < 0) {
        throw std::invalid_argument("merit_max_samples: negative Zahl von Geisterplatten");
    }
    check_precision(precision);
    // Ein Paket aus einem Tupel mit den Brechzahlen einer Wellenlänge für S = kMaxSurfaces
    const std::size_t fixed =
        static_shared_bytes(extrema, precision) +
        pack_table_bytes(kMaxSurfaces, ghost_plates, ghost_terms, 1, non_coaxial);
    if (fixed >= kMeritMaxSharedBytes) {
        return 0;
    }
    return static_cast<int>((kMeritMaxSharedBytes - fixed) /
                            bytes_per_ray(failure == FailureMode::Soft));
}

DeviceMerit::~DeviceMerit() {
    release();
}

DeviceMerit::DeviceMerit(DeviceMerit&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_), ran_(other.ran_),
      extrema_(other.extrema_), variants_(other.variants_), surfaces_(other.surfaces_),
      rays_(other.rays_), layout_(other.layout_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.ran_ = false;
}

DeviceMerit& DeviceMerit::operator=(DeviceMerit&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        ran_ = other.ran_;
        extrema_ = other.extrema_;
        variants_ = other.variants_;
        surfaces_ = other.surfaces_;
        rays_ = other.rays_;
        layout_ = other.layout_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.ran_ = false;
    }
    return *this;
}

void DeviceMerit::release() noexcept {
    ran_ = false;
    if (buffer_ == nullptr) {
        return;
    }
    try {
        const detail::DeviceGuard guard(device_);
        cudaFree(buffer_); // Fehler beim Freigeben sind nicht behebbar
    } catch (...) {
        cudaFree(buffer_);
    }
    buffer_ = nullptr;
    capacity_ = 0;
}

void DeviceMerit::run(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                      const MeritRunOptions& options) {
    const SystemBatchView<double> view = batch.view(); // wirft bei leerem Batch
    check_residual_layout(batch.topology(), layout);
    check_precision(options.precision);
    if (options.pack_rays < 1) {
        throw std::invalid_argument("DeviceMerit::run: pack_rays muss >= 1 sein");
    }
    const int V = batch.variants();
    if (!options.extra_merit.empty() && options.extra_merit.size() != static_cast<std::size_t>(V)) {
        throw std::invalid_argument("DeviceMerit::run: extra_merit muss leer sein oder " +
                                    std::to_string(V) + " Werte haben");
    }
    const int B = view.shape.fields * view.shape.wavelengths;
    const std::int64_t blocks = static_cast<std::int64_t>(V) * B;
    if (blocks > kMaxGridBlocks) {
        throw std::invalid_argument("DeviceMerit::run: " + std::to_string(blocks) +
                                    " Blöcke überschreiten das größte Grid; Batch aufteilen");
    }
    OutputRequest request;
    request.rays_per_variant = layout.rays_per_variant();
    request.residuals = false;
    request.ray_status = false;
    request.opd = false;
    request.merit = true;
    request.surface_max = options.extrema;
    const OutputLayout out = make_output_layout(batch.topology(), V, request);
    const BufferRegion plan_region{out.total_bytes,
                                   layout.plan().size() * sizeof(ResidualFieldPlan)};
    const BufferRegion extra_region{plan_region.offset + align_bytes(plan_region.bytes),
                                    options.extra_merit.size_bytes()};

    ran_ = false;
    if (buffer_ != nullptr && device_ != batch.device()) {
        release();
    }
    device_ = batch.device();
    const detail::DeviceGuard guard(device_);

    // Tupelpakete (M7-T4): Tabellen, OPL, Strafresiduen und Status im dynamischen Shared Memory
    const bool soft = layout.soft_failure();
    const std::size_t static_bytes = static_shared_bytes(options.extrema, options.precision);
    // Grenze wie merit_max_samples (für kMaxSurfaces Flächen; mit der tatsächlichen
    // Flächenzahl passt das Paket dann sicher)
    int max_samples = 0;
    for (const ResidualFieldPlan& p : layout.plan()) {
        max_samples = std::max(max_samples, p.samples);
    }
    const int limit = merit_max_samples(options.extrema, view.shape.ghost_plates,
                                        view.shape.ghost_terms, options.precision,
                                        layout.settings().failure, view.decenter_tilt != nullptr);
    if (max_samples > limit) {
        throw std::invalid_argument("DeviceMerit::run: " + std::to_string(max_samples) +
                                    " Stützstellen pro Feld passen nicht in das Shared Memory "
                                    "(höchstens " +
                                    std::to_string(limit) + ")");
    }
    const bool transforms = view.decenter_tilt != nullptr; // nicht koaxial (M8-T3)
    const std::vector<TuplePack> packs = make_tuple_packs(
        layout, view.shape, options.pack_rays, bytes_per_ray(soft),
        kMeritMaxSharedBytes - std::min(static_bytes, kMeritMaxSharedBytes), transforms);
    const std::size_t dynamic =
        pack_dynamic_bytes(packs, view.shape, bytes_per_ray(soft), transforms);
    const auto P = static_cast<int>(packs.size());
    // Platz für F · L Pakete (höchstens so viele), damit der Puffer zur Abschätzung der
    // TraceEngine passt (fixed_run_bytes) und über Läufe wiederverwendet wird.
    const BufferRegion pack_region{extra_region.offset + align_bytes(extra_region.bytes),
                                   packs.size() * sizeof(TuplePack)};
    const std::size_t total =
        pack_region.offset + align_bytes(static_cast<std::size_t>(view.shape.fields) *
                                         view.shape.wavelengths * sizeof(TuplePack));

    if (total > capacity_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            capacity_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, total));
        capacity_ = total;
    }
    MKO_CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer_) + plan_region.offset,
                              layout.plan().data(), plan_region.bytes, cudaMemcpyHostToDevice));
    if (extra_region.bytes > 0) {
        MKO_CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer_) + extra_region.offset,
                                  options.extra_merit.data(), extra_region.bytes,
                                  cudaMemcpyHostToDevice));
    }
    MKO_CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer_) + pack_region.offset, packs.data(),
                              pack_region.bytes, cudaMemcpyHostToDevice));

    if (blocks > 0) {
        MeritKernelArgs args{};
        args.batch = view;
        args.pupil = make_pupil_quadrature<double>(layout.settings().rings,
                                                   layout.settings().spokes, PupilMode::Half);
        args.plan = region_ptr<ResidualFieldPlan>(buffer_, plan_region);
        args.packs = region_ptr<TuplePack>(buffer_, pack_region);
        args.packs_per_variant = P;
        args.soft = layout.soft_failure() ? 1 : 0;
        args.fail_weight = layout.settings().fail_weight;
        args.partial = region_ptr<double>(buffer_, out.merit_partial);
        args.failed_partial = region_ptr<std::int32_t>(buffer_, out.failed_rays_partial);
        args.hard_partial = region_ptr<std::int32_t>(buffer_, out.hard_failed_rays_partial);
        args.extrema_partial = region_ptr<double>(buffer_, out.surface_max_partial);
        const auto grid = static_cast<unsigned>(static_cast<std::int64_t>(V) * P);
        if (options.precision == Precision::Df64) {
            launch_merit<df64>(args, options.extrema, grid, dynamic);
        } else {
            launch_merit<double>(args, options.extrema, grid, dynamic);
        }

        MeritFinalizeArgs<double> fin{};
        fin.variants = V;
        fin.blocks_per_variant = B;
        fin.partial = args.partial;
        fin.failed_partial = args.failed_partial;
        fin.hard_partial = args.hard_partial;
        fin.extra = region_ptr<double>(buffer_, extra_region);
        fin.merit = region_ptr<double>(buffer_, out.merit);
        fin.failed = region_ptr<std::int64_t>(buffer_, out.failed_rays);
        fin.hard_failed = region_ptr<std::int64_t>(buffer_, out.hard_failed_rays);
        const auto fin_grid =
            static_cast<unsigned>((static_cast<std::int64_t>(V) + detail::kFinalizeThreads - 1) /
                                  detail::kFinalizeThreads);
        merit_finalize_kernel<double><<<fin_grid, detail::kFinalizeThreads>>>(fin);
        MKO_CUDA_CHECK(cudaGetLastError());

        if (options.extrema) {
            detail::ExtremaFinalizeArgs<double> ext{};
            ext.variants = V;
            ext.surfaces = view.shape.surfaces;
            ext.blocks_per_variant = P;
            ext.partial = args.extrema_partial;
            ext.out = region_ptr<double>(buffer_, out.surface_max);
            detail::launch_extrema_finalize(ext);
        }
        MKO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    layout_ = out;
    variants_ = V;
    surfaces_ = view.shape.surfaces;
    rays_ = layout.rays_per_variant();
    extrema_ = options.extrema;
    ran_ = true;
}

MeritResult DeviceMerit::download() const {
    MeritResult r;
    download(r);
    return r;
}

void DeviceMerit::download(MeritResult& r) const {
    if (!ran_) {
        throw std::logic_error("DeviceMerit::download: kein Lauf vorhanden");
    }
    r.variants = variants_;
    r.rays_per_variant = rays_;
    r.merit.resize(static_cast<std::size_t>(variants_));
    r.failed.resize(static_cast<std::size_t>(variants_));
    r.hard_failed.resize(static_cast<std::size_t>(variants_));
    r.extrema.variants = extrema_ ? variants_ : 0;
    r.extrema.surfaces = extrema_ ? surfaces_ : 0;
    r.extrema.values.resize(extrema_ ? layout_.surface_max.bytes / sizeof(double) : 0);
    const detail::DeviceGuard guard(device_);
    const char* base = static_cast<const char*>(buffer_);
    const auto copy = [base](void* dst, const BufferRegion& region) {
        if (region.bytes > 0) {
            MKO_CUDA_CHECK(
                cudaMemcpy(dst, base + region.offset, region.bytes, cudaMemcpyDeviceToHost));
        }
    };
    copy(r.merit.data(), layout_.merit);
    copy(r.failed.data(), layout_.failed_rays);
    copy(r.hard_failed.data(), layout_.hard_failed_rays);
    if (extrema_) {
        copy(r.extrema.values.data(), layout_.surface_max);
    }
}

const double* DeviceMerit::device_merit() const {
    return ran_ ? region_ptr<double>(static_cast<const void*>(buffer_), layout_.merit) : nullptr;
}

const std::int64_t* DeviceMerit::device_failed() const {
    return ran_ ? region_ptr<std::int64_t>(static_cast<const void*>(buffer_), layout_.failed_rays)
                : nullptr;
}

const std::int64_t* DeviceMerit::device_hard_failed() const {
    return ran_ ? region_ptr<std::int64_t>(static_cast<const void*>(buffer_),
                                           layout_.hard_failed_rays)
                : nullptr;
}

const double* DeviceMerit::device_extrema() const {
    return ran_ && extrema_
               ? region_ptr<double>(static_cast<const void*>(buffer_), layout_.surface_max)
               : nullptr;
}

MeritResult run_merit(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                      const MeritRunOptions& options) {
    DeviceMerit d;
    d.run(batch, layout, options);
    return d.download();
}

} // namespace mko::cuda
