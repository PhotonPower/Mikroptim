// Kernel MERIT (M2-T3): Merit F_v = Σ r_k² pro Variante mit deterministischer zweistufiger
// Reduktion (ADR-0007): Teilsumme pro Block (v, f, l) als Baumreduktion, danach Summe in
// fester Reihenfolge (f, l) pro Variante. Die Residuen entstehen im Block wie im Kernel
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
    int max_samples;               // Plätze für OPL und Status im dynamischen Shared Memory
    unsigned ghost_bytes;          // Geisterdaten am Anfang des dynamischen Shared Memory
    int soft;                      // 1: weiches Versagen mit Strafresiduen (M4-T2)
    double fail_weight;            // w_fail [Wellen²]
    double* partial;               // [V · F · L] Σ r_k² (+ Σ p_k²) des Blocks
    std::int32_t* failed_partial;  // [V · F · L] versagte Strahlen des Blocks
    std::int32_t* hard_partial;    // [V · F · L] hart versagte Strahlen des Blocks
    double* extrema_partial;       // [V · F · L][S][2] (nur mit kExtrema)
};

// Block b = (v · F + f) · L + l. Ablauf wie residuals_kernel (Laden, Durchlauf 1, Kolben,
// Durchlauf 2), aber OPL und Status im Shared Memory und statt der Ablage der Residuen die
// Quadratsumme als Baumreduktion (im Rechentyp R, Ablage als double). Alle Threads
// durchlaufen jede Schleife und jedes __syncthreads.
template <class R, bool kExtrema>
__global__ void __launch_bounds__(kBlockThreads) merit_kernel(const MeritKernelArgs a) {
    __shared__ detail::BlockTable<R> sh_table;
    __shared__ detail::PistonShared<R> sh_piston;
    __shared__ R sh_sum[kBlockWarps];
    __shared__ std::int32_t sh_failed[kBlockWarps];
    __shared__ std::int32_t sh_hard[kBlockWarps];
    __shared__ detail::ExtremaSharedIf<R, kExtrema> sh_extrema;
    // Dynamisch: Geisterplatten (ghost_bytes, Vielfaches von 8), OPL [max_samples] (R), danach
    // Status [max_samples] (1 Byte)
    extern __shared__ __align__(16) unsigned char sh_dynamic[];
    R* const sh_opl = reinterpret_cast<R*>(sh_dynamic + a.ghost_bytes);
    std::uint8_t* const sh_status = reinterpret_cast<std::uint8_t*>(sh_opl + a.max_samples);

    const int S = a.batch.shape.surfaces;
    const detail::BlockTuple tu =
        detail::block_tuple(a.batch.shape.fields, a.batch.shape.wavelengths);
    const int v = tu.v;
    const int f = tu.f;
    const int l = tu.l;
    const int tid = static_cast<int>(threadIdx.x);
    const ResidualFieldPlan plan = a.plan[f];

    detail::load_block_table(sh_table, a.batch, a.pupil, plan, tu, tid);
    const detail::BlockGhosts<R> ghosts =
        detail::load_block_ghosts<R>(sh_dynamic, a.batch, tu, tid);
    __syncthreads();

    const SurfaceTableView<R> table = detail::block_table_view(sh_table, S, ghosts);
    const Vec3<R> center = batch_image_point(a.batch, table, v, f);
    const int n = plan.samples;

    // Durchlauf 1 (wie RESIDUALS): Trace, OPL und Status in den Shared Memory, Teilsummen
    // (W_k, W_k · OPL_k) der gültigen Strahlen in aufsteigender Reihenfolge.
    [[maybe_unused]] detail::ExtremaIf<R, kExtrema> extrema;
    [[maybe_unused]] detail::RayRecordIf<R, kExtrema> record;
    if constexpr (kExtrema) {
        extrema.reset(S);
    }
    // Weiches Versagen (M4-T2): Σ p_k² der eigenen Stützstellen schon in Durchlauf 1 (die
    // Strafresiduen hängen nicht vom Kolben ab), in aufsteigender Reihenfolge.
    const R pscale = a.soft ? R(penalty_scale(a.batch, f, l, a.fail_weight)) : R(0);
    R penalty_sum = R(0);
    R weight_ok = R(0);
    R weighted_opl = R(0);
    for (int k = tid; k < n; k += kBlockThreads) {
        const PupilSample<R> s = sample(sh_table.pupil, k);
        R opl = R(0);
        SoftFailure<R> soft{R(0), TraceStatus::Ok};
        SoftFailure<R>* const soft_ptr = a.soft ? &soft : nullptr;
        TraceStatus st;
        if constexpr (kExtrema) {
            st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl, &record, soft_ptr);
        } else {
            NoSurfaceObserver* const none = nullptr;
            st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl, none, soft_ptr);
        }
        if (in_piston(st)) {
            weight_ok += s.weight;
            weighted_opl += s.weight * opl;
        }
        if constexpr (kExtrema) {
            if (st == TraceStatus::Ok) {
                extrema.add_ray(record, S);
            }
        }
        if (a.soft) {
            const R p = ray_penalty(pscale, s, st, soft.penalty);
            penalty_sum += p * p;
        }
        sh_opl[k] = opl;
        sh_status[k] = static_cast<std::uint8_t>(st);
    }

    R piston;
    bool valid;
    detail::block_piston(sh_piston, weight_ok, weighted_opl, tid, piston, valid);
    const R scale = R(residual_scale(a.batch, f, l));

    // Durchlauf 2: Residuen derselben Stützstellen wie in RESIDUALS, Quadratsumme in
    // aufsteigender Reihenfolge der eigenen Stützstellen, versagte Strahlen zählen.
    // Die Strafresiduen kommen nach der Quadratsumme der Wellenfrontresiduen hinzu.
    R sum = R(0);
    std::int32_t failed = 0;
    std::int32_t hard = 0;
    for (int k = tid; k < n; k += kBlockThreads) {
        const auto st = static_cast<TraceStatus>(sh_status[k]);
        R e = R(0);
        if (valid && in_piston(st)) {
            e = sh_opl[k] - piston;
        }
        const R r = ray_residual(scale, sample(sh_table.pupil, k), e);
        sum += r * r;
        failed += st == TraceStatus::Ok ? 0 : 1;
        hard += is_hard_failure(st) ? 1 : 0;
    }
    sum += penalty_sum;

    const R total = detail::block_sum(sh_sum, sum, tid);
    const std::int32_t total_failed = detail::block_sum(sh_failed, failed, tid);
    const std::int32_t total_hard = detail::block_sum(sh_hard, hard, tid);
    if (tid == 0) {
        a.partial[tu.block] = static_cast<double>(total);
        a.failed_partial[tu.block] = total_failed;
        a.hard_partial[tu.block] = total_hard;
    }
    if constexpr (kExtrema) {
        detail::block_extrema(sh_extrema, extrema, S, tid,
                              a.extrema_partial + tu.block * static_cast<std::int64_t>(S) * 2);
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
        MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, merit_kernel<R, true>));
    } else {
        MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, merit_kernel<R, false>));
    }
    return attr.sharedSizeBytes;
}

std::size_t static_shared_bytes(bool extrema, Precision precision) {
    return precision == Precision::Df64 ? static_shared_bytes_of<df64>(extrema)
                                        : static_shared_bytes_of<double>(extrema);
}

// OPL im Rechentyp (df64 und double: 8 Byte) und Status (1 Byte) pro Stützstelle.
constexpr std::size_t kBytesPerSample = sizeof(double) + sizeof(std::uint8_t);
static_assert(sizeof(df64) == sizeof(double));

// Startet den Block-Kernel mit Rechentyp R (Gerät ist gesetzt) und prüft den Start.
template <class R>
void launch_merit(const MeritKernelArgs& args, bool extrema, unsigned grid, std::size_t dynamic) {
    if (extrema) {
        merit_kernel<R, true><<<grid, kBlockThreads, dynamic>>>(args);
    } else {
        merit_kernel<R, false><<<grid, kBlockThreads, dynamic>>>(args);
    }
    MKO_CUDA_CHECK(cudaGetLastError());
}

std::size_t ghost_bytes_of(int ghost_plates, int ghost_terms) {
    BatchShape shape{};
    shape.ghost_plates = ghost_plates;
    shape.ghost_terms = ghost_terms;
    return detail::ghost_shared_bytes(shape);
}

} // namespace

int merit_max_samples(bool extrema, int ghost_plates, int ghost_terms, Precision precision) {
    if (ghost_plates < 0 || ghost_terms < 0) {
        throw std::invalid_argument("merit_max_samples: negative Zahl von Geisterplatten");
    }
    check_precision(precision);
    const std::size_t fixed =
        static_shared_bytes(extrema, precision) + ghost_bytes_of(ghost_plates, ghost_terms);
    if (fixed >= kMeritMaxSharedBytes) {
        return 0;
    }
    return static_cast<int>((kMeritMaxSharedBytes - fixed) / kBytesPerSample);
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
    int max_samples = 0;
    for (const ResidualFieldPlan& p : layout.plan()) {
        max_samples = std::max(max_samples, p.samples);
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
    const std::size_t total = extra_region.offset + align_bytes(extra_region.bytes);

    ran_ = false;
    if (buffer_ != nullptr && device_ != batch.device()) {
        release();
    }
    device_ = batch.device();
    const detail::DeviceGuard guard(device_);

    const std::size_t ghost_bytes = detail::ghost_shared_bytes(view.shape);
    const std::size_t dynamic =
        ghost_bytes + static_cast<std::size_t>(max_samples) * kBytesPerSample;
    if (static_shared_bytes(options.extrema, options.precision) + dynamic > kMeritMaxSharedBytes) {
        throw std::invalid_argument(
            "DeviceMerit::run: " + std::to_string(max_samples) +
            " Stützstellen pro Feld passen nicht in das Shared Memory (höchstens " +
            std::to_string(merit_max_samples(options.extrema, view.shape.ghost_plates,
                                             view.shape.ghost_terms, options.precision)) +
            ")");
    }

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

    if (blocks > 0) {
        MeritKernelArgs args{};
        args.batch = view;
        args.pupil = make_pupil_quadrature<double>(layout.settings().rings,
                                                   layout.settings().spokes, PupilMode::Half);
        args.plan = region_ptr<ResidualFieldPlan>(buffer_, plan_region);
        args.max_samples = max_samples;
        args.ghost_bytes = static_cast<unsigned>(ghost_bytes);
        args.soft = layout.soft_failure() ? 1 : 0;
        args.fail_weight = layout.settings().fail_weight;
        args.partial = region_ptr<double>(buffer_, out.merit_partial);
        args.failed_partial = region_ptr<std::int32_t>(buffer_, out.failed_rays_partial);
        args.hard_partial = region_ptr<std::int32_t>(buffer_, out.hard_failed_rays_partial);
        args.extrema_partial = region_ptr<double>(buffer_, out.surface_max_partial);
        if (options.precision == Precision::Df64) {
            launch_merit<df64>(args, options.extrema, static_cast<unsigned>(blocks), dynamic);
        } else {
            launch_merit<double>(args, options.extrema, static_cast<unsigned>(blocks), dynamic);
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
        const auto grid =
            static_cast<unsigned>((static_cast<std::int64_t>(V) + detail::kFinalizeThreads - 1) /
                                  detail::kFinalizeThreads);
        merit_finalize_kernel<double><<<grid, detail::kFinalizeThreads>>>(fin);
        MKO_CUDA_CHECK(cudaGetLastError());

        if (options.extrema) {
            detail::ExtremaFinalizeArgs<double> ext{};
            ext.variants = V;
            ext.surfaces = view.shape.surfaces;
            ext.blocks_per_variant = B;
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
