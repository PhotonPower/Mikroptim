// FP32-Vorfilterung (M7-T2): Kernel in float mit Grenzabständen pro Strahl, deterministische
// zweistufige Reduktion (ADR-0007) und TraceEngine::prefilter_variants. Ablauf wie der Kernel
// MERIT (gemeinsame Bausteine in wavefront_block.hpp); docs/architektur.md, Abschnitt
// "FP32-Vorfilterung", Fehlerabschätzung: docs/numerik.md, Abschnitt 12.
//
// VORFILTERUNG: Einziger Kernel mit R = float (AGENTS.md §6.1, ADR-0003). Die Ergebnisse
// (PrefilterStats) dürfen nie in Merit, Residuen, Jacobi-Matrix oder Optimierung einfließen.

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/prefilter.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/merit/prefilter_ray.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"
#include "wavefront_block.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko::cuda {

namespace {

using detail::kBlockThreads;
using detail::kBlockWarps;
using detail::region_ptr;

static_assert(kPrefilterThreads == kBlockThreads, "Vorfilterung mit der Blockgröße von MERIT");

// Rechentyp der Vorfilterung. Nur hier und nur für die Vorfilterung (AGENTS.md §6.1).
using PrefilterReal = float;

// Klassen der Strahlen im Shared Memory.
enum RayClass : std::uint8_t {
    kCertainOk = 0,       // gültig, Grenzabstand > η
    kCertainFailed = 1,   // versagt, Betrag des Versagens > η
    kUncertainOk = 2,     // gültig, aber nahe der Grenze
    kUncertainFailed = 3, // versagt, aber nahe der Grenze
};

struct PrefilterKernelArgs {
    SystemBatchView<double> batch;
    PupilQuadrature<double> pupil;
    const ResidualFieldPlan* plan; // [F]
    int max_samples;               // Plätze für OPL und Klasse im dynamischen Shared Memory
    unsigned ghost_bytes;          // Geisterdaten (float) am Anfang des dynamischen Shared Memory
    PrefilterReal boundary;        // η
    double* merit_partial;         // [V · B]
    double* length_partial;        // [V · B]
    std::int32_t* counts_partial;  // [V · B][3]
};

// Maximum über den Block (Ergebnis nur in Thread 0), Struktur wie block_sum.
template <class R>
__device__ inline R block_max(R* sh, R x, int tid) {
    x = detail::warp_max(x);
    if (tid % detail::kWarpSize == 0) {
        sh[tid / detail::kWarpSize] = x;
    }
    __syncthreads();
    R m = R(0);
    if (tid == 0) {
        for (int i = 0; i < kBlockWarps; ++i) {
            m = extremum_max(m, sh[i]);
        }
    }
    return m;
}

// Block b = (v · F + f) · L + l. Ablauf wie merit_kernel: Laden, Durchlauf 1 (prefilter_ray,
// Klasse, Teilsummen des Kolbens über die sicher gültigen Strahlen), Kolben, Durchlauf 2
// (Quadratsumme der sicher gültigen Strahlen), Blockreduktionen. Alle Threads durchlaufen jede
// Schleife und jedes __syncthreads; versagte Strahlen laufen maskiert weiter.
__global__ void __launch_bounds__(kBlockThreads) prefilter_kernel(const PrefilterKernelArgs a) {
    using R = PrefilterReal;
    __shared__ detail::BlockTable<R> sh_table;
    __shared__ detail::PistonShared<R> sh_piston;
    __shared__ R sh_sum[kBlockWarps];
    __shared__ R sh_length[kBlockWarps];
    __shared__ std::int32_t sh_failed[kBlockWarps];
    __shared__ std::int32_t sh_uncertain_ok[kBlockWarps];
    __shared__ std::int32_t sh_uncertain_failed[kBlockWarps];
    // Dynamisch: Geisterplatten (ghost_bytes), OPL [max_samples] (float), Klasse [max_samples]
    extern __shared__ __align__(16) unsigned char sh_dynamic[];
    R* const sh_opl = reinterpret_cast<R*>(sh_dynamic + a.ghost_bytes);
    std::uint8_t* const sh_class = reinterpret_cast<std::uint8_t*>(sh_opl + a.max_samples);

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
    const R offset = prefilter_opl_offset(table, 0);
    const R abs_offset = offset < R(0) ? -offset : offset;
    const int n = plan.samples;

    R weight_ok = R(0);
    R weighted_opl = R(0);
    R length = R(0);
    std::int32_t failed = 0;
    std::int32_t uncertain_ok = 0;
    std::int32_t uncertain_failed = 0;
    for (int k = tid; k < n; k += kBlockThreads) {
        const PupilSample<R> s = sample(sh_table.pupil, k);
        const PrefilterRay<R> r = prefilter_ray(a.batch, table, 0, v, f, l, s, center, offset);
        const bool ok = r.status == TraceStatus::Ok;
        const bool certain = r.margin > a.boundary;
        const std::uint8_t cls = ok ? (certain ? kCertainOk : kUncertainOk)
                                    : (certain ? kCertainFailed : kUncertainFailed);
        if (cls == kCertainOk) {
            weight_ok += s.weight;
            weighted_opl += s.weight * r.opl;
            const R len = abs_offset + (r.opl < R(0) ? -r.opl : r.opl);
            length = extremum_max(length, len);
        }
        failed += cls == kCertainFailed ? 1 : 0;
        uncertain_ok += cls == kUncertainOk ? 1 : 0;
        uncertain_failed += cls == kUncertainFailed ? 1 : 0;
        sh_opl[k] = r.opl;
        sh_class[k] = cls;
    }

    R piston;
    bool valid;
    detail::block_piston(sh_piston, weight_ok, weighted_opl, tid, piston, valid);
    const R scale = R(residual_scale(a.batch, f, l));

    R sum = R(0);
    for (int k = tid; k < n; k += kBlockThreads) {
        R e = R(0);
        if (valid && sh_class[k] == kCertainOk) {
            e = sh_opl[k] - piston;
        }
        const R r = ray_residual(scale, sample(sh_table.pupil, k), e);
        sum += r * r;
    }

    const R total = detail::block_sum(sh_sum, sum, tid);
    const R max_length = block_max(sh_length, length, tid);
    const std::int32_t total_failed = detail::block_sum(sh_failed, failed, tid);
    const std::int32_t total_uncertain_ok = detail::block_sum(sh_uncertain_ok, uncertain_ok, tid);
    const std::int32_t total_uncertain_failed =
        detail::block_sum(sh_uncertain_failed, uncertain_failed, tid);
    if (tid == 0) {
        a.merit_partial[tu.block] = static_cast<double>(total);
        a.length_partial[tu.block] = static_cast<double>(max_length);
        a.counts_partial[tu.block * 3] = total_failed;
        a.counts_partial[tu.block * 3 + 1] = total_uncertain_ok;
        a.counts_partial[tu.block * 3 + 2] = total_uncertain_failed;
    }
}

// Zweite Stufe: ein Thread pro Variante, Blöcke in fester Reihenfolge b = f · L + l.
struct PrefilterFinalizeArgs {
    std::int64_t variants;
    int blocks_per_variant;
    const double* merit_partial;
    const double* length_partial;
    const std::int32_t* counts_partial;
    double* merit;
    double* length;
    std::int64_t* failed;
    std::int64_t* uncertain_ok;
    std::int64_t* uncertain_failed;
};

__global__ void prefilter_finalize_kernel(const PrefilterFinalizeArgs a) {
    const std::int64_t v =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
    if (v >= a.variants) {
        return;
    }
    const std::int64_t first = v * a.blocks_per_variant;
    double sum = 0.0;
    double length = 0.0;
    std::int64_t counts[3] = {0, 0, 0};
    for (int b = 0; b < a.blocks_per_variant; ++b) {
        sum += a.merit_partial[first + b];
        length = extremum_max(length, a.length_partial[first + b]);
        for (int c = 0; c < 3; ++c) {
            counts[c] += a.counts_partial[(first + b) * 3 + c];
        }
    }
    a.merit[v] = sum;
    a.length[v] = length;
    a.failed[v] = counts[0];
    a.uncertain_ok[v] = counts[1];
    a.uncertain_failed[v] = counts[2];
}

// Geisterdaten in float im dynamischen Shared Memory (auf 8 Byte aufgerundet).
std::size_t ghost_bytes_float(int ghost_plates, int ghost_terms) {
    const auto G = static_cast<std::size_t>(ghost_plates);
    const auto K = static_cast<std::size_t>(ghost_terms);
    const std::size_t bytes = (G * K + 2 * G) * sizeof(PrefilterReal);
    return (bytes + 7) / 8 * 8;
}

// OPL (float) und Klasse (1 Byte) pro Stützstelle.
constexpr std::size_t kBytesPerSample = sizeof(PrefilterReal) + sizeof(std::uint8_t);

std::size_t static_shared_bytes() {
    cudaFuncAttributes attr{};
    MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, prefilter_kernel));
    return attr.sharedSizeBytes;
}

// Größte Tranche für `usable` Byte (Bisektion, prefilter_device_bytes ist monoton in V).
int prefilter_capacity(std::size_t usable, const BatchTopology& topology, int limit) {
    const std::int64_t blocks = static_cast<std::int64_t>(topology.fields) * topology.wavelengths;
    const std::int64_t grid = kMaxGridBlocks / std::max<std::int64_t>(blocks, 1);
    int hi = static_cast<int>(std::min<std::int64_t>(limit, grid));
    if (hi < 1 || prefilter_device_bytes(topology, 1) > usable) {
        return 0;
    }
    int lo = 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        if (prefilter_device_bytes(topology, mid) <= usable) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

std::string mib(std::size_t bytes) {
    return std::to_string(static_cast<double>(bytes) / (1024.0 * 1024.0)) + " MiB";
}

} // namespace

PrefilterOutputLayout make_prefilter_output_layout(const BatchTopology& topology, int variants) {
    validate_topology(topology);
    if (variants < 0) {
        throw std::invalid_argument("make_prefilter_output_layout: negative Variantenzahl");
    }
    const auto V = static_cast<std::size_t>(variants);
    const auto B =
        static_cast<std::size_t>(topology.fields) * static_cast<std::size_t>(topology.wavelengths);
    PrefilterOutputLayout o;
    std::size_t offset = 0;
    const auto place = [&offset](BufferRegion& r, std::size_t bytes) {
        r.offset = offset;
        r.bytes = bytes;
        offset += align_bytes(bytes);
    };
    place(o.merit, V * sizeof(double));
    place(o.optical_length, V * sizeof(double));
    place(o.failed, V * sizeof(std::int64_t));
    place(o.uncertain_ok, V * sizeof(std::int64_t));
    place(o.uncertain_failed, V * sizeof(std::int64_t));
    place(o.merit_partial, V * B * sizeof(double));
    place(o.length_partial, V * B * sizeof(double));
    place(o.counts_partial, V * B * 3 * sizeof(std::int32_t));
    place(o.plan, static_cast<std::size_t>(topology.fields) * sizeof(ResidualFieldPlan));
    o.total_bytes = offset;
    return o;
}

std::size_t prefilter_device_bytes(const BatchTopology& topology, int variants) {
    return make_batch_layout(topology, variants).total_bytes +
           make_prefilter_output_layout(topology, variants).total_bytes;
}

int prefilter_max_samples(int ghost_plates, int ghost_terms) {
    if (ghost_plates < 0 || ghost_terms < 0) {
        throw std::invalid_argument("prefilter_max_samples: negative Zahl von Geisterplatten");
    }
    const std::size_t fixed = static_shared_bytes() + ghost_bytes_float(ghost_plates, ghost_terms);
    if (fixed >= kPrefilterMaxSharedBytes) {
        return 0;
    }
    return static_cast<int>((kPrefilterMaxSharedBytes - fixed) / kBytesPerSample);
}

DevicePrefilter::~DevicePrefilter() {
    release();
}

DevicePrefilter::DevicePrefilter(DevicePrefilter&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_), ran_(other.ran_),
      variants_(other.variants_), rays_(other.rays_), layout_(other.layout_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.ran_ = false;
}

DevicePrefilter& DevicePrefilter::operator=(DevicePrefilter&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        ran_ = other.ran_;
        variants_ = other.variants_;
        rays_ = other.rays_;
        layout_ = other.layout_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.ran_ = false;
    }
    return *this;
}

void DevicePrefilter::release() noexcept {
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

void DevicePrefilter::run(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                          double boundary_margin) {
    const SystemBatchView<double> view = batch.view(); // wirft bei leerem Batch
    check_residual_layout(batch.topology(), layout);
    if (!(boundary_margin >= 0.0) || !std::isfinite(boundary_margin)) {
        throw std::invalid_argument("DevicePrefilter::run: η muss endlich und >= 0 sein");
    }
    const int V = batch.variants();
    const int B = view.shape.fields * view.shape.wavelengths;
    const std::int64_t blocks = static_cast<std::int64_t>(V) * B;
    if (blocks > kMaxGridBlocks) {
        throw std::invalid_argument("DevicePrefilter::run: " + std::to_string(blocks) +
                                    " Blöcke überschreiten das größte Grid; Batch aufteilen");
    }
    int max_samples = 0;
    for (const ResidualFieldPlan& p : layout.plan()) {
        max_samples = std::max(max_samples, p.samples);
    }
    const PrefilterOutputLayout out = make_prefilter_output_layout(batch.topology(), V);

    ran_ = false;
    if (buffer_ != nullptr && device_ != batch.device()) {
        release();
    }
    device_ = batch.device();
    const detail::DeviceGuard guard(device_);

    const std::size_t ghost_bytes =
        ghost_bytes_float(view.shape.ghost_plates, view.shape.ghost_terms);
    const std::size_t dynamic =
        ghost_bytes + static_cast<std::size_t>(max_samples) * kBytesPerSample;
    if (static_shared_bytes() + dynamic > kPrefilterMaxSharedBytes) {
        throw std::invalid_argument(
            "DevicePrefilter::run: " + std::to_string(max_samples) +
            " Stützstellen pro Feld passen nicht in das Shared Memory (höchstens " +
            std::to_string(prefilter_max_samples(view.shape.ghost_plates, view.shape.ghost_terms)) +
            ")");
    }

    if (out.total_bytes > capacity_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            capacity_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, out.total_bytes));
        capacity_ = out.total_bytes;
    }
    MKO_CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer_) + out.plan.offset, layout.plan().data(),
                              out.plan.bytes, cudaMemcpyHostToDevice));

    if (blocks > 0) {
        PrefilterKernelArgs args{};
        args.batch = view;
        args.pupil = make_pupil_quadrature<double>(layout.settings().rings,
                                                   layout.settings().spokes, PupilMode::Half);
        args.plan = region_ptr<ResidualFieldPlan>(buffer_, out.plan);
        args.max_samples = max_samples;
        args.ghost_bytes = static_cast<unsigned>(ghost_bytes);
        args.boundary = static_cast<PrefilterReal>(boundary_margin);
        args.merit_partial = region_ptr<double>(buffer_, out.merit_partial);
        args.length_partial = region_ptr<double>(buffer_, out.length_partial);
        args.counts_partial = region_ptr<std::int32_t>(buffer_, out.counts_partial);
        prefilter_kernel<<<static_cast<unsigned>(blocks), kBlockThreads, dynamic>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());

        PrefilterFinalizeArgs fin{};
        fin.variants = V;
        fin.blocks_per_variant = B;
        fin.merit_partial = args.merit_partial;
        fin.length_partial = args.length_partial;
        fin.counts_partial = args.counts_partial;
        fin.merit = region_ptr<double>(buffer_, out.merit);
        fin.length = region_ptr<double>(buffer_, out.optical_length);
        fin.failed = region_ptr<std::int64_t>(buffer_, out.failed);
        fin.uncertain_ok = region_ptr<std::int64_t>(buffer_, out.uncertain_ok);
        fin.uncertain_failed = region_ptr<std::int64_t>(buffer_, out.uncertain_failed);
        const auto grid =
            static_cast<unsigned>((static_cast<std::int64_t>(V) + detail::kFinalizeThreads - 1) /
                                  detail::kFinalizeThreads);
        prefilter_finalize_kernel<<<grid, detail::kFinalizeThreads>>>(fin);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    layout_ = out;
    variants_ = V;
    rays_ = layout.rays_per_variant();
    ran_ = true;
}

PrefilterStats DevicePrefilter::download() const {
    if (!ran_) {
        throw std::logic_error("DevicePrefilter::download: kein Lauf vorhanden");
    }
    PrefilterStats s;
    const auto n = static_cast<std::size_t>(variants_);
    s.variants = variants_;
    s.rays_per_variant = rays_;
    s.merit.resize(n);
    s.optical_length.resize(n);
    s.failed.resize(n);
    s.uncertain_ok.resize(n);
    s.uncertain_failed.resize(n);
    download(s, 0);
    return s;
}

void DevicePrefilter::download(PrefilterStats& s, int first) const {
    if (!ran_) {
        throw std::logic_error("DevicePrefilter::download: kein Lauf vorhanden");
    }
    const auto end = static_cast<std::size_t>(first) + static_cast<std::size_t>(variants_);
    if (first < 0 || s.merit.size() < end || s.optical_length.size() < end ||
        s.failed.size() < end || s.uncertain_ok.size() < end || s.uncertain_failed.size() < end) {
        throw std::invalid_argument("DevicePrefilter::download: Ziel zu klein");
    }
    const detail::DeviceGuard guard(device_);
    const char* base = static_cast<const char*>(buffer_);
    const auto copy = [base](void* dst, const BufferRegion& region) {
        if (region.bytes > 0) {
            MKO_CUDA_CHECK(
                cudaMemcpy(dst, base + region.offset, region.bytes, cudaMemcpyDeviceToHost));
        }
    };
    const auto f = static_cast<std::size_t>(first);
    copy(s.merit.data() + f, layout_.merit);
    copy(s.optical_length.data() + f, layout_.optical_length);
    copy(s.failed.data() + f, layout_.failed);
    copy(s.uncertain_ok.data() + f, layout_.uncertain_ok);
    copy(s.uncertain_failed.data() + f, layout_.uncertain_failed);
}

// ---------------------------------------------------------------------------------------------
// TraceEngine::prefilter_variants
// ---------------------------------------------------------------------------------------------

PrefilterResult TraceEngine::prefilter_variants(const SystemBatchHost& batch,
                                                const PrefilterSettings& settings) {
    PrefilterResult out;
    prefilter_variants(batch, settings, out);
    return out;
}

TraceRunInfo TraceEngine::prefilter_variants(const SystemBatchHost& batch,
                                             const PrefilterSettings& settings,
                                             PrefilterResult& out) {
    // ---- Prüfen (Host, vor jeder Gerätearbeit) ----
    check_prefilter_options(settings.options);
    const BatchTopology& topology = batch.topology();
    const ResidualLayout layout(topology, settings.quadrature);
    batch.validate_values();
    const int V = batch.variants();
    if (!settings.extra_merit.empty() &&
        settings.extra_merit.size() != static_cast<std::size_t>(V)) {
        throw std::invalid_argument("TraceEngine: extra_merit muss leer sein oder " +
                                    std::to_string(V) + " Werte haben");
    }
    TraceRunInfo info;
    info.variants = V;
    PrefilterStats stats;
    stats.variants = V;
    stats.rays_per_variant = layout.rays_per_variant();
    const auto n = static_cast<std::size_t>(V);
    stats.merit.assign(n, 0.0);
    stats.optical_length.assign(n, 0.0);
    stats.failed.assign(n, 0);
    stats.uncertain_ok.assign(n, 0);
    stats.uncertain_failed.assign(n, 0);
    if (V > 0) {
        int samples = 0;
        for (const ResidualFieldPlan& p : layout.plan()) {
            samples = std::max(samples, p.samples);
        }
        const detail::DeviceGuard guard(device_);
        const int limit = prefilter_max_samples(topology.ghost_plates, topology.ghost_terms);
        if (samples > limit) {
            throw std::invalid_argument("TraceEngine: die Vorfilterung erlaubt höchstens " +
                                        std::to_string(limit) + " Stützstellen pro Feld (" +
                                        std::to_string(samples) + " angefordert)");
        }

        // ---- Tranchengröße: Puffer der Modi RESIDUALS und MERIT bleiben erhalten, solange
        // die Tranche dadurch nicht kleiner als V wird; sonst werden sie freigegeben ----
        const std::size_t available = available_bytes();
        const std::size_t other = residuals_.capacity_bytes() + merit_.capacity_bytes();
        std::size_t usable = usable_bytes(available - std::min(other, available), options_.reserve);
        int tranche = prefilter_capacity(usable, topology, V);
        if (other > 0 && tranche < V) {
            const std::size_t usable_all = usable_bytes(available, options_.reserve);
            const int tranche_all = prefilter_capacity(usable_all, topology, V);
            if (tranche_all > tranche) {
                residuals_.release();
                merit_.release();
                usable = usable_all;
                tranche = tranche_all;
            }
        }
        if (tranche < 1) {
            throw CudaError("TraceEngine: nicht einmal eine Variante passt für die Vorfilterung "
                            "in den Gerätespeicher (Gerät " +
                            std::to_string(device_) + ", verfügbar " + mib(available) +
                            ", nach Reserve " + mib(usable) + ", benötigt " +
                            mib(prefilter_device_bytes(topology, 1)) + ")");
        }
        // Reichen die Puffer nicht, vorher freigeben (alter und neuer Puffer nie gleichzeitig)
        if (make_batch_layout(topology, tranche).total_bytes > batch_.capacity_bytes() ||
            make_prefilter_output_layout(topology, tranche).total_bytes >
                prefilter_.capacity_bytes()) {
            batch_.release();
            prefilter_.release();
        }

        // ---- Tranchen: Upload, Kernel, Download der Rohdaten an die Stelle first ----
        for (int first = 0; first < V; first += tranche) {
            const int count = std::min(tranche, V - first);
            batch_.upload(batch, first, count);
            prefilter_.run(batch_, layout, settings.options.margins.boundary);
            prefilter_.download(stats, first);
            ++info.tranches;
        }
        info.tranche_variants = tranche;
        info.budget_bytes = usable;
    }
    info.device_bytes = device_bytes();
    out = prefilter_decide(std::move(stats), batch, layout, settings.options, settings.extra_merit);
    return info;
}

} // namespace mko::cuda
