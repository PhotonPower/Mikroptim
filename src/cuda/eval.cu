// Kernel EVAL (M8-T1): dichtes kartesisches Pupillenraster pro (Variante, Feld, Wellenlänge),
// OPD nach Kolbenabzug, Schnittpunkte mit der Bildebene und vier Feldstrahlen pro Tupel; jedes
// Versagen hart (docs/numerik.md, Abschnitt 11). Dazu DeviceEval und TraceEngine::evaluate
// (Tranchen nach dem freien Gerätespeicher). docs/architektur.md, Abschnitt "Modus EVAL".

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/eval.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/eval/eval_ray.hpp"
#include "mikroptim/merit/ray_residual.hpp"
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
using detail::region_ptr;

static_assert(kEvalThreads == kBlockThreads, "EVAL mit der Blockgröße von RESIDUALS");

struct EvalKernelArgs {
    SystemBatchView<double> batch;
    PupilQuadrature<double> no_pupil; // leer (load_block_table lädt keine Quadratur)
    const EvalPoint* points;          // [n]
    std::int64_t samples;             // n
    EvalReference reference;
    double field_offset;               // δ
    const double* reference_curvature; // [V·F]
    double* opd;                       // [V·B·n]
    double* spot_x;                    // [V·B·n]
    double* spot_y;                    // [V·B·n]
    std::uint8_t* status;              // [V·B·n]
    double* field_rays;                // [V·B·4·5]
    std::uint8_t* field_status;        // [V·B·4]
};

__device__ inline double quiet_nan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// Block b = (v · F + f) · L + l. 1. Flächentabelle (Brechzahlen der Wellenlänge l) und
// Geisterdaten in den Shared Memory, Thread 0 den Hauptstrahl als Bezugswert der OPL;
// 2. Durchlauf 1: eval_ray pro Rasterpunkt, Status und Spot in den globalen Speicher,
// OPL − Bezug vorläufig auf den OPD-Platz, Teilsummen (W, W·(OPL − Bezug)) der gültigen
// Strahlen; 3. Kolben (block_piston, feste Reihenfolge); 4. Durchlauf 2: OPD = (OPL − Bezug)
// − Kolben (NaN bei Versagen oder ohne gültigen Strahl); 5. Threads 0 … 3 die Feldstrahlen.
// Versagte Strahlen laufen maskiert weiter; alle Threads erreichen jedes __syncthreads.
template <bool kTransforms>
__global__ void __launch_bounds__(kBlockThreads) eval_kernel(const EvalKernelArgs a) {
    using R = double;
    __shared__ detail::BlockTable<R> sh_table;
    __shared__ detail::PistonShared<R> sh_piston;
    __shared__ R sh_offset;
    extern __shared__ __align__(16) unsigned char sh_dynamic[]; // Geisterplatten

    const int S = a.batch.shape.surfaces;
    const detail::BlockTuple tu =
        detail::block_tuple(a.batch.shape.fields, a.batch.shape.wavelengths);
    const int v = tu.v;
    const int f = tu.f;
    const int l = tu.l;
    const int tid = static_cast<int>(threadIdx.x);
    const ResidualFieldPlan no_plan{0, 0, PupilMode::Full};

    detail::load_block_table(sh_table, a.batch, a.no_pupil, no_plan, tu, tid);
    const detail::BlockGhosts<R> ghosts =
        detail::load_block_ghosts<R>(sh_dynamic, a.batch, tu, tid);
    // Dezentrierung und Verkippung hinter den Geisterdaten (nur Instanz kTransforms, M8-T3)
    const detail::KernelTable<R, kTransforms> table = detail::block_kernel_table<R, kTransforms>(
        sh_table, S, ghosts,
        reinterpret_cast<R*>(sh_dynamic + detail::ghost_shared_bytes(a.batch.shape)), a.batch, v,
        tid);
    __syncthreads();

    const Vec3<R> center = batch_image_point(a.batch, table, v, f);
    const R kappa = a.reference_curvature[a.batch.shape.field_at(v, f)];
    const std::int64_t n = a.samples;
    const std::int64_t base = tu.block * n;
    const R no_value = quiet_nan();

    // Bezugswert der OPL: Hauptstrahl (Pupillenmitte, Thread 0), 0 wenn er versagt; der Kolben
    // summiert OPL − Bezug (wie eval_cpu).
    if (tid == 0) {
        const EvalRay<R> chief =
            eval_ray(a.batch, table, 0, v, f, l, R(0), R(0), center, a.reference, kappa);
        sh_offset = chief.status == TraceStatus::Ok ? chief.opl : R(0);
    }
    __syncthreads();
    const R offset = sh_offset;

    R weight_ok = R(0);
    R weighted_opl = R(0);
    for (std::int64_t k = tid; k < n; k += kBlockThreads) {
        const EvalPoint p = a.points[k];
        const EvalRay<R> r =
            eval_ray(a.batch, table, 0, v, f, l, p.x, p.y, center, a.reference, kappa);
        const bool ok = r.status == TraceStatus::Ok;
        const R opl = r.opl - offset;
        if (ok) {
            weight_ok += p.weight;
            weighted_opl += p.weight * opl;
        }
        a.status[base + k] = static_cast<std::uint8_t>(r.status);
        a.opd[base + k] = opl;
        a.spot_x[base + k] = ok ? r.image.x - center.x : no_value;
        a.spot_y[base + k] = ok ? r.image.y - center.y : no_value;
    }

    R piston;
    bool valid;
    detail::block_piston(sh_piston, weight_ok, weighted_opl, tid, piston, valid);

    for (std::int64_t k = tid; k < n; k += kBlockThreads) {
        const bool ok = static_cast<TraceStatus>(a.status[base + k]) == TraceStatus::Ok;
        a.opd[base + k] = valid && ok ? a.opd[base + k] - piston : no_value;
    }

    if (tid < kEvalFieldRays) {
        R x;
        R y;
        eval_field_ray_point(tid, R(a.field_offset), x, y);
        const EvalRay<R> r = eval_ray(a.batch, table, 0, v, f, l, x, y, center, a.reference, kappa);
        const bool ok = r.status == TraceStatus::Ok;
        const std::int64_t j = tu.block * kEvalFieldRays + tid;
        double* fr = a.field_rays + j * kEvalFieldRayValues;
        fr[0] = ok ? r.image.x - center.x : no_value;
        fr[1] = ok ? r.image.y - center.y : no_value;
        fr[2] = ok ? r.d.x : no_value;
        fr[3] = ok ? r.d.y : no_value;
        fr[4] = ok ? r.d.z : no_value;
        a.field_status[j] = static_cast<std::uint8_t>(r.status);
    }
}

// Größte Tranche für `usable` Byte (Bisektion, eval_device_bytes ist monoton in V).
int eval_capacity(std::size_t usable, const BatchTopology& topology, std::int64_t samples,
                  int limit) {
    const std::int64_t blocks = static_cast<std::int64_t>(topology.fields) * topology.wavelengths;
    const std::int64_t grid = kMaxGridBlocks / std::max<std::int64_t>(blocks, 1);
    int hi = static_cast<int>(std::min<std::int64_t>(limit, grid));
    if (hi < 1 || eval_device_bytes(topology, 1, samples) > usable) {
        return 0;
    }
    int lo = 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        if (eval_device_bytes(topology, mid, samples) <= usable) {
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

void copy_region(void* dst, const void* buffer, const BufferRegion& region) {
    if (region.bytes > 0) {
        MKO_CUDA_CHECK(cudaMemcpy(dst, static_cast<const char*>(buffer) + region.offset,
                                  region.bytes, cudaMemcpyDeviceToHost));
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------
// DeviceEval
// ---------------------------------------------------------------------------------------------

DeviceEval::~DeviceEval() {
    release();
}

DeviceEval::DeviceEval(DeviceEval&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_), ran_(other.ran_),
      variants_(other.variants_), fields_(other.fields_), wavelengths_(other.wavelengths_),
      samples_(other.samples_), layout_(other.layout_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.ran_ = false;
}

DeviceEval& DeviceEval::operator=(DeviceEval&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        ran_ = other.ran_;
        variants_ = other.variants_;
        fields_ = other.fields_;
        wavelengths_ = other.wavelengths_;
        samples_ = other.samples_;
        layout_ = other.layout_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.ran_ = false;
    }
    return *this;
}

void DeviceEval::release() noexcept {
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

void DeviceEval::run(const DeviceSystemBatch& batch, const EvalGrid& grid,
                     const EvalSettings& settings, std::span<const double> reference_curvature) {
    const SystemBatchView<double> view = batch.view(); // wirft bei leerem Batch
    check_eval_settings(settings);
    const std::int64_t n = grid.samples();
    if (n < 1) {
        throw std::invalid_argument("DeviceEval::run: leeres Raster");
    }
    const int V = batch.variants();
    const int F = view.shape.fields;
    const int L = view.shape.wavelengths;
    if (reference_curvature.size() != static_cast<std::size_t>(V) * static_cast<std::size_t>(F)) {
        throw std::invalid_argument("DeviceEval::run: " + std::to_string(V * F) +
                                    " Krümmungen der Referenzsphäre erwartet");
    }
    const std::int64_t blocks = static_cast<std::int64_t>(V) * F * L;
    if (blocks > kMaxGridBlocks) {
        throw std::invalid_argument("DeviceEval::run: " + std::to_string(blocks) +
                                    " Blöcke überschreiten das größte Grid; Batch aufteilen");
    }
    const EvalOutputLayout out = make_eval_output_layout(batch.topology(), V, n);

    ran_ = false;
    if (buffer_ != nullptr && device_ != batch.device()) {
        release();
    }
    device_ = batch.device();
    const detail::DeviceGuard guard(device_);

    if (out.total_bytes > capacity_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            capacity_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, out.total_bytes));
        capacity_ = out.total_bytes;
    }
    char* base = static_cast<char*>(buffer_);
    MKO_CUDA_CHECK(cudaMemcpy(base + out.points.offset, grid.points.data(), out.points.bytes,
                              cudaMemcpyHostToDevice));
    MKO_CUDA_CHECK(cudaMemcpy(base + out.reference_curvature.offset, reference_curvature.data(),
                              out.reference_curvature.bytes, cudaMemcpyHostToDevice));

    if (blocks > 0) {
        EvalKernelArgs args{};
        args.batch = view;
        args.no_pupil = PupilQuadrature<double>{};
        args.points = region_ptr<EvalPoint>(buffer_, out.points);
        args.samples = n;
        args.reference = settings.reference;
        args.field_offset = settings.field_ray_offset;
        args.reference_curvature = region_ptr<double>(buffer_, out.reference_curvature);
        args.opd = region_ptr<double>(buffer_, out.opd);
        args.spot_x = region_ptr<double>(buffer_, out.spot_x);
        args.spot_y = region_ptr<double>(buffer_, out.spot_y);
        args.status = region_ptr<std::uint8_t>(buffer_, out.status);
        args.field_rays = region_ptr<double>(buffer_, out.field_rays);
        args.field_status = region_ptr<std::uint8_t>(buffer_, out.field_status);
        const std::size_t dynamic =
            detail::ghost_shared_bytes(view.shape) + detail::transform_shared_bytes(view);
        if (view.decenter_tilt != nullptr) { // nicht koaxial (M8-T3)
            eval_kernel<true><<<static_cast<unsigned>(blocks), kBlockThreads, dynamic>>>(args);
        } else {
            eval_kernel<false><<<static_cast<unsigned>(blocks), kBlockThreads, dynamic>>>(args);
        }
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    layout_ = out;
    variants_ = V;
    fields_ = F;
    wavelengths_ = L;
    samples_ = n;
    ran_ = true;
}

void DeviceEval::download(EvalResult& r, int first) const {
    if (!ran_) {
        throw std::logic_error("DeviceEval::download: kein Lauf vorhanden");
    }
    const std::size_t tuples_before = static_cast<std::size_t>(first) *
                                      static_cast<std::size_t>(fields_) *
                                      static_cast<std::size_t>(wavelengths_);
    const std::size_t tuples = static_cast<std::size_t>(variants_) *
                               static_cast<std::size_t>(fields_) *
                               static_cast<std::size_t>(wavelengths_);
    const auto n = static_cast<std::size_t>(samples_);
    const std::size_t rays_end = (tuples_before + tuples) * n;
    if (first < 0 || r.fields != fields_ || r.wavelengths != wavelengths_ ||
        r.samples() != samples_ || r.opd.size() < rays_end || r.spot_x.size() < rays_end ||
        r.spot_y.size() < rays_end || r.status.size() < rays_end ||
        r.field_rays.size() < (tuples_before + tuples) * kEvalFieldRays * kEvalFieldRayValues ||
        r.field_status.size() < (tuples_before + tuples) * kEvalFieldRays) {
        throw std::invalid_argument("DeviceEval::download: Ziel passt nicht");
    }
    const detail::DeviceGuard guard(device_);
    const std::size_t ray0 = tuples_before * n;
    copy_region(r.opd.data() + ray0, buffer_, layout_.opd);
    copy_region(r.spot_x.data() + ray0, buffer_, layout_.spot_x);
    copy_region(r.spot_y.data() + ray0, buffer_, layout_.spot_y);
    copy_region(r.status.data() + ray0, buffer_, layout_.status);
    copy_region(r.field_rays.data() + tuples_before * kEvalFieldRays * kEvalFieldRayValues, buffer_,
                layout_.field_rays);
    copy_region(r.field_status.data() + tuples_before * kEvalFieldRays, buffer_,
                layout_.field_status);
}

// ---------------------------------------------------------------------------------------------
// TraceEngine::evaluate
// ---------------------------------------------------------------------------------------------

EvalResult TraceEngine::evaluate(const SystemBatchHost& batch, const EvalSettings& settings) {
    EvalResult out;
    evaluate(batch, settings, out);
    return out;
}

TraceRunInfo TraceEngine::evaluate(const SystemBatchHost& batch, const EvalSettings& settings,
                                   EvalResult& out) {
    // ---- Prüfen und Host-Ausgabe anlegen (vor jeder Gerätearbeit) ----
    check_eval_settings(settings, batch);
    batch.validate_values();
    const BatchTopology& topology = batch.topology();
    prepare_eval_result(out, batch, settings, make_eval_grid(settings.grid, settings.half_pupil),
                        eval_reference_geometry(batch, settings.reference));
    const int V = batch.variants();
    const std::int64_t n = out.samples();
    TraceRunInfo info;
    info.variants = V;
    if (V == 0) {
        info.device_bytes = device_bytes();
        return info;
    }

    // ---- Tranchengröße: die Puffer der anderen Modi bleiben erhalten, solange die Tranche
    // dadurch nicht kleiner als V wird; sonst werden sie freigegeben (wie die Vorfilterung) ----
    const std::size_t available = available_bytes();
    const std::size_t other =
        residuals_.capacity_bytes() + merit_.capacity_bytes() + prefilter_.capacity_bytes();
    std::size_t usable = usable_bytes(available - std::min(other, available), options_.reserve);
    int tranche = eval_capacity(usable, topology, n, V);
    if (other > 0 && tranche < V) {
        const std::size_t usable_all = usable_bytes(available, options_.reserve);
        const int tranche_all = eval_capacity(usable_all, topology, n, V);
        if (tranche_all > tranche) {
            residuals_.release();
            merit_.release();
            prefilter_.release();
            usable = usable_all;
            tranche = tranche_all;
        }
    }
    if (tranche < 1) {
        throw CudaError("TraceEngine: nicht einmal eine Variante passt für EVAL in den "
                        "Gerätespeicher (Gerät " +
                        std::to_string(device_) + ", verfügbar " + mib(available) +
                        ", nach Reserve " + mib(usable) + ", benötigt " +
                        mib(eval_device_bytes(topology, 1, n)) + "); Raster verkleinern");
    }
    // Reicht der Batchpuffer nicht, vorher freigeben (alter und neuer Puffer nie gleichzeitig)
    if (make_batch_layout(topology, tranche).total_bytes > batch_.capacity_bytes()) {
        batch_.release();
    }

    // ---- Tranchen: Upload, Kernel, Download an die Stelle first ----
    // Der Ausgabepuffer lebt nur während des Aufrufs (EVAL ist selten, die Puffer groß).
    DeviceEval eval;
    const auto F = static_cast<std::size_t>(batch.fields());
    const std::span<const double> curvature(out.reference_curvature);
    const detail::DeviceGuard guard(device_);
    for (int first = 0; first < V; first += tranche) {
        const int count = std::min(tranche, V - first);
        batch_.upload(batch, first, count);
        eval.run(batch_, out.grid, settings,
                 curvature.subspan(static_cast<std::size_t>(first) * F,
                                   static_cast<std::size_t>(count) * F));
        eval.download(out, first);
        ++info.tranches;
    }
    info.tranche_variants = tranche;
    info.budget_bytes = usable;
    info.device_bytes = device_bytes() + eval.capacity_bytes();
    return info;
}

} // namespace mko::cuda
