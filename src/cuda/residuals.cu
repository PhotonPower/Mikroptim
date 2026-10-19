// Kernel RESIDUALS (M2-T2): seit M7-T4 ein Block pro Paket aufeinanderfolgender Tupel
// (Feld, Wellenlänge) einer Variante, Threads über die Pupillenstrahlen des Pakets,
// Flächentabelle im Shared Memory, Kolbenabzug pro Tupel per deterministischer Reduktion
// (ADR-0004, ADR-0007; docs/architektur.md, Abschnitte "Kernel-Abbildung" und "Kernel RESIDUALS").
// Optional die Max-Reduktionen pro Fläche (M2-T3, Abschnitt "Max-Reduktionen"). Rechentyp
// double oder df64 (M7-T1, ResidualRunOptions::precision); Batch und Ausgaben sind FP64.

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/residuals.hpp"
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
#include <utility>
#include <vector>

namespace mko::cuda {

namespace {

using detail::kBlockThreads;

// Parameter des Kernels (per Wert, < 1 KB Parameterraum). Batch und Ausgaben in FP64,
// unabhängig vom Rechentyp.
struct ResidualKernelArgs {
    SystemBatchView<double> batch;
    PupilQuadrature<double> pupil; // Knoten und Gewichte (N, M); Pupillenart pro Feld aus plan
    const ResidualFieldPlan* plan; // [F]
    const TuplePack* packs;        // [P], gleich für alle Varianten (M7-T4)
    int packs_per_variant;         // P
    std::int64_t rays;             // Strahlresiduen pro Variante
    std::int64_t stride;           // Residuen pro Variante (rays + Strafresiduen + weitere)
    int soft;                      // 1: weiches Versagen, Strafresiduen ab Zeile rays (M4-T2)
    double fail_weight;            // w_fail [Wellen²]
    double* residuals;             // [V][stride]
    double* opd;                   // [V][rays] oder nullptr
    std::uint8_t* status;          // [V][rays]
    double* extrema_partial;       // [V · P][S][2] (nur mit kExtrema)
};

// Block b = v · P + p: Paket p der Variante v (M7-T4). Alle Threads durchlaufen jede Schleife
// und jedes __syncthreads (versagte Strahlen und Threads ohne Strahl sind nur maskiert).
// R: Rechentyp (double oder df64). kExtrema: zusätzlich Max-Reduktionen pro Fläche (ohne
// kExtrema entfällt jeder Aufwand).
template <class R, bool kExtrema>
__global__ void MKO_TRACE_LAUNCH_BOUNDS residuals_kernel(const ResidualKernelArgs a) {
    // Shared Memory: statisch Flächentabelle, Quadratur, Angaben pro Tupel, Kolben und
    // Teilsummen (PackShared, für R = double ca. 3,6 KB; mit kExtrema zusätzlich 4 KB);
    // dynamisch Geisterdaten und Brechzahlen des Wellenlängenfensters (pack_table_bytes).
    __shared__ detail::PackShared<R> sh;
    __shared__ detail::ExtremaSharedIf<R, kExtrema> sh_extrema;
    extern __shared__ __align__(16) unsigned char sh_dynamic[];

    const int S = a.batch.shape.surfaces;
    const auto block = static_cast<std::int64_t>(blockIdx.x);
    const int v = static_cast<int>(block / a.packs_per_variant);
    const TuplePack pack = a.packs[block % a.packs_per_variant];
    const int tid = static_cast<int>(threadIdx.x);

    const SurfaceTableView<R> table = detail::load_pack(sh, sh_dynamic, a.batch, a.pupil, a.plan,
                                                        pack, v, a.soft != 0, a.fail_weight, tid);
    __syncthreads();

    // Strahlen des Pakets liegen ab Zeile first_row hintereinander (Strahl r im Paket).
    double* const residuals =
        a.residuals + static_cast<std::int64_t>(v) * a.stride + pack.first_row;
    std::uint8_t* const status = a.status + static_cast<std::int64_t>(v) * a.rays + pack.first_row;
    double* const opd =
        a.opd != nullptr ? a.opd + static_cast<std::int64_t>(v) * a.rays + pack.first_row : nullptr;
    // Vorläufige OPL im Rechentyp auf dem Residuenplatz (sizeof(R) == sizeof(double)); für
    // R = df64 ohne Rundung nach double zwischen den Durchläufen.
    R* const opl_scratch = reinterpret_cast<R*>(residuals);

    // Durchlauf 1: Trace, OPL vorläufig in den Residuenplatz, Status; Strahlen dicht über die
    // Threads verteilt (r = tid, tid + 128, …), Tupel und Stützstelle aus r.
    // Max-Reduktionen: Akkumulator mit Doppelpuffer, zugleich Beobachter (PendingExtrema)
    [[maybe_unused]] detail::PendingExtremaIf<R, kExtrema> extrema;
    if constexpr (kExtrema) {
        extrema.reset(S);
    }
    // Weiches Versagen (M4-T2): Strafresiduen direkt in Durchlauf 1 (sie hängen nicht vom
    // Kolben ab), Ablage ab Zeile rays derselben Variante.
    double* const penalties = a.soft ? residuals + a.rays : nullptr;
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
        if (penalties != nullptr) {
            penalties[r] = static_cast<double>(ray_penalty(sh.pscale[j], s, st, soft.penalty));
        }
        opl_scratch[r] = opl;
        status[r] = static_cast<std::uint8_t>(st);
    }
    __syncthreads();

    // Kolben pro Tupel in der Reihenfolge von M2 (pack_piston; ohne gültigen Strahl:
    // valid = false, alle OPD und Residuen des Tupels werden 0, der Status bleibt gesetzt, wie
    // opd_statistics).
    detail::pack_piston(sh, pack.tuples, opl_scratch, status, tid);

    // Durchlauf 2: OPD und Residuen derselben Strahlen (gleicher Thread, eigene
    // Schreibzugriffe), Ausgabe in FP64.
    for (int r = tid; r < pack.rays; r += kBlockThreads) {
        const int j = detail::pack_tuple_of(sh, pack.tuples, r);
        R e = R(0);
        if (sh.valid[j] != 0 && in_piston(static_cast<TraceStatus>(status[r]))) {
            e = opl_scratch[r] - sh.piston[j];
        }
        if (opd != nullptr) {
            opd[r] = static_cast<double>(e);
        }
        residuals[r] = static_cast<double>(
            ray_residual(sh.scale[j], detail::pack_sample(sh, j, r - sh.start[j]), e));
    }

    if constexpr (kExtrema) {
        detail::block_extrema(sh_extrema, extrema, S, tid,
                              a.extrema_partial + block * static_cast<std::int64_t>(S) * 2);
    }
}

// Startet den Kernel mit Rechentyp R (Gerät ist gesetzt) und prüft den Start.
template <class R>
void launch_residuals(const ResidualKernelArgs& args, bool extrema, unsigned grid,
                      std::size_t dynamic) {
    if (extrema) {
        residuals_kernel<R, true><<<grid, kBlockThreads, dynamic>>>(args);
    } else {
        residuals_kernel<R, false><<<grid, kBlockThreads, dynamic>>>(args);
    }
    MKO_CUDA_CHECK(cudaGetLastError());
}

// Statisches Shared Memory der Kernel-Variante (Kernelattribut).
template <class R>
std::size_t residuals_static_shared(bool extrema) {
    cudaFuncAttributes attr{};
    if (extrema) {
        MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, residuals_kernel<R, true>));
    } else {
        MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, residuals_kernel<R, false>));
    }
    return attr.sharedSizeBytes;
}

using detail::region_ptr;

} // namespace

DeviceResiduals::~DeviceResiduals() {
    release();
}

DeviceResiduals::DeviceResiduals(DeviceResiduals&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_), ran_(other.ran_),
      opd_(other.opd_), extrema_(other.extrema_), variants_(other.variants_),
      surfaces_(other.surfaces_), rays_(other.rays_), penalty_rows_(other.penalty_rows_),
      stride_(other.stride_), layout_(other.layout_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.ran_ = false;
}

DeviceResiduals& DeviceResiduals::operator=(DeviceResiduals&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        ran_ = other.ran_;
        opd_ = other.opd_;
        extrema_ = other.extrema_;
        variants_ = other.variants_;
        surfaces_ = other.surfaces_;
        rays_ = other.rays_;
        penalty_rows_ = other.penalty_rows_;
        stride_ = other.stride_;
        layout_ = other.layout_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.ran_ = false;
    }
    return *this;
}

void DeviceResiduals::release() noexcept {
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

void DeviceResiduals::run(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                          const ResidualRunOptions& options) {
    const SystemBatchView<double> view = batch.view(); // wirft bei leerem Batch
    check_residual_layout(batch.topology(), layout);
    if (options.extra_residuals < 0) {
        throw std::invalid_argument("DeviceResiduals::run: negative Zahl weiterer Residuen");
    }
    check_precision(options.precision);
    if (options.pack_rays < 1) {
        throw std::invalid_argument("DeviceResiduals::run: pack_rays muss >= 1 sein");
    }
    const int V = batch.variants();
    const std::int64_t blocks =
        static_cast<std::int64_t>(V) * view.shape.fields * view.shape.wavelengths;
    if (blocks > kMaxGridBlocks) {
        throw std::invalid_argument("DeviceResiduals::run: " + std::to_string(blocks) +
                                    " Blöcke überschreiten das größte Grid; Batch aufteilen");
    }

    OutputRequest request;
    request.rays_per_variant = layout.rays_per_variant();
    request.penalty_residuals = layout.penalty_rows();
    request.extra_residuals = options.extra_residuals;
    request.residuals = true;
    request.ray_status = true;
    request.opd = options.opd;
    request.merit = false;
    request.surface_max = options.extrema;
    const OutputLayout out = make_output_layout(batch.topology(), V, request);
    const BufferRegion plan_region{out.total_bytes,
                                   layout.plan().size() * sizeof(ResidualFieldPlan)};
    const std::size_t ghost_bytes = detail::ghost_shared_bytes(view.shape);
    if (ghost_bytes > detail::kMaxGhostSharedBytes) {
        throw std::invalid_argument("DeviceResiduals::run: Geisterdaten (" +
                                    std::to_string(ghost_bytes) +
                                    " Byte) passen nicht in den Shared Memory");
    }

    ran_ = false;
    if (buffer_ != nullptr && device_ != batch.device()) {
        release();
    }
    device_ = batch.device();
    const detail::DeviceGuard guard(device_);

    // Tupelpakete (M7-T4): nur die Tabellen im dynamischen Shared Memory, OPL und Status liegen
    // im Ausgabepuffer.
    const bool df = options.precision == Precision::Df64;
    const std::size_t static_bytes = df ? residuals_static_shared<df64>(options.extrema)
                                        : residuals_static_shared<double>(options.extrema);
    const std::vector<TuplePack> packs = make_tuple_packs(
        layout, view.shape, options.pack_rays, 0,
        detail::kBlockMaxSharedBytes - std::min(static_bytes, detail::kBlockMaxSharedBytes));
    const std::size_t dynamic = pack_dynamic_bytes(packs, view.shape, 0);
    const auto P = static_cast<int>(packs.size());
    // Platz für F · L Pakete (höchstens so viele), damit der Puffer zur Abschätzung der
    // TraceEngine passt (fixed_run_bytes) und über Läufe wiederverwendet wird.
    const BufferRegion pack_region{plan_region.offset + align_bytes(plan_region.bytes),
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
    MKO_CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer_) + pack_region.offset, packs.data(),
                              pack_region.bytes, cudaMemcpyHostToDevice));
    if (options.extra_residuals > 0 && out.residuals.bytes > 0) {
        MKO_CUDA_CHECK(
            cudaMemset(static_cast<char*>(buffer_) + out.residuals.offset, 0, out.residuals.bytes));
    }

    if (blocks > 0) {
        ResidualKernelArgs args{};
        args.batch = view;
        args.pupil = make_pupil_quadrature<double>(layout.settings().rings,
                                                   layout.settings().spokes, PupilMode::Half);
        args.plan = region_ptr<ResidualFieldPlan>(buffer_, plan_region);
        args.packs = region_ptr<TuplePack>(buffer_, pack_region);
        args.packs_per_variant = P;
        args.rays = layout.rays_per_variant();
        args.stride = layout.trace_rows() + options.extra_residuals;
        args.soft = layout.soft_failure() ? 1 : 0;
        args.fail_weight = layout.settings().fail_weight;
        args.residuals = region_ptr<double>(buffer_, out.residuals);
        args.opd = region_ptr<double>(buffer_, out.opd);
        args.status = region_ptr<std::uint8_t>(buffer_, out.ray_status);
        args.extrema_partial = region_ptr<double>(buffer_, out.surface_max_partial);
        const auto grid = static_cast<unsigned>(static_cast<std::int64_t>(V) * P);
        if (df) {
            launch_residuals<df64>(args, options.extrema, grid, dynamic);
        } else {
            launch_residuals<double>(args, options.extrema, grid, dynamic);
        }
        if (options.extrema) {
            detail::ExtremaFinalizeArgs<double> fin{};
            fin.variants = V;
            fin.surfaces = view.shape.surfaces;
            fin.blocks_per_variant = P;
            fin.partial = args.extrema_partial;
            fin.out = region_ptr<double>(buffer_, out.surface_max);
            detail::launch_extrema_finalize(fin);
        }
        MKO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    layout_ = out;
    variants_ = V;
    surfaces_ = view.shape.surfaces;
    rays_ = layout.rays_per_variant();
    penalty_rows_ = layout.penalty_rows();
    stride_ = layout.trace_rows() + options.extra_residuals;
    opd_ = options.opd;
    extrema_ = options.extrema;
    ran_ = true;
}

ResidualResult DeviceResiduals::download() const {
    ResidualResult r;
    download(r);
    return r;
}

void DeviceResiduals::download(ResidualResult& r) const {
    if (!ran_) {
        throw std::logic_error("DeviceResiduals::download: kein Lauf vorhanden");
    }
    r.variants = variants_;
    r.rays_per_variant = rays_;
    r.penalty_rows = penalty_rows_;
    r.stride = stride_;
    r.residuals.resize(layout_.residuals.bytes / sizeof(double));
    r.status.resize(layout_.ray_status.bytes);
    r.opd.resize(opd_ ? layout_.opd.bytes / sizeof(double) : 0);
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
    copy(r.residuals.data(), layout_.residuals);
    copy(r.status.data(), layout_.ray_status);
    if (opd_) {
        copy(r.opd.data(), layout_.opd);
    }
    if (extrema_) {
        copy(r.extrema.values.data(), layout_.surface_max);
    }
}

const double* DeviceResiduals::device_residuals() const {
    return ran_ ? region_ptr<double>(static_cast<const void*>(buffer_), layout_.residuals)
                : nullptr;
}

const std::uint8_t* DeviceResiduals::device_status() const {
    return ran_ ? region_ptr<std::uint8_t>(static_cast<const void*>(buffer_), layout_.ray_status)
                : nullptr;
}

const double* DeviceResiduals::device_opd() const {
    return ran_ && opd_ ? region_ptr<double>(static_cast<const void*>(buffer_), layout_.opd)
                        : nullptr;
}

const double* DeviceResiduals::device_extrema() const {
    return ran_ && extrema_
               ? region_ptr<double>(static_cast<const void*>(buffer_), layout_.surface_max)
               : nullptr;
}

ResidualResult run_residuals(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                             const ResidualRunOptions& options) {
    DeviceResiduals d;
    d.run(batch, layout, options);
    return d.download();
}

} // namespace mko::cuda
