// Kernel RESIDUALS (M2-T2): ein Block pro (Variante, Feld, Wellenlänge), Threads über die
// Pupillenstützstellen, Flächentabelle im Shared Memory, Kolbenabzug per deterministischer
// Block-Reduktion (ADR-0004, ADR-0007; docs/architektur.md, Abschnitt "Kernel RESIDUALS").
// Optional die Max-Reduktionen pro Fläche (M2-T3, Abschnitt "Max-Reduktionen").

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"
#include "wavefront_block.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko::cuda {

namespace {

using detail::kBlockThreads;

// Parameter des Kernels (per Wert, < 1 KB Parameterraum).
template <class R>
struct ResidualKernelArgs {
    SystemBatchView<R> batch;
    PupilQuadrature<R> pupil;      // Knoten und Gewichte (N, M); Pupillenart pro Feld aus plan
    const ResidualFieldPlan* plan; // [F]
    std::int64_t rays;             // Strahlresiduen pro Variante
    std::int64_t stride;           // Residuen pro Variante (rays + weitere)
    R* residuals;                  // [V][stride]
    R* opd;                        // [V][rays] oder nullptr
    std::uint8_t* status;          // [V][rays]
    R* extrema_partial;            // [V · F · L][S][2] (nur mit kExtrema)
};

// Block b = (v · F + f) · L + l. Alle Threads durchlaufen jede Schleife und jedes
// __syncthreads (versagte Strahlen und Threads ohne Stützstelle sind nur maskiert).
// kExtrema: zusätzlich Max-Reduktionen pro Fläche (ohne kExtrema entfällt jeder Aufwand).
template <class R, bool kExtrema>
__global__ void __launch_bounds__(kBlockThreads) residuals_kernel(const ResidualKernelArgs<R> a) {
    // Shared Memory (statisch, für S = kMaxSurfaces und R = double ca. 1,9 KB, mit kExtrema
    // zusätzlich 4 KB): Flächentabelle der Variante mit den Brechzahlen der Wellenlänge l,
    // Quadratur des Feldes, Teilsummen pro Warp, Kolben; dynamisch die Geisterplatten
    // (ghost_shared_bytes, höchstens 17 KiB).
    __shared__ detail::BlockTable<R> sh_table;
    __shared__ detail::PistonShared<R> sh_piston;
    __shared__ detail::ExtremaSharedIf<R, kExtrema> sh_extrema;
    extern __shared__ __align__(16) unsigned char sh_dynamic[];

    const int S = a.batch.shape.surfaces;
    const detail::BlockTuple tu =
        detail::block_tuple(a.batch.shape.fields, a.batch.shape.wavelengths);
    const int v = tu.v;
    const int f = tu.f;
    const int l = tu.l;
    const int tid = static_cast<int>(threadIdx.x);
    const ResidualFieldPlan plan = a.plan[f];

    detail::load_block_table(sh_table, a.batch, a.pupil, plan, tu, tid);
    const detail::BlockGhosts<R> ghosts = detail::load_block_ghosts(sh_dynamic, a.batch, tu, tid);
    __syncthreads();

    const SurfaceTableView<R> table = detail::block_table_view(sh_table, S, ghosts);
    const Vec3<R> center = batch_image_point(a.batch, table, v, f);
    const int n = plan.samples;
    const std::int64_t first = plan.offset + static_cast<std::int64_t>(l) * n; // Tupel (f, l)
    R* const residuals = a.residuals + static_cast<std::int64_t>(v) * a.stride + first;
    std::uint8_t* const status = a.status + static_cast<std::int64_t>(v) * a.rays + first;
    R* const opd =
        a.opd != nullptr ? a.opd + static_cast<std::int64_t>(v) * a.rays + first : nullptr;

    // Durchlauf 1: Trace, OPL vorläufig in den Residuenplatz, Teilsummen (W_k, W_k · OPL_k)
    // der gültigen Strahlen in aufsteigender Reihenfolge der eigenen Stützstellen.
    [[maybe_unused]] detail::ExtremaIf<R, kExtrema> extrema;
    [[maybe_unused]] detail::RayRecordIf<R, kExtrema> record;
    if constexpr (kExtrema) {
        extrema.reset(S);
    }
    R weight_ok = R(0);
    R weighted_opl = R(0);
    for (int k = tid; k < n; k += kBlockThreads) {
        const PupilSample<R> s = sample(sh_table.pupil, k);
        R opl = R(0);
        TraceStatus st;
        if constexpr (kExtrema) {
            st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl, &record);
        } else {
            st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl);
        }
        if (st == TraceStatus::Ok) {
            weight_ok += s.weight;
            weighted_opl += s.weight * opl;
            if constexpr (kExtrema) {
                extrema.add_ray(record, S);
            }
        }
        residuals[k] = opl;
        status[k] = static_cast<std::uint8_t>(st);
    }

    // Kolben per Baumreduktion (ohne gültigen Strahl: valid = false, alle OPD und Residuen des
    // Tupels werden 0, der Status bleibt gesetzt, wie opd_statistics).
    R piston;
    bool valid;
    detail::block_piston(sh_piston, weight_ok, weighted_opl, tid, piston, valid);
    const R scale = residual_scale(a.batch, f, l);

    // Durchlauf 2: OPD und Residuen derselben Stützstellen (gleicher Thread, eigene
    // Schreibzugriffe).
    for (int k = tid; k < n; k += kBlockThreads) {
        R e = R(0);
        if (valid && static_cast<TraceStatus>(status[k]) == TraceStatus::Ok) {
            e = residuals[k] - piston;
        }
        if (opd != nullptr) {
            opd[k] = e;
        }
        residuals[k] = ray_residual(scale, sample(sh_table.pupil, k), e);
    }

    if constexpr (kExtrema) {
        detail::block_extrema(sh_extrema, extrema, S, tid,
                              a.extrema_partial + tu.block * static_cast<std::int64_t>(S) * 2);
    }
}

using detail::region_ptr;

} // namespace

DeviceResiduals::~DeviceResiduals() {
    release();
}

DeviceResiduals::DeviceResiduals(DeviceResiduals&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_), ran_(other.ran_),
      opd_(other.opd_), extrema_(other.extrema_), variants_(other.variants_),
      surfaces_(other.surfaces_), rays_(other.rays_), stride_(other.stride_),
      layout_(other.layout_) {
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
    const int V = batch.variants();
    const std::int64_t blocks =
        static_cast<std::int64_t>(V) * view.shape.fields * view.shape.wavelengths;
    if (blocks > kMaxGridBlocks) {
        throw std::invalid_argument("DeviceResiduals::run: " + std::to_string(blocks) +
                                    " Blöcke überschreiten das größte Grid; Batch aufteilen");
    }

    OutputRequest request;
    request.rays_per_variant = layout.rays_per_variant();
    request.extra_residuals = options.extra_residuals;
    request.residuals = true;
    request.ray_status = true;
    request.opd = options.opd;
    request.merit = false;
    request.surface_max = options.extrema;
    const OutputLayout out = make_output_layout(batch.topology(), V, request);
    const BufferRegion plan_region{out.total_bytes,
                                   layout.plan().size() * sizeof(ResidualFieldPlan)};
    const std::size_t total = plan_region.offset + align_bytes(plan_region.bytes);

    ran_ = false;
    if (buffer_ != nullptr && device_ != batch.device()) {
        release();
    }
    device_ = batch.device();
    const detail::DeviceGuard guard(device_);
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
    if (options.extra_residuals > 0 && out.residuals.bytes > 0) {
        MKO_CUDA_CHECK(
            cudaMemset(static_cast<char*>(buffer_) + out.residuals.offset, 0, out.residuals.bytes));
    }

    if (blocks > 0) {
        ResidualKernelArgs<double> args{};
        args.batch = view;
        args.pupil = make_pupil_quadrature<double>(layout.settings().rings,
                                                   layout.settings().spokes, PupilMode::Half);
        args.plan = region_ptr<ResidualFieldPlan>(buffer_, plan_region);
        args.rays = layout.rays_per_variant();
        args.stride = layout.rays_per_variant() + options.extra_residuals;
        args.residuals = region_ptr<double>(buffer_, out.residuals);
        args.opd = region_ptr<double>(buffer_, out.opd);
        args.status = region_ptr<std::uint8_t>(buffer_, out.ray_status);
        args.extrema_partial = region_ptr<double>(buffer_, out.surface_max_partial);
        const std::size_t ghost_bytes = detail::ghost_shared_bytes(view.shape);
        if (ghost_bytes > detail::kMaxGhostSharedBytes) {
            throw std::invalid_argument("DeviceResiduals::run: Geisterdaten (" +
                                        std::to_string(ghost_bytes) +
                                        " Byte) passen nicht in den Shared Memory");
        }
        const auto grid = static_cast<unsigned>(blocks);
        if (options.extrema) {
            residuals_kernel<double, true><<<grid, kBlockThreads, ghost_bytes>>>(args);
        } else {
            residuals_kernel<double, false><<<grid, kBlockThreads, ghost_bytes>>>(args);
        }
        MKO_CUDA_CHECK(cudaGetLastError());
        if (options.extrema) {
            detail::ExtremaFinalizeArgs<double> fin{};
            fin.variants = V;
            fin.surfaces = view.shape.surfaces;
            fin.blocks_per_variant = view.shape.fields * view.shape.wavelengths;
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
    stride_ = rays_ + options.extra_residuals;
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
