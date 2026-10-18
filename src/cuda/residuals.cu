// Kernel RESIDUALS (M2-T2): ein Block pro (Variante, Feld, Wellenlänge), Threads über die
// Pupillenstützstellen, Flächentabelle im Shared Memory, Kolbenabzug per deterministischer
// Block-Reduktion (ADR-0004, ADR-0007; docs/architektur.md, Abschnitt "Kernel RESIDUALS").

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko::cuda {

namespace {

constexpr int kWarpSize = 32;
constexpr int kResidualWarps = kResidualThreads / kWarpSize;
static_assert(kResidualThreads % kWarpSize == 0, "Blockgröße muss ein Vielfaches von 32 sein");
static_assert(kMaxPupilRings <= kResidualThreads, "Knoten werden von je einem Thread geladen");

// Dynamischer Shared Memory für die Geisterplatten der Variante: Koeffizienten [G·K],
// Brechzahlen der Gläser bei der Wellenlänge des Blocks [G] und Normierungsradien [G].
// Höchstens (kMaxSurfaces · kMaxGhostTerms + 2 · kMaxSurfaces) · 8 B = 17 KiB; zusammen mit
// dem statischen Teil (ca. 1,9 KB) unter 48 KB auf beiden Karten (kein Opt-in nötig).
constexpr std::size_t kMaxGhostSharedBytes =
    (static_cast<std::size_t>(kMaxSurfaces) * kMaxGhostTerms + 2 * kMaxSurfaces) * sizeof(double);
static_assert(kMaxGhostSharedBytes + 4096 <= 48 * 1024, "Shared Memory pro Block über 48 KB");

std::size_t ghost_shared_bytes(const BatchShape& shape) {
    const std::size_t G = static_cast<std::size_t>(shape.ghost_plates);
    const std::size_t K = static_cast<std::size_t>(shape.ghost_terms);
    return (G * K + 2 * G) * sizeof(double);
}

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
};

// Summe über die 32 Lanes eines Warps als feste Baumreduktion (Abstände 16, 8, 4, 2, 1);
// das Ergebnis steht in Lane 0. Alle Lanes müssen teilnehmen.
template <class R>
__device__ inline R warp_sum(R x) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

// Block b = (v · F + f) · L + l. Alle Threads durchlaufen jede Schleife und jedes
// __syncthreads (versagte Strahlen und Threads ohne Stützstelle sind nur maskiert).
template <class R>
__global__ void __launch_bounds__(kResidualThreads)
    residuals_kernel(const ResidualKernelArgs<R> a) {
    // Shared Memory (statisch, für S = kMaxSurfaces und R = double ca. 1,9 KB):
    // Flächentabelle der Variante mit den Brechzahlen der Wellenlänge l, Quadratur des Feldes,
    // Teilsummen pro Warp, Kolben; dazu dynamisch die Geisterplatten (höchstens 17 KiB).
    __shared__ SurfaceKind sh_type[kMaxSurfaces];
    __shared__ R sh_curvature[kMaxSurfaces];
    __shared__ R sh_thickness[kMaxSurfaces];
    __shared__ R sh_index[kMaxSurfaces];
    __shared__ PupilQuadrature<R> sh_pupil;
    __shared__ R sh_weight[kResidualWarps];
    __shared__ R sh_weighted_opl[kResidualWarps];
    __shared__ R sh_piston;
    __shared__ int sh_valid;
    // Geisterplatten (dynamisch, ghost_shared_bytes): Koeffizienten, Brechzahlen, Radien
    extern __shared__ __align__(16) unsigned char sh_dynamic[];

    const int S = a.batch.shape.surfaces;
    const int L = a.batch.shape.wavelengths;
    const int F = a.batch.shape.fields;
    const std::int64_t block = static_cast<std::int64_t>(blockIdx.x);
    const int l = static_cast<int>(block % L);
    const std::int64_t vf = block / L;
    const int f = static_cast<int>(vf % F);
    const int v = static_cast<int>(vf / F);
    const int tid = static_cast<int>(threadIdx.x);
    const ResidualFieldPlan plan = a.plan[f];

    // Flächentabelle der Variante v laden (koaleszierend über s)
    for (int s = tid; s < S; s += kResidualThreads) {
        sh_type[s] = a.batch.type[s];
        sh_curvature[s] = a.batch.curvature[a.batch.shape.surface_at(v, s)];
        sh_thickness[s] = a.batch.thickness[a.batch.shape.surface_at(v, s)];
        sh_index[s] = a.batch.index[a.batch.shape.index_at(v, s, l)];
    }
    // Geisterplatten der Variante v mit den Brechzahlen der Wellenlänge l
    const int G = a.batch.shape.ghost_plates;
    const int K = a.batch.shape.ghost_terms;
    R* const sh_ghost_coeff = reinterpret_cast<R*>(sh_dynamic);
    R* const sh_ghost_index = sh_ghost_coeff + G * K;
    R* const sh_ghost_radius = sh_ghost_index + G;
    if (G > 0) {
        const R* const coeff = a.batch.ghost_coeff + a.batch.shape.ghost_coeff_at(v, 0, 0);
        for (int i = tid; i < G * K; i += kResidualThreads) {
            sh_ghost_coeff[i] = coeff[i];
        }
        for (int g = tid; g < G; g += kResidualThreads) {
            sh_ghost_index[g] = a.batch.ghost_index[a.batch.shape.ghost_index_at(v, g, l)];
            sh_ghost_radius[g] = a.batch.ghost_norm_radius[g];
        }
    }
    // Quadratur des Feldes: wie make_pupil_quadrature(N, M, mode)
    if (tid < a.pupil.rings) {
        sh_pupil.u[tid] = a.pupil.u[tid];
        sh_pupil.w[tid] = a.pupil.w[tid];
    }
    if (tid == 0) {
        sh_pupil.rings = a.pupil.rings;
        sh_pupil.spokes = plan.mode == PupilMode::Axis ? 1 : a.pupil.spokes;
        sh_pupil.mode = plan.mode;
    }
    __syncthreads();

    // Sicht mit nur einer Wellenlänge (table_l = 0); dieselben Zahlen wie surface_table(v).
    const SurfaceTableView<R> table{S,
                                    1,
                                    sh_type,
                                    sh_curvature,
                                    sh_thickness,
                                    sh_index,
                                    G,
                                    K,
                                    G > 0 ? sh_ghost_coeff : nullptr,
                                    G > 0 ? sh_ghost_index : nullptr,
                                    G > 0 ? sh_ghost_radius : nullptr};
    const Vec3<R> center = batch_image_point(a.batch, table, v, f);
    const int n = plan.samples;
    const std::int64_t first = plan.offset + static_cast<std::int64_t>(l) * n; // Tupel (f, l)
    R* const residuals = a.residuals + static_cast<std::int64_t>(v) * a.stride + first;
    std::uint8_t* const status = a.status + static_cast<std::int64_t>(v) * a.rays + first;
    R* const opd =
        a.opd != nullptr ? a.opd + static_cast<std::int64_t>(v) * a.rays + first : nullptr;

    // Durchlauf 1: Trace, OPL vorläufig in den Residuenplatz, Teilsummen (W_k, W_k · OPL_k)
    // der gültigen Strahlen in aufsteigender Reihenfolge der eigenen Stützstellen.
    R weight_ok = R(0);
    R weighted_opl = R(0);
    for (int k = tid; k < n; k += kResidualThreads) {
        const PupilSample<R> s = sample(sh_pupil, k);
        R opl = R(0);
        const TraceStatus st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl);
        if (st == TraceStatus::Ok) {
            weight_ok += s.weight;
            weighted_opl += s.weight * opl;
        }
        residuals[k] = opl;
        status[k] = static_cast<std::uint8_t>(st);
    }

    // Baumreduktion: im Warp per Shuffle, dann die Warps in fester Reihenfolge (Thread 0).
    weight_ok = warp_sum(weight_ok);
    weighted_opl = warp_sum(weighted_opl);
    if (tid % kWarpSize == 0) {
        sh_weight[tid / kWarpSize] = weight_ok;
        sh_weighted_opl[tid / kWarpSize] = weighted_opl;
    }
    __syncthreads();
    if (tid == 0) {
        R w = R(0);
        R wo = R(0);
        for (int i = 0; i < kResidualWarps; ++i) {
            w += sh_weight[i];
            wo += sh_weighted_opl[i];
        }
        // Ohne gültigen Strahl (W_ok = 0) ist der Kolben undefiniert: alle OPD und Residuen
        // des Tupels werden 0, der Status bleibt gesetzt (wie opd_statistics).
        const bool valid = w > R(0);
        sh_valid = valid ? 1 : 0;
        sh_piston = valid ? wo / w : R(0);
    }
    __syncthreads();
    const bool valid = sh_valid != 0;
    const R piston = sh_piston;
    const R scale = residual_scale(a.batch, f, l);

    // Durchlauf 2: OPD und Residuen derselben Stützstellen (gleicher Thread, eigene
    // Schreibzugriffe).
    for (int k = tid; k < n; k += kResidualThreads) {
        R e = R(0);
        if (valid && static_cast<TraceStatus>(status[k]) == TraceStatus::Ok) {
            e = residuals[k] - piston;
        }
        if (opd != nullptr) {
            opd[k] = e;
        }
        residuals[k] = ray_residual(scale, sample(sh_pupil, k), e);
    }
}

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

} // namespace

DeviceResiduals::~DeviceResiduals() {
    release();
}

DeviceResiduals::DeviceResiduals(DeviceResiduals&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_), ran_(other.ran_),
      opd_(other.opd_), variants_(other.variants_), rays_(other.rays_), stride_(other.stride_),
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
        variants_ = other.variants_;
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
    request.surface_max = false;
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
        const std::size_t ghost_bytes = ghost_shared_bytes(view.shape);
        if (ghost_bytes > kMaxGhostSharedBytes) {
            throw std::invalid_argument("DeviceResiduals::run: Geisterdaten (" +
                                        std::to_string(ghost_bytes) +
                                        " Byte) passen nicht in den Shared Memory");
        }
        residuals_kernel<double>
            <<<static_cast<unsigned>(blocks), kResidualThreads, ghost_bytes>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    layout_ = out;
    variants_ = V;
    rays_ = layout.rays_per_variant();
    stride_ = rays_ + options.extra_residuals;
    opd_ = options.opd;
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

ResidualResult run_residuals(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                             const ResidualRunOptions& options) {
    DeviceResiduals d;
    d.run(batch, layout, options);
    return d.download();
}

} // namespace mko::cuda
