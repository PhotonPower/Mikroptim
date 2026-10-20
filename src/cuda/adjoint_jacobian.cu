// Modus ADJOINT, adjungierte Jacobi-Matrix (M5-T2): ein Block pro (Lauf, Feld, Wellenlänge),
// Threads über die Pupillenstützstellen wie RESIDUALS (ADR-0004, ADR-0007). Pro Strahl
// Vorwärtsdurchlauf mit Aufzeichnung (AdjointRecord im Local Memory), Residuen wie das
// Grundgerüst (dieselben Bausteine), Rückwärtsdurchlauf mit ray_gradient_rows (derselbe Code wie
// adjoint_jacobian_cpu) direkt in die Spalten von J; danach pro Variable der Kolbenabzug als
// Projektion mit einer Baumreduktion fester Reihenfolge. Übersetzt mit --fmad=false wie
// adjoint.cu (bitgleiche Aufzeichnung und Neuberechnung). docs/architektur.md, Abschnitt
// "Modus ADJOINT: Jacobi-Matrix (M5-T2)".

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/adjoint_jacobian.hpp"
#include "mikroptim/math/df64.hpp"
#include "mikroptim/merit/ray_jacobian.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"
#include "wavefront_block.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko::cuda {

namespace {

using detail::kBlockThreads;
using detail::kBlockWarps;
using detail::kWarpSize;

static_assert(kAdjointJacobianThreads == kBlockThreads,
              "ADJOINT nutzt die Blockgröße von RESIDUALS");

// Parameter des Kernels (per Wert).
struct JacobianKernelArgs {
    SystemBatchView<double> batch;
    PupilQuadrature<double> pupil; // Quadratur der Nicht-Achsfelder (Half), wie RESIDUALS
    const ResidualFieldPlan* plan;
    AdjointParameterMap map; // Gerätezeiger
    std::int64_t rays;       // Strahlen pro Variante
    std::int64_t rows;       // R = Trace-Zeilen + Zusatzzeilen (Spaltenlänge von J)
    int variables;           // P
    int soft;
    double fail_weight;
    double* jacobian;     // [V][P][R]
    double* residuals;    // [V][R]
    std::uint8_t* status; // [V][rays]
};

// Kolben und W_ok des Blocks (wie detail::block_piston, zusätzlich W_ok für die Projektion).
template <class R>
struct PistonWeightShared {
    R weight[kBlockWarps];
    R weighted_opl[kBlockWarps];
    R piston;
    R weight_ok;
    int valid;
};

template <class R>
__device__ inline void block_piston_weight(PistonWeightShared<R>& sh, R weight_ok, R weighted_opl,
                                           int tid, R& piston, R& w_ok, bool& valid) {
    weight_ok = detail::warp_sum(weight_ok);
    weighted_opl = detail::warp_sum(weighted_opl);
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
        sh.weight_ok = w;
    }
    __syncthreads();
    valid = sh.valid != 0;
    piston = sh.piston;
    w_ok = sh.weight_ok;
}

// Block b = (v · F + f) · L + l.
template <class R, int kCapacity, bool kTransforms>
__global__ void __launch_bounds__(kBlockThreads)
    adjoint_jacobian_kernel(const JacobianKernelArgs a) {
    __shared__ detail::BlockTable<R> sh_table;
    __shared__ PistonWeightShared<R> sh_piston;
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
    const detail::BlockGhosts<R> ghosts =
        detail::load_block_ghosts<R>(sh_dynamic, a.batch, tu, tid);
    // Dezentrierung und Verkippung hinter den Geisterdaten (nur Instanz kTransforms, M8-T3)
    const detail::KernelTable<R, kTransforms> table = detail::block_kernel_table<R, kTransforms>(
        sh_table, S, ghosts,
        reinterpret_cast<R*>(sh_dynamic + detail::ghost_shared_bytes(a.batch.shape)), a.batch, v,
        tid);
    __syncthreads();

    const Vec3<R> center = batch_image_point(a.batch, table, v, f);
    const bool point_source = a.batch.source == SourceKind::Point;
    const int n = plan.samples;
    const std::int64_t first = plan.offset + static_cast<std::int64_t>(l) * n;
    const std::int64_t R_rows = a.rows;
    double* const residuals = a.residuals + static_cast<std::int64_t>(v) * R_rows + first;
    std::uint8_t* const status = a.status + static_cast<std::int64_t>(v) * a.rays + first;
    double* const col0 = a.jacobian + static_cast<std::int64_t>(v) * a.variables * R_rows;
    R* const opl_scratch = reinterpret_cast<R*>(residuals);

    // ---- Durchlauf 1: Trace mit Aufzeichnung, Strafresiduen, Rückwärtsdurchlauf ----
    double* const penalties = a.soft ? residuals + a.rays : nullptr;
    const R pscale = a.soft ? R(penalty_scale(a.batch, f, l, a.fail_weight)) : R(0);
    R weight_ok = R(0);
    R weighted_opl = R(0);
    for (int k = tid; k < n; k += kBlockThreads) {
        const PupilSample<R> s = sample(sh_table.pupil, k);
        R opl = R(0);
        SoftFailure<R> soft{R(0), TraceStatus::Ok};
        SoftFailure<R>* const soft_ptr = a.soft ? &soft : nullptr;
        NoSurfaceObserver* const no_observer = nullptr;
        AdjointRecord<R, kCapacity> record;
        const TraceStatus st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl,
                                             no_observer, soft_ptr, &record);
        if (in_piston(st)) {
            weight_ok += s.weight;
            weighted_opl += s.weight * opl;
        }
        if (penalties != nullptr) {
            penalties[k] = static_cast<double>(ray_penalty(pscale, s, st, soft.penalty));
        }
        opl_scratch[k] = opl;
        status[k] = static_cast<std::uint8_t>(st);
        const JacobianRowWriter<R> opl_row{a.map, l, col0, R_rows, first + k, R(1)};
        const JacobianRowWriter<R> penalty_row{
            a.map, l, col0, R_rows, a.rays + first + k, pscale * residual_weight(s)};
        ray_gradient_rows(table, 0, record, st, soft.first, a.soft != 0, center, point_source,
                          opl_row, penalty_row);
    }

    // ---- Kolben und Wellenfrontresiduen (wie RESIDUALS) ----
    R piston;
    R w_ok;
    bool valid;
    block_piston_weight(sh_piston, weight_ok, weighted_opl, tid, piston, w_ok, valid);
    const R scale = R(residual_scale(a.batch, f, l));
    for (int k = tid; k < n; k += kBlockThreads) {
        R e = R(0);
        if (valid && in_piston(static_cast<TraceStatus>(status[k]))) {
            e = opl_scratch[k] - piston;
        }
        residuals[k] = static_cast<double>(ray_residual(scale, sample(sh_table.pupil, k), e));
    }

    // ---- Kolbenabzug als Projektion: ein Warp pro Variable (p = Warp, Warp + 4, …) ----
    // ∂r_k/∂x = s √W_k (∂OPL_k/∂x − Σ_j W_j ∂OPL_j/∂x / W_ok) über die Strahlen im Kolben (FP64)
    // Gewichte W_k aus der Quadratur des Blocks (für R = double dieselben Zahlen wie auf der CPU).
    // Feste Reihenfolge: Lane j summiert k = j, j + 32, … aufsteigend, dann Baumreduktion über
    // die Lanes (warp_sum, Ergebnis in Lane 0, per Shuffle an alle verteilt). Die Spalten der
    // Warps sind disjunkt, J(k, p) und der Status sind seit block_piston_weight für alle sichtbar;
    // keine Synchronisation des Blocks pro Variable (M5-T3).
    const auto weight_of = [&](int k) {
        return static_cast<double>(sample(sh_table.pupil, k).weight);
    };
    const double scale_d = residual_scale(a.batch, f, l);
    const double w_ok_d = static_cast<double>(w_ok);
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    for (int p = warp; p < a.variables; p += kBlockWarps) {
        double* const col = col0 + static_cast<std::int64_t>(p) * R_rows + first;
        double part = 0.0;
        if (valid) {
            for (int k = lane; k < n; k += kWarpSize) {
                if (in_piston(static_cast<TraceStatus>(status[k]))) {
                    part += weight_of(k) * col[k];
                }
            }
        }
        part = detail::warp_sum(part);
        part = __shfl_sync(0xffffffffu, part, 0);
        const double mean = valid ? part / w_ok_d : 0.0;
        for (int k = lane; k < n; k += kWarpSize) {
            double value = 0.0;
            if (valid && in_piston(static_cast<TraceStatus>(status[k]))) {
                value = scale_d * msqrt(weight_of(k)) * (col[k] - mean);
            }
            col[k] = value;
        }
    }
}

using KernelFn = void (*)(JacobianKernelArgs);

template <class R, bool kTransforms>
KernelFn select_kernel(int capacity) {
    if (capacity == kRecordCapacitySmall) {
        return adjoint_jacobian_kernel<R, kRecordCapacitySmall, kTransforms>;
    }
    if (capacity == kRecordCapacityMedium) {
        return adjoint_jacobian_kernel<R, kRecordCapacityMedium, kTransforms>;
    }
    return adjoint_jacobian_kernel<R, kRecordCapacityLarge, kTransforms>;
}

void check_capacity(int capacity) {
    if (capacity != kRecordCapacitySmall && capacity != kRecordCapacityMedium &&
        capacity != kRecordCapacityLarge) {
        throw std::invalid_argument("ADJOINT: Kapazität " + std::to_string(capacity) +
                                    " ist keine Kapazitätsklasse (16, 32, 64)");
    }
}

// transforms: Instanz für nicht koaxiale Batches (M8-T3).
KernelFn select_kernel(Precision precision, int capacity, bool transforms = false) {
    check_precision(precision);
    check_capacity(capacity);
    if (transforms) {
        return precision == Precision::Df64 ? select_kernel<df64, true>(capacity)
                                            : select_kernel<double, true>(capacity);
    }
    return precision == Precision::Df64 ? select_kernel<df64, false>(capacity)
                                        : select_kernel<double, false>(capacity);
}

// Regionen des Puffers.
struct JacobianBufferLayout {
    BufferRegion jacobian, residuals, status, plan, curvature, thickness, glass, ghost, slope;
    std::size_t total = 0;
};

JacobianBufferLayout make_buffer_layout(const JacobianShape& shape, std::int64_t rays, int fields,
                                        const AdjointParameters& params) {
    JacobianBufferLayout l;
    std::size_t offset = 0;
    const auto place = [&offset](std::size_t bytes) {
        const BufferRegion r{offset, bytes};
        offset += align_bytes(bytes);
        return r;
    };
    l.jacobian = place(shape.jacobian_size() * sizeof(double));
    l.residuals = place(shape.residual_size() * sizeof(double));
    l.status = place(static_cast<std::size_t>(shape.runs) * static_cast<std::size_t>(rays));
    l.plan = place(static_cast<std::size_t>(fields) * sizeof(ResidualFieldPlan));
    l.curvature = place(params.curvature().size() * sizeof(int));
    l.thickness = place(params.thickness().size() * sizeof(int));
    l.glass = place(params.glass().size() * sizeof(int));
    l.ghost = place(params.ghost().size() * sizeof(int));
    l.slope = place(params.glass_slope().size() * sizeof(double));
    l.total = offset;
    return l;
}

double seconds_between(std::chrono::steady_clock::time_point a,
                       std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

void upload_region(void* buffer, const BufferRegion& r, const void* src) {
    if (r.bytes > 0) {
        MKO_CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer) + r.offset, src, r.bytes,
                                  cudaMemcpyHostToDevice));
    }
}

} // namespace

AdjointKernelInfo adjoint_jacobian_kernel_info(int capacity, Precision precision, int device) {
    const KernelFn fn = select_kernel(precision, capacity);
    const detail::DeviceGuard guard(device);
    cudaFuncAttributes attr{};
    MKO_CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
    AdjointKernelInfo info;
    info.registers = attr.numRegs;
    info.local_bytes = attr.localSizeBytes;
    info.shared_bytes = attr.sharedSizeBytes;
    info.max_threads_per_block = attr.maxThreadsPerBlock;
    return info;
}

std::size_t DeviceAdjointJacobian::buffer_bytes(const JacobianShape& shape,
                                                std::int64_t rays_per_variant) {
    constexpr std::size_t kAuxiliaryBytes = std::size_t{64} << 10;
    return align_bytes(shape.jacobian_size() * sizeof(double)) +
           align_bytes(shape.residual_size() * sizeof(double)) +
           align_bytes(static_cast<std::size_t>(shape.runs) *
                       static_cast<std::size_t>(rays_per_variant)) +
           kAuxiliaryBytes;
}

DeviceAdjointJacobian::~DeviceAdjointJacobian() {
    release();
}

DeviceAdjointJacobian::DeviceAdjointJacobian(DeviceAdjointJacobian&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), bytes_(other.bytes_), ready_(other.ready_),
      shape_(other.shape_), rays_(other.rays_), capacity_class_(other.capacity_class_),
      residual_offset_(other.residual_offset_), status_offset_(other.status_offset_),
      batch_(std::move(other.batch_)) {
    other.buffer_ = nullptr;
    other.bytes_ = 0;
    other.ready_ = false;
}

DeviceAdjointJacobian& DeviceAdjointJacobian::operator=(DeviceAdjointJacobian&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        bytes_ = other.bytes_;
        ready_ = other.ready_;
        shape_ = other.shape_;
        rays_ = other.rays_;
        capacity_class_ = other.capacity_class_;
        residual_offset_ = other.residual_offset_;
        status_offset_ = other.status_offset_;
        batch_ = std::move(other.batch_);
        other.buffer_ = nullptr;
        other.bytes_ = 0;
        other.ready_ = false;
    }
    return *this;
}

void DeviceAdjointJacobian::release() noexcept {
    ready_ = false;
    batch_.release();
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
    bytes_ = 0;
}

const double* DeviceAdjointJacobian::device_jacobian() const {
    return ready_ ? static_cast<const double*>(buffer_) : nullptr;
}

const double* DeviceAdjointJacobian::device_residuals() const {
    return ready_ ? reinterpret_cast<const double*>(static_cast<const char*>(buffer_) +
                                                    residual_offset_)
                  : nullptr;
}

const std::uint8_t* DeviceAdjointJacobian::device_status() const {
    return ready_ ? reinterpret_cast<const std::uint8_t*>(static_cast<const char*>(buffer_) +
                                                          status_offset_)
                  : nullptr;
}

void DeviceAdjointJacobian::compute(TraceEngine& engine, const VariableSet& set,
                                    const SystemBatchHost& base, std::span<const double> x0,
                                    const AdjointJacobianOptions& options, Precision precision) {
    ready_ = false;
    timing_ = AdjointJacobianTiming{};
    const auto t_start = std::chrono::steady_clock::now();
    check_precision(precision);
    // ---- Host: Läufe, Spaltenzuordnung, Prüfungen ----
    const ResidualLayout layout(base.topology(), options.quadrature);
    const SystemBatchHost batch = make_adjoint_batch(set, base, x0);
    batch.validate_values();
    const AdjointParameters params(set, batch);
    const int runs = batch.variants();
    const int P = set.size();
    const int E = options.extra_count();
    const JacobianShape shape{runs, P, layout.trace_rows(), E};
    const BatchShape& bs = batch.view().shape;
    const std::int64_t blocks = static_cast<std::int64_t>(runs) * bs.fields * bs.wavelengths;
    if (blocks > kMaxGridBlocks) {
        throw std::invalid_argument("DeviceAdjointJacobian: " + std::to_string(blocks) +
                                    " Blöcke überschreiten das größte Grid; Läufe aufteilen");
    }
    const std::size_t ghost_bytes = detail::ghost_shared_bytes(bs);
    if (ghost_bytes > detail::kMaxGhostSharedBytes) {
        throw std::invalid_argument("DeviceAdjointJacobian: Geisterdaten passen nicht in den "
                                    "Shared Memory");
    }
    if (options.extra.count < 0 || (options.extra.count > 0 && !options.extra.evaluate)) {
        throw std::invalid_argument("DeviceAdjointJacobian: Zusatzresiduen ungültig");
    }
    check_trace_extra_residuals(options.trace_extra, "DeviceAdjointJacobian");
    const auto t_prepared = std::chrono::steady_clock::now();

    // ---- Gerät: Puffer und Batch vor dem Lauf der Engine anlegen (sie sieht sie als belegt,
    // wie bei DeviceFdJacobian) ----
    const JacobianBufferLayout bl =
        make_buffer_layout(shape, layout.rays_per_variant(), layout.fields(), params);
    if (device_ != engine.device()) {
        release();
        batch_ = DeviceSystemBatch(engine.device());
    }
    device_ = engine.device();
    const detail::DeviceGuard guard(device_);
    batch_.upload(batch);
    if (bl.total > bytes_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            bytes_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, bl.total));
        bytes_ = bl.total;
    }
    const auto t_allocated = std::chrono::steady_clock::now();

    // ---- Zusatzzeilen auf dem Host (finite Differenzen, hybrider Ansatz) ----
    ExtraRowsJacobian extra;
    if (E > 0 && options.trace_extra.count == 0) {
        // ohne Trace: FD-Batch pro Lauf auf Host-Threads (M5-T3)
        extra = extra_rows_untraced(set, base, x0, options);
    } else if (E > 0) {
        const FdJacobianBatch fd = make_fd_jacobian_batch(set, base, x0, FdScheme::Central);
        SurfaceExtremaResult extrema;
        if (options.trace_extra.count > 0) {
            // Extrema jeder FD-Variante aus ihrem eigenen Trace (wie DeviceFdJacobian)
            TraceSettings ts;
            ts.quadrature = options.quadrature;
            ts.extrema = true;
            ts.precision = precision;
            extrema = engine.run(fd.batch, TraceMode::Merit, ts).merit.extrema;
        }
        const std::vector<double> values =
            fd_extra_residuals(fd_options_of(options), fd.batch, extrema);
        extra = extra_rows_from_fd(fd, values, E);
    }
    const auto t_extra = std::chrono::steady_clock::now();

    // ---- Gerät: Hilfsdaten, Kernel, Zusatzzeilen ----
    char* const base_ptr = static_cast<char*>(buffer_);
    // J und r₀ auf 0: Nicht gemeldete Ableitungen (Strafzeilen ohne Ereignis, Flächen hinter
    // einer hart versagenden) bleiben 0
    MKO_CUDA_CHECK(cudaMemset(base_ptr + bl.jacobian.offset, 0, bl.jacobian.bytes));
    MKO_CUDA_CHECK(cudaMemset(base_ptr + bl.residuals.offset, 0, bl.residuals.bytes));
    upload_region(buffer_, bl.plan, layout.plan().data());
    upload_region(buffer_, bl.curvature, params.curvature().data());
    upload_region(buffer_, bl.thickness, params.thickness().data());
    upload_region(buffer_, bl.glass, params.glass().data());
    upload_region(buffer_, bl.ghost, params.ghost().data());
    upload_region(buffer_, bl.slope, params.glass_slope().data());

    const int capacity = record_capacity_class(bs.surfaces);
    if (blocks > 0) {
        JacobianKernelArgs args{};
        args.batch = batch_.view();
        args.pupil = make_pupil_quadrature<double>(layout.settings().rings,
                                                   layout.settings().spokes, PupilMode::Half);
        args.plan = detail::region_ptr<ResidualFieldPlan>(buffer_, bl.plan);
        AdjointParameterMap map = params.view();
        map.curvature = detail::region_ptr<int>(buffer_, bl.curvature);
        map.thickness = detail::region_ptr<int>(buffer_, bl.thickness);
        map.glass = detail::region_ptr<int>(buffer_, bl.glass);
        map.ghost = detail::region_ptr<int>(buffer_, bl.ghost);
        map.glass_slope = detail::region_ptr<double>(buffer_, bl.slope);
        args.map = map;
        args.rays = layout.rays_per_variant();
        args.rows = shape.rows();
        args.variables = P;
        args.soft = layout.soft_failure() ? 1 : 0;
        args.fail_weight = layout.settings().fail_weight;
        args.jacobian = detail::region_ptr<double>(buffer_, bl.jacobian);
        args.residuals = detail::region_ptr<double>(buffer_, bl.residuals);
        args.status = detail::region_ptr<std::uint8_t>(buffer_, bl.status);
        const KernelFn fn = select_kernel(precision, capacity, args.batch.decenter_tilt != nullptr);
        fn<<<static_cast<unsigned>(blocks), kBlockThreads,
             ghost_bytes + detail::transform_shared_bytes(args.batch)>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUDA_CHECK(cudaDeviceSynchronize());
    }
    // Zusatzzeilen (Host, finite Differenzen) in J und r₀: je Spalte E Werte ab Zeile R_v
    if (E > 0) {
        const std::size_t width = static_cast<std::size_t>(E) * sizeof(double);
        const std::size_t pitch = static_cast<std::size_t>(shape.rows()) * sizeof(double);
        const std::size_t rv_bytes = static_cast<std::size_t>(shape.ray_rows) * sizeof(double);
        MKO_CUDA_CHECK(cudaMemcpy2D(base_ptr + bl.jacobian.offset + rv_bytes, pitch,
                                    extra.jacobian.data(), width, width, shape.column_count(),
                                    cudaMemcpyHostToDevice));
        MKO_CUDA_CHECK(cudaMemcpy2D(base_ptr + bl.residuals.offset + rv_bytes, pitch,
                                    extra.residuals.data(), width, width,
                                    static_cast<std::size_t>(runs), cudaMemcpyHostToDevice));
    }
    const auto t_end = std::chrono::steady_clock::now();
    timing_.prepare_seconds = seconds_between(t_start, t_prepared);
    timing_.extra_seconds = seconds_between(t_allocated, t_extra);
    timing_.device_seconds =
        seconds_between(t_prepared, t_allocated) + seconds_between(t_extra, t_end);
    timing_.total_seconds = seconds_between(t_start, t_end);
    shape_ = shape;
    rays_ = layout.rays_per_variant();
    capacity_class_ = capacity;
    residual_offset_ = bl.residuals.offset;
    status_offset_ = bl.status.offset;
    ready_ = true;
}

AdjointJacobian DeviceAdjointJacobian::download() const {
    AdjointJacobian out;
    download(out);
    return out;
}

void DeviceAdjointJacobian::download(AdjointJacobian& out) const {
    if (!ready_) {
        throw std::logic_error("DeviceAdjointJacobian::download: keine Jacobi-Matrix vorhanden");
    }
    out.shape = shape_;
    out.rays_per_variant = rays_;
    out.jacobian.resize(shape_.jacobian_size());
    out.residuals.resize(shape_.residual_size());
    out.status.resize(static_cast<std::size_t>(shape_.runs) * static_cast<std::size_t>(rays_));
    const detail::DeviceGuard guard(device_);
    const char* base_ptr = static_cast<const char*>(buffer_);
    MKO_CUDA_CHECK(cudaMemcpy(out.jacobian.data(), base_ptr, out.jacobian.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    MKO_CUDA_CHECK(cudaMemcpy(out.residuals.data(), base_ptr + residual_offset_,
                              out.residuals.size() * sizeof(double), cudaMemcpyDeviceToHost));
    MKO_CUDA_CHECK(cudaMemcpy(out.status.data(), base_ptr + status_offset_, out.status.size(),
                              cudaMemcpyDeviceToHost));
}

AdjointJacobian adjoint_jacobian_gpu(TraceEngine& engine, const VariableSet& set,
                                     const SystemBatchHost& base, std::span<const double> x0,
                                     const AdjointJacobianOptions& options, Precision precision) {
    DeviceAdjointJacobian d;
    d.compute(engine, set, base, x0, options, precision);
    return d.download();
}

} // namespace mko::cuda
