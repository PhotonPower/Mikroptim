// Finite-Differenzen-Jacobi auf der GPU (M3-T2): FD-Varianten über die TraceEngine
// (run_on_device), Differenzen elementweise im Kernel fd_difference_kernel aus dem
// Ausgabepuffer der Engine; J, r₀ und Status-Wechsel bleiben im Gerätespeicher.
// docs/optimierung.md, Abschnitt „Finite-Differenzen-Jacobi“.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/optim/fd_formula.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko::cuda {

namespace {

// Parameter des Differenzkernels für eine Tranche (per Wert).
struct FdKernelArgs {
    const double* residuals;    // Tranche: [count · stride], Variante first_variant + j ab j·stride
    const std::uint8_t* status; // Tranche: [count · rays]
    std::int64_t stride;        // Residuen pro Variante im Puffer der Engine
    std::int64_t rays;          // Strahlen pro Variante (Status)
    std::int64_t rows;          // Zeilen des Traces pro Variante (R_v: Strahl- und Strafresiduen)
    const double* extra;        // [V · E], ganzer Batch (nullptr bei E = 0)
    int extra_count;            // E
    int variables;              // P
    int variants_per_run;       // n
    int first_run;              // erster Lauf der Tranche
    int first_variant;          // erste Variante der Tranche (= first_run · n)
    int central;                // 1: zentrales Schema (2P + 1 Varianten), 0: vorwärts
    const std::uint8_t* stencil; // [runs · P] FdStencil
    const double* step_a;        // [runs · P] d_a
    const double* step_b;        // [runs · P] d_b (vorwärts 0)
    double* jacobian;            // [runs][P][R]
    double* r0;                  // [runs][R]
    std::int64_t* changes;       // [runs][P]
};

// Zeile r der Variante v (Index im ganzen Batch): Strahlresiduum aus der Tranche oder
// Zusatzresiduum.
__device__ inline double row_value(const FdKernelArgs& a, std::int64_t v, std::int64_t r) {
    if (r < a.rows) {
        return a.residuals[(v - a.first_variant) * a.stride + r];
    }
    return a.extra[v * a.extra_count + (r - a.rows)];
}

__device__ inline bool ray_ok(const FdKernelArgs& a, std::int64_t v, std::int64_t r) {
    return a.status[(v - a.first_variant) * a.rays + r] ==
           static_cast<std::uint8_t>(TraceStatus::Ok);
}

// Block b = j · (P + 1) + p für den Lauf first_run + j: p < P bildet Spalte p (J elementweise
// über die Zeilen, dazu die Status-Wechsel als ganzzahlige Baumreduktion), p = P kopiert r₀.
// Keine Gleitkomma-Reduktion; jedes Element hängt nur von seinen drei Residuen ab.
__global__ void __launch_bounds__(kFdJacobianThreads) fd_difference_kernel(const FdKernelArgs a) {
    __shared__ std::int64_t sh_count[kFdJacobianThreads];
    const int P = a.variables;
    const std::int64_t run = a.first_run + static_cast<std::int64_t>(blockIdx.x) / (P + 1);
    const int p = static_cast<int>(blockIdx.x % static_cast<unsigned>(P + 1));
    const std::int64_t R = a.rows + a.extra_count;
    const std::int64_t v0 = run * a.variants_per_run;
    if (p == P) { // ganzer Block: r₀ kopieren (kein __syncthreads danach)
        for (std::int64_t r = threadIdx.x; r < R; r += blockDim.x) {
            a.r0[run * R + r] = row_value(a, v0, r);
        }
        return;
    }
    const std::int64_t col = run * P + p;
    const auto stencil = static_cast<FdStencil>(a.stencil[col]);
    const double da = a.step_a[col];
    const double db = a.step_b[col];
    const std::int64_t va = v0 + (a.central ? 2 * p + 1 : p + 1);
    const std::int64_t vb = a.central ? v0 + 2 * p + 2 : va;
    double* out = a.jacobian + col * R;
    std::int64_t count = 0;
    for (std::int64_t r = threadIdx.x; r < R; r += blockDim.x) {
        out[r] = fd_derivative(stencil, row_value(a, v0, r), row_value(a, va, r),
                               row_value(a, vb, r), da, db);
        if (r < a.rays) {
            const bool ok0 = ray_ok(a, v0, r);
            if (ray_ok(a, va, r) != ok0 || ray_ok(a, vb, r) != ok0) {
                ++count;
            }
        }
    }
    sh_count[threadIdx.x] = count;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_count[threadIdx.x] += sh_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        a.changes[col] = sh_count[0];
    }
}

// Regionen des Puffers (jeweils auf kBatchAlignment ausgerichtet).
struct FdBufferLayout {
    BufferRegion jacobian, residuals, changes, extra, stencil, step_a, step_b;
    std::size_t total = 0;
};

FdBufferLayout make_fd_buffer_layout(const JacobianShape& shape, int variants) {
    FdBufferLayout l;
    std::size_t offset = 0;
    const auto place = [&offset](std::size_t bytes) {
        const BufferRegion r{offset, bytes};
        offset += align_bytes(bytes);
        return r;
    };
    const std::size_t cols = shape.column_count();
    l.jacobian = place(shape.jacobian_size() * sizeof(double));
    l.residuals = place(shape.residual_size() * sizeof(double));
    l.changes = place(cols * sizeof(std::int64_t));
    l.extra = place(static_cast<std::size_t>(variants) *
                    static_cast<std::size_t>(shape.extra_rows) * sizeof(double));
    l.stencil = place(cols * sizeof(std::uint8_t));
    l.step_a = place(cols * sizeof(double));
    l.step_b = place(cols * sizeof(double));
    l.total = offset;
    return l;
}

template <class T>
T* at(void* buffer, const BufferRegion& r) {
    return r.bytes == 0 ? nullptr : reinterpret_cast<T*>(static_cast<char*>(buffer) + r.offset);
}

void upload(void* buffer, const BufferRegion& r, const void* src) {
    if (r.bytes > 0) {
        MKO_CUDA_CHECK(cudaMemcpy(static_cast<char*>(buffer) + r.offset, src, r.bytes,
                                  cudaMemcpyHostToDevice));
    }
}

} // namespace

std::size_t DeviceFdJacobian::buffer_bytes(const JacobianShape& shape, int variants) {
    return make_fd_buffer_layout(shape, variants).total;
}

DeviceFdJacobian::~DeviceFdJacobian() {
    release();
}

DeviceFdJacobian::DeviceFdJacobian(DeviceFdJacobian&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_),
      ready_(other.ready_), shape_(other.shape_), scheme_(other.scheme_),
      patterns_(std::move(other.patterns_)), info_(other.info_),
      residual_offset_(other.residual_offset_), changes_offset_(other.changes_offset_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.ready_ = false;
}

DeviceFdJacobian& DeviceFdJacobian::operator=(DeviceFdJacobian&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        ready_ = other.ready_;
        shape_ = other.shape_;
        scheme_ = other.scheme_;
        patterns_ = std::move(other.patterns_);
        info_ = other.info_;
        residual_offset_ = other.residual_offset_;
        changes_offset_ = other.changes_offset_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.ready_ = false;
    }
    return *this;
}

void DeviceFdJacobian::release() noexcept {
    ready_ = false;
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

void DeviceFdJacobian::compute(TraceEngine& engine, const VariableSet& set,
                               const SystemBatchHost& base, std::span<const double> x0,
                               const FdJacobianOptions& options) {
    ready_ = false;
    // ---- Host: Batch, Ablage, Zusatzresiduen, Schritte ----
    const ResidualLayout layout(base.topology(), options.quadrature);
    FdJacobianBatch fd = make_fd_jacobian_batch(set, base, x0, options.scheme);
    if (options.extra.count < 0) {
        throw std::invalid_argument("DeviceFdJacobian: negative Zahl von Zusatzresiduen");
    }
    check_trace_extra_residuals(options.trace_extra, "DeviceFdJacobian");
    // Zusatzresiduen [V · E] mit E = E_extra + E_trace; die trace-abhängigen (Spalten ab
    // E_extra) entstehen erst pro Tranche aus deren Extrema und sind bis dahin 0.
    const int V = fd.batch.variants();
    const int e_extra = options.extra.count;
    const int e_trace = options.trace_extra.count;
    const int E = e_extra + e_trace;
    std::vector<double> extra_only(static_cast<std::size_t>(V) * static_cast<std::size_t>(e_extra));
    evaluate_extra_residuals(options.extra, fd.batch, extra_only, 0, -1, options.extra_threads);
    std::vector<double> extra =
        e_trace == 0 ? std::move(extra_only)
                     : join_extra_rows(extra_only, e_extra,
                                       std::vector<double>(static_cast<std::size_t>(V) *
                                                               static_cast<std::size_t>(e_trace),
                                                           0.0),
                                       e_trace, V);
    const int P = set.size();
    const int n = fd.variants_per_run;
    const JacobianShape shape{fd.runs, P, layout.trace_rows(), E};
    if (static_cast<std::int64_t>(fd.runs) * (P + 1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("DeviceFdJacobian: zu viele Läufe und Variablen für ein Grid");
    }
    const std::size_t cols = shape.column_count();
    std::vector<std::uint8_t> stencil(cols);
    std::vector<double> step_a(cols);
    std::vector<double> step_b(cols, 0.0);
    const bool central = options.scheme == FdScheme::Central;
    for (int i = 0; i < fd.runs; ++i) {
        const FdPattern& pat = fd.patterns[static_cast<std::size_t>(i)];
        for (int k = 0; k < P; ++k) {
            const std::size_t c = static_cast<std::size_t>(i) * static_cast<std::size_t>(P) +
                                  static_cast<std::size_t>(k);
            const auto kk = static_cast<std::size_t>(k);
            stencil[c] = static_cast<std::uint8_t>(pat.stencil[kk]);
            step_a[c] = pat.step_plus[kk];
            if (central) {
                step_b[c] = pat.step_minus[kk];
            }
        }
    }

    // ---- Gerät: Puffer vor dem Lauf der Engine anlegen (sie sieht ihn als belegt) ----
    const FdBufferLayout l = make_fd_buffer_layout(shape, fd.batch.variants());
    if (buffer_ != nullptr && device_ != engine.device()) {
        release();
    }
    device_ = engine.device();
    const detail::DeviceGuard guard(device_);
    if (l.total > capacity_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            capacity_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, l.total));
        capacity_ = l.total;
    }
    upload(buffer_, l.extra, extra.data());
    upload(buffer_, l.stencil, stencil.data());
    upload(buffer_, l.step_a, step_a.data());
    upload(buffer_, l.step_b, step_b.data());

    // ---- Engine: Tranchen aus ganzen Läufen, Differenzen pro Tranche ----
    FdKernelArgs args{};
    args.extra = at<const double>(buffer_, l.extra);
    args.extra_count = shape.extra_rows;
    args.variables = P;
    args.variants_per_run = n;
    args.central = central ? 1 : 0;
    args.stencil = at<const std::uint8_t>(buffer_, l.stencil);
    args.step_a = at<const double>(buffer_, l.step_a);
    args.step_b = at<const double>(buffer_, l.step_b);
    args.jacobian = at<double>(buffer_, l.jacobian);
    args.r0 = at<double>(buffer_, l.residuals);
    args.changes = at<std::int64_t>(buffer_, l.changes);
    TraceSettings settings;
    settings.quadrature = options.quadrature;
    settings.extra_residuals = 0;   // Zusatzresiduen liegen im eigenen Puffer
    settings.extrema = e_trace > 0; // Max-Reduktionen für trace-abhängige Residuen
    const int S = fd.batch.surfaces();
    SurfaceExtremaResult tranche_extrema; // Extrema der aktuellen Tranche (Host)
    std::vector<double> tranche_traced;
    const auto consume = [&](const DeviceTranche& t) {
        if (e_trace > 0) {
            // Extrema der Tranche herunterladen, trace-abhängige Residuen ihrer Varianten auf
            // dem Host auswerten und deren Zeilen im Gerätepuffer der Zusatzresiduen ersetzen.
            // Alle Varianten eines Laufs liegen in derselben Tranche (Granularität n), der
            // Differenzkernel sieht also nur fertige Zeilen.
            tranche_extrema.variants = t.count;
            tranche_extrema.surfaces = S;
            tranche_extrema.values.resize(static_cast<std::size_t>(t.count) *
                                          static_cast<std::size_t>(S) * 2);
            MKO_CUDA_CHECK(cudaMemcpy(tranche_extrema.values.data(), t.extrema,
                                      tranche_extrema.values.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            tranche_traced.resize(static_cast<std::size_t>(t.count) *
                                  static_cast<std::size_t>(e_trace));
            evaluate_trace_extra_residuals(options.trace_extra, fd.batch, tranche_extrema, t.first,
                                           tranche_traced, t.first, t.count, options.extra_threads);
            for (int j = 0; j < t.count; ++j) {
                const std::size_t row =
                    static_cast<std::size_t>(t.first + j) * static_cast<std::size_t>(E);
                for (int i = 0; i < e_trace; ++i) {
                    extra[row + static_cast<std::size_t>(e_extra + i)] =
                        tranche_traced[static_cast<std::size_t>(j) *
                                           static_cast<std::size_t>(e_trace) +
                                       static_cast<std::size_t>(i)];
                }
            }
            const std::size_t first_row =
                static_cast<std::size_t>(t.first) * static_cast<std::size_t>(E);
            MKO_CUDA_CHECK(cudaMemcpy(
                at<double>(buffer_, l.extra) + first_row, extra.data() + first_row,
                static_cast<std::size_t>(t.count) * static_cast<std::size_t>(E) * sizeof(double),
                cudaMemcpyHostToDevice));
        }
        FdKernelArgs a = args;
        a.residuals = t.residuals;
        a.status = t.status;
        a.stride = t.stride;
        a.rays = t.rays_per_variant;
        a.rows = t.rays_per_variant + t.penalty_rows;
        a.first_run = t.first / n;
        a.first_variant = t.first;
        const int runs = t.count / n;
        const auto blocks = static_cast<unsigned>(runs * (P + 1));
        fd_difference_kernel<<<blocks, kFdJacobianThreads>>>(a);
        MKO_CUDA_CHECK(cudaGetLastError());
    };
    info_ = engine.run_on_device(fd.batch, settings, n, consume);
    MKO_CUDA_CHECK(cudaDeviceSynchronize());

    shape_ = shape;
    scheme_ = options.scheme;
    patterns_ = std::move(fd.patterns);
    residual_offset_ = l.residuals.offset;
    changes_offset_ = l.changes.offset;
    ready_ = true;
}

const double* DeviceFdJacobian::device_jacobian() const {
    return ready_ ? static_cast<const double*>(buffer_) : nullptr;
}

const double* DeviceFdJacobian::device_residuals() const {
    return ready_ ? reinterpret_cast<const double*>(static_cast<const char*>(buffer_) +
                                                    residual_offset_)
                  : nullptr;
}

const std::int64_t* DeviceFdJacobian::device_status_changes() const {
    return ready_ ? reinterpret_cast<const std::int64_t*>(static_cast<const char*>(buffer_) +
                                                          changes_offset_)
                  : nullptr;
}

FdJacobian DeviceFdJacobian::download() const {
    FdJacobian out;
    download(out);
    return out;
}

void DeviceFdJacobian::download(FdJacobian& out) const {
    if (!ready_) {
        throw std::logic_error("DeviceFdJacobian::download: keine Jacobi-Matrix berechnet");
    }
    out.shape = shape_;
    out.scheme = scheme_;
    out.patterns = patterns_;
    out.jacobian.resize(shape_.jacobian_size());
    out.residuals.resize(shape_.residual_size());
    out.status_changes.resize(shape_.column_count());
    const detail::DeviceGuard guard(device_);
    MKO_CUDA_CHECK(cudaMemcpy(out.jacobian.data(), device_jacobian(),
                              out.jacobian.size() * sizeof(double), cudaMemcpyDeviceToHost));
    MKO_CUDA_CHECK(cudaMemcpy(out.residuals.data(), device_residuals(),
                              out.residuals.size() * sizeof(double), cudaMemcpyDeviceToHost));
    MKO_CUDA_CHECK(cudaMemcpy(out.status_changes.data(), device_status_changes(),
                              out.status_changes.size() * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
}

FdJacobian fd_jacobian_gpu(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const FdJacobianOptions& options) {
    DeviceFdJacobian device;
    device.compute(engine, set, base, x0, options);
    return device.download();
}

} // namespace mko::cuda
