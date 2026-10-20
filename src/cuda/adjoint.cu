// Modus ADJOINT, Grundgerüst (M5-T1): ein Block pro (Variante, Feld, Wellenlänge), Threads über
// die Pupillenstützstellen wie RESIDUALS (ADR-0004, ADR-0007). Pro Strahl Vorwärtsdurchlauf mit
// Aufzeichnung im Local Memory (AdjointRecord oder CheckpointRecord), danach die Aufzeichnung in
// absteigender Reihenfolge der Flächen lesen (Zugriffsmuster des Rückwärtsdurchlaufs in M5-T2;
// bei Checkpoints mit Neu-Vorwärtsrechnen der Abschnitte), Prüfsumme und optional Ablage im
// Gerätespeicher. Residuen, Status und OPD wie im Kernel RESIDUALS mit denselben Bausteinen
// (bitgleich). docs/architektur.md, Abschnitt "Modus ADJOINT: Vorwärtsdurchlauf mit Aufzeichnung
// (M5-T1)".

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/adjoint.hpp"
#include "mikroptim/math/df64.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/surface_table.hpp"
#include "mikroptim/trace/trace.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"
#include "wavefront_block.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mko::cuda {

namespace {

using detail::kBlockThreads;
using detail::region_ptr;

static_assert(kAdjointThreads == kBlockThreads, "ADJOINT nutzt die Blockgröße von RESIDUALS");

// Speicherform als Template-Parameter.
constexpr int kStorageNone = static_cast<int>(RecordStorage::None);
constexpr int kStorageFull = static_cast<int>(RecordStorage::Full);
constexpr int kStorageCheckpoint = static_cast<int>(RecordStorage::Checkpoint);

// Aufzeichnungstyp pro Speicherform und Kapazität.
template <class R, int kStorage, int kCapacity>
struct KernelRecord {
    using type = NoTraceRecorder;
};
template <class R, int kCapacity>
struct KernelRecord<R, kStorageFull, kCapacity> {
    using type = AdjointRecord<R, kCapacity>;
};
template <class R, int kCapacity>
struct KernelRecord<R, kStorageCheckpoint, kCapacity> {
    using type = CheckpointRecord<R, kCapacity, kCheckpointInterval>;
};

// Parameter des Kernels (per Wert). Wie ResidualKernelArgs, dazu OPL, Prüfsumme und optional die
// Ablage der Aufzeichnung (alle [V · rays] bzw. [V · rays][S]).
struct AdjointKernelArgs {
    SystemBatchView<double> batch;
    PupilQuadrature<double> pupil;
    const ResidualFieldPlan* plan;
    std::int64_t rays;
    std::int64_t stride;
    int soft;
    double fail_weight;
    double* residuals;
    double* opd;
    std::uint8_t* status;
    double* opl;
    std::uint64_t* checksum;
    std::int32_t* count;        // oder nullptr
    double* states;             // [ray][S][6] oder nullptr
    std::uint8_t* branch;       // [ray][S] oder nullptr
    double* image_s;            // oder nullptr
    std::uint8_t* image_branch; // oder nullptr
};

__device__ inline std::uint64_t bits_of(double x) {
    return static_cast<std::uint64_t>(__double_as_longlong(x));
}

// Liest die Aufzeichnung eines Strahls in absteigender Reihenfolge der Flächen
// (for_each_state_reverse), bildet die Prüfsumme und legt sie optional ab.
template <class R, class Table, class Record>
__device__ inline void consume_record(const Table& table, const Record& rec,
                                      const AdjointKernelArgs& a, std::int64_t ray, int S) {
    std::uint64_t h = kRecordChecksumSeed;
    double* const states = a.states != nullptr ? a.states + ray * S * 6 : nullptr;
    std::uint8_t* const branch = a.branch != nullptr ? a.branch + ray * S : nullptr;
    for_each_state_reverse(
        table, 0, rec, [&](int s, const Vec3<R>& p, const Vec3<R>& d, unsigned char b) {
            const double w[6] = {static_cast<double>(p.x), static_cast<double>(p.y),
                                 static_cast<double>(p.z), static_cast<double>(d.x),
                                 static_cast<double>(d.y), static_cast<double>(d.z)};
            for (int i = 0; i < 6; ++i) {
                h = record_checksum_step(h, bits_of(w[i]));
            }
            h = record_checksum_step(h, b);
            if (states != nullptr) {
                for (int i = 0; i < 6; ++i) {
                    states[s * 6 + i] = w[i];
                }
                branch[s] = b;
            }
        });
    const double image_s = static_cast<double>(rec.image_s);
    h = record_checksum_step(h, bits_of(image_s));
    h = record_checksum_step(h, rec.image_branch);
    h = record_checksum_step(h, static_cast<std::uint64_t>(rec.count));
    a.checksum[ray] = h;
    if (a.count != nullptr) {
        a.count[ray] = rec.count;
        a.image_s[ray] = image_s;
        a.image_branch[ray] = rec.image_branch;
    }
}

// Block b = (v · F + f) · L + l; Ablauf wie residuals_kernel (gleiche Bausteine, gleiche
// Reihenfolge), pro Strahl zusätzlich Aufzeichnung und deren Verbrauch.
template <class R, int kStorage, int kCapacity, bool kTransforms>
__global__ void __launch_bounds__(kBlockThreads) adjoint_forward_kernel(const AdjointKernelArgs a) {
    __shared__ detail::BlockTable<R> sh_table;
    __shared__ detail::PistonShared<R> sh_piston;
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
    const int n = plan.samples;
    const std::int64_t first = plan.offset + static_cast<std::int64_t>(l) * n;
    const std::int64_t ray0 = static_cast<std::int64_t>(v) * a.rays + first;
    double* const residuals = a.residuals + static_cast<std::int64_t>(v) * a.stride + first;
    std::uint8_t* const status = a.status + ray0;
    double* const opd = a.opd != nullptr ? a.opd + ray0 : nullptr;
    R* const opl_scratch = reinterpret_cast<R*>(residuals);

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
        TraceStatus st;
        if constexpr (kStorage == kStorageNone) {
            NoTraceRecorder* const no_record = nullptr;
            st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl, no_observer, soft_ptr,
                               no_record);
            a.checksum[ray0 + k] = 0;
        } else {
            typename KernelRecord<R, kStorage, kCapacity>::type record;
            st = batch_ray_opl(a.batch, table, 0, v, f, l, s, center, &opl, no_observer, soft_ptr,
                               &record);
            consume_record<R>(table, record, a, ray0 + k, S);
        }
        if (in_piston(st)) {
            weight_ok += s.weight;
            weighted_opl += s.weight * opl;
        }
        if (penalties != nullptr) {
            penalties[k] = static_cast<double>(ray_penalty(pscale, s, st, soft.penalty));
        }
        a.opl[ray0 + k] = static_cast<double>(opl);
        opl_scratch[k] = opl;
        status[k] = static_cast<std::uint8_t>(st);
    }

    R piston;
    bool valid;
    detail::block_piston(sh_piston, weight_ok, weighted_opl, tid, piston, valid);
    const R scale = R(residual_scale(a.batch, f, l));
    for (int k = tid; k < n; k += kBlockThreads) {
        R e = R(0);
        if (valid && in_piston(static_cast<TraceStatus>(status[k]))) {
            e = opl_scratch[k] - piston;
        }
        if (opd != nullptr) {
            opd[k] = static_cast<double>(e);
        }
        residuals[k] = static_cast<double>(ray_residual(scale, sample(sh_table.pupil, k), e));
    }
}

using KernelFn = void (*)(AdjointKernelArgs);

// Kernel-Variante zu Rechentyp, Speicherform und Kapazität.
template <class R, bool kTransforms>
KernelFn select_kernel(RecordStorage storage, int capacity) {
    switch (storage) {
    case RecordStorage::None:
        return adjoint_forward_kernel<R, kStorageNone, kRecordCapacitySmall, kTransforms>;
    case RecordStorage::Full:
        if (capacity == kRecordCapacitySmall) {
            return adjoint_forward_kernel<R, kStorageFull, kRecordCapacitySmall, kTransforms>;
        }
        if (capacity == kRecordCapacityMedium) {
            return adjoint_forward_kernel<R, kStorageFull, kRecordCapacityMedium, kTransforms>;
        }
        return adjoint_forward_kernel<R, kStorageFull, kRecordCapacityLarge, kTransforms>;
    case RecordStorage::Checkpoint:
        if (capacity == kRecordCapacitySmall) {
            return adjoint_forward_kernel<R, kStorageCheckpoint, kRecordCapacitySmall, kTransforms>;
        }
        if (capacity == kRecordCapacityMedium) {
            return adjoint_forward_kernel<R, kStorageCheckpoint, kRecordCapacityMedium,
                                          kTransforms>;
        }
        return adjoint_forward_kernel<R, kStorageCheckpoint, kRecordCapacityLarge, kTransforms>;
    }
    throw std::invalid_argument("ADJOINT: unbekannte Speicherform (RecordStorage)");
}

// transforms: Instanz für nicht koaxiale Batches (M8-T3, Transformationsdaten im Shared Memory).
KernelFn select_kernel(Precision precision, RecordStorage storage, int capacity,
                       bool transforms = false) {
    check_precision(precision);
    if (transforms) {
        return precision == Precision::Df64 ? select_kernel<df64, true>(storage, capacity)
                                            : select_kernel<double, true>(storage, capacity);
    }
    return precision == Precision::Df64 ? select_kernel<df64, false>(storage, capacity)
                                        : select_kernel<double, false>(storage, capacity);
}

void check_storage(RecordStorage storage) {
    switch (storage) {
    case RecordStorage::None:
    case RecordStorage::Full:
    case RecordStorage::Checkpoint:
        return;
    }
    throw std::invalid_argument("ADJOINT: unbekannte Speicherform (RecordStorage)");
}

void check_capacity_class(int capacity) {
    if (capacity != kRecordCapacitySmall && capacity != kRecordCapacityMedium &&
        capacity != kRecordCapacityLarge) {
        throw std::invalid_argument("ADJOINT: Kapazität " + std::to_string(capacity) +
                                    " ist keine Kapazitätsklasse (16, 32, 64)");
    }
}

// Kapazität für S Flächen aus den Optionen (0 = automatisch).
int resolve_capacity(const AdjointForwardOptions& o, int surfaces) {
    if (o.storage == RecordStorage::None) {
        return 0;
    }
    if (o.capacity == 0) {
        return record_capacity_class(surfaces);
    }
    check_capacity_class(o.capacity);
    if (o.capacity < surfaces) {
        throw std::invalid_argument("ADJOINT: Kapazität " + std::to_string(o.capacity) +
                                    " kleiner als die Flächenzahl " + std::to_string(surfaces));
    }
    return o.capacity;
}

// Hängt einen ausgerichteten Bereich von `bytes` an.
BufferRegion append_region(std::size_t& total, std::size_t bytes) {
    const BufferRegion r{total, bytes};
    total += align_bytes(bytes);
    return r;
}

} // namespace

const char* to_string(RecordStorage storage) {
    switch (storage) {
    case RecordStorage::None:
        return "none";
    case RecordStorage::Full:
        return "full";
    case RecordStorage::Checkpoint:
        return "checkpoint";
    }
    return "?";
}

AdjointKernelInfo adjoint_kernel_info(RecordStorage storage, int capacity, Precision precision,
                                      int device) {
    check_storage(storage);
    if (storage != RecordStorage::None) {
        check_capacity_class(capacity);
    }
    const KernelFn fn = select_kernel(precision, storage, capacity);
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

DeviceAdjointForward::~DeviceAdjointForward() {
    release();
}

DeviceAdjointForward::DeviceAdjointForward(DeviceAdjointForward&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), bytes_(other.bytes_), ran_(other.ran_),
      options_(other.options_), capacity_class_(other.capacity_class_), variants_(other.variants_),
      surfaces_(other.surfaces_), rays_(other.rays_), penalty_rows_(other.penalty_rows_),
      stride_(other.stride_), regions_(other.regions_) {
    other.buffer_ = nullptr;
    other.bytes_ = 0;
    other.ran_ = false;
}

DeviceAdjointForward& DeviceAdjointForward::operator=(DeviceAdjointForward&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        bytes_ = other.bytes_;
        ran_ = other.ran_;
        options_ = other.options_;
        capacity_class_ = other.capacity_class_;
        variants_ = other.variants_;
        surfaces_ = other.surfaces_;
        rays_ = other.rays_;
        penalty_rows_ = other.penalty_rows_;
        stride_ = other.stride_;
        regions_ = other.regions_;
        other.buffer_ = nullptr;
        other.bytes_ = 0;
        other.ran_ = false;
    }
    return *this;
}

void DeviceAdjointForward::release() noexcept {
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
    bytes_ = 0;
}

void DeviceAdjointForward::run(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                               const AdjointForwardOptions& options) {
    const SystemBatchView<double> view = batch.view(); // wirft bei leerem Batch
    check_residual_layout(batch.topology(), layout);
    check_precision(options.precision);
    check_storage(options.storage);
    const int S = view.shape.surfaces;
    const int capacity = resolve_capacity(options, S);
    const int V = batch.variants();
    const std::int64_t blocks =
        static_cast<std::int64_t>(V) * view.shape.fields * view.shape.wavelengths;
    if (blocks > kMaxGridBlocks) {
        throw std::invalid_argument("DeviceAdjointForward::run: " + std::to_string(blocks) +
                                    " Blöcke überschreiten das größte Grid; Batch aufteilen");
    }

    OutputRequest request;
    request.rays_per_variant = layout.rays_per_variant();
    request.penalty_residuals = layout.penalty_rows();
    request.residuals = true;
    request.ray_status = true;
    request.opd = options.opd;
    request.merit = false;
    request.surface_max = false;
    Regions rg;
    rg.out = make_output_layout(batch.topology(), V, request);
    const auto rays =
        static_cast<std::size_t>(V) * static_cast<std::size_t>(request.rays_per_variant);
    std::size_t total = rg.out.total_bytes;
    rg.plan = append_region(total, layout.plan().size() * sizeof(ResidualFieldPlan));
    rg.opl = append_region(total, rays * sizeof(double));
    rg.checksum = append_region(total, rays * sizeof(std::uint64_t));
    const bool states = options.states && options.storage != RecordStorage::None;
    if (states) {
        const auto s = static_cast<std::size_t>(S);
        rg.count = append_region(total, rays * sizeof(std::int32_t));
        rg.states = append_region(total, rays * s * 6 * sizeof(double));
        rg.branch = append_region(total, rays * s);
        rg.image_s = append_region(total, rays * sizeof(double));
        rg.image_branch = append_region(total, rays);
    }
    rg.total = total;

    ran_ = false;
    if (buffer_ != nullptr && device_ != batch.device()) {
        release();
    }
    device_ = batch.device();
    const detail::DeviceGuard guard(device_);
    if (total > bytes_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            bytes_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, total));
        bytes_ = total;
    }
    char* const base = static_cast<char*>(buffer_);
    MKO_CUDA_CHECK(cudaMemcpy(base + rg.plan.offset, layout.plan().data(), rg.plan.bytes,
                              cudaMemcpyHostToDevice));
    if (states) {
        // Einträge nach count bleiben 0 (Kernel schreibt nur s < count)
        MKO_CUDA_CHECK(cudaMemset(base + rg.states.offset, 0, rg.states.bytes));
        MKO_CUDA_CHECK(cudaMemset(base + rg.branch.offset, 0, rg.branch.bytes));
    }

    if (blocks > 0) {
        AdjointKernelArgs args{};
        args.batch = view;
        args.pupil = make_pupil_quadrature<double>(layout.settings().rings,
                                                   layout.settings().spokes, PupilMode::Half);
        args.plan = region_ptr<ResidualFieldPlan>(buffer_, rg.plan);
        args.rays = layout.rays_per_variant();
        args.stride = layout.trace_rows();
        args.soft = layout.soft_failure() ? 1 : 0;
        args.fail_weight = layout.settings().fail_weight;
        args.residuals = region_ptr<double>(buffer_, rg.out.residuals);
        args.opd = region_ptr<double>(buffer_, rg.out.opd);
        args.status = region_ptr<std::uint8_t>(buffer_, rg.out.ray_status);
        args.opl = region_ptr<double>(buffer_, rg.opl);
        args.checksum = region_ptr<std::uint64_t>(buffer_, rg.checksum);
        args.count = region_ptr<std::int32_t>(buffer_, rg.count);
        args.states = region_ptr<double>(buffer_, rg.states);
        args.branch = region_ptr<std::uint8_t>(buffer_, rg.branch);
        args.image_s = region_ptr<double>(buffer_, rg.image_s);
        args.image_branch = region_ptr<std::uint8_t>(buffer_, rg.image_branch);
        const std::size_t ghost_bytes = detail::ghost_shared_bytes(view.shape);
        if (ghost_bytes > detail::kMaxGhostSharedBytes) {
            throw std::invalid_argument("DeviceAdjointForward::run: Geisterdaten (" +
                                        std::to_string(ghost_bytes) +
                                        " Byte) passen nicht in den Shared Memory");
        }
        const KernelFn fn = select_kernel(options.precision, options.storage, capacity,
                                          view.decenter_tilt != nullptr);
        fn<<<static_cast<unsigned>(blocks), kBlockThreads,
             ghost_bytes + detail::transform_shared_bytes(view)>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    regions_ = rg;
    options_ = options;
    options_.states = states;
    capacity_class_ = capacity;
    variants_ = V;
    surfaces_ = S;
    rays_ = layout.rays_per_variant();
    penalty_rows_ = layout.penalty_rows();
    stride_ = layout.trace_rows();
    ran_ = true;
}

AdjointForwardResult DeviceAdjointForward::download() const {
    AdjointForwardResult r;
    download(r);
    return r;
}

void DeviceAdjointForward::download(AdjointForwardResult& r) const {
    if (!ran_) {
        throw std::logic_error("DeviceAdjointForward::download: kein Lauf vorhanden");
    }
    const Regions& rg = regions_;
    ResidualResult& res = r.residuals;
    res.variants = variants_;
    res.rays_per_variant = rays_;
    res.penalty_rows = penalty_rows_;
    res.stride = stride_;
    res.residuals.resize(rg.out.residuals.bytes / sizeof(double));
    res.status.resize(rg.out.ray_status.bytes);
    res.opd.resize(options_.opd ? rg.out.opd.bytes / sizeof(double) : 0);
    res.extrema = SurfaceExtremaResult{};
    r.storage = options_.storage;
    r.capacity = capacity_class_;
    r.surfaces = surfaces_;
    const std::size_t rays = static_cast<std::size_t>(variants_) * static_cast<std::size_t>(rays_);
    r.opl.resize(rays);
    r.checksum.resize(rays);
    const bool states = options_.states;
    const auto s = static_cast<std::size_t>(surfaces_);
    r.count.resize(states ? rays : 0);
    r.states.resize(states ? rays * s * 6 : 0);
    r.branch.resize(states ? rays * s : 0);
    r.image_s.resize(states ? rays : 0);
    r.image_branch.resize(states ? rays : 0);

    const detail::DeviceGuard guard(device_);
    const char* base = static_cast<const char*>(buffer_);
    const auto copy = [base](void* dst, const BufferRegion& region) {
        if (region.bytes > 0) {
            MKO_CUDA_CHECK(
                cudaMemcpy(dst, base + region.offset, region.bytes, cudaMemcpyDeviceToHost));
        }
    };
    copy(res.residuals.data(), rg.out.residuals);
    copy(res.status.data(), rg.out.ray_status);
    if (options_.opd) {
        copy(res.opd.data(), rg.out.opd);
    }
    copy(r.opl.data(), rg.opl);
    copy(r.checksum.data(), rg.checksum);
    if (states) {
        copy(r.count.data(), rg.count);
        copy(r.states.data(), rg.states);
        copy(r.branch.data(), rg.branch);
        copy(r.image_s.data(), rg.image_s);
        copy(r.image_branch.data(), rg.image_branch);
    }
}

AdjointForwardResult run_adjoint_forward(const DeviceSystemBatch& batch,
                                         const ResidualLayout& layout,
                                         const AdjointForwardOptions& options) {
    DeviceAdjointForward d;
    d.run(batch, layout, options);
    return d.download();
}

} // namespace mko::cuda
