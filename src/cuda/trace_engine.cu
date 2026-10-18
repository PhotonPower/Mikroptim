// Host-API TraceEngine (M2-T4): Gerätewahl, Prüfung, Tranchen nach freiem Gerätespeicher,
// Rückgabe in Host-Vektoren (docs/architektur.md, Abschnitt "Host-API TraceEngine").
// Modi RESIDUALS (M2-T2) und MERIT (M2-T3).

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/trace_engine.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko::cuda {

namespace {

void check_mode(TraceMode mode) {
    switch (mode) {
    case TraceMode::Residuals:
    case TraceMode::Merit:
        return;
    }
    throw std::invalid_argument("TraceEngine: unbekannter Modus");
}

void check_settings(const TraceSettings& settings) {
    if (settings.extra_residuals < 0) {
        throw std::invalid_argument("TraceEngine: negative Zahl weiterer Residuen");
    }
}

// Ausgaben auf dem Gerät, wie DeviceResiduals::run bzw. DeviceMerit::run sie anlegen. Im
// Modus RESIDUALS wird der Status immer geschrieben, weil Durchlauf 2 des Kernels ihn liest;
// im Modus MERIT liegen Residuen, Status und OPL nur im Shared Memory.
OutputRequest output_request(TraceMode mode, const ResidualLayout& layout,
                             const TraceSettings& settings) {
    OutputRequest r;
    r.rays_per_variant = layout.rays_per_variant();
    r.surface_max = settings.extrema;
    if (mode == TraceMode::Merit) {
        r.residuals = false;
        r.ray_status = false;
        r.opd = false;
        r.merit = true;
    } else {
        r.extra_residuals = settings.extra_residuals;
        r.residuals = true;
        r.ray_status = true;
        r.opd = settings.opd;
        r.merit = false;
    }
    return r;
}

// Puffer pro Lauf, unabhängig von der Variantenzahl: Feldablage ResidualFieldPlan[F].
std::size_t fixed_run_bytes(const BatchTopology& topology) {
    return align_bytes(static_cast<std::size_t>(topology.fields) * sizeof(ResidualFieldPlan));
}

// Speicher pro Variante außerhalb von OutputRequest: im Modus MERIT der Anteil weiterer
// Residuen extra_merit[v] (ein FP64 pro Variante, wie DeviceMerit ihn ablegt).
std::size_t extra_bytes_per_variant(TraceMode mode, const TraceSettings& settings) {
    return mode == TraceMode::Merit && !settings.extra_merit.empty() ? sizeof(double) : 0;
}

// Größte Tranche für `usable` Byte nutzbaren Speicher (nach Abzug der Reserve).
int tranche_capacity(std::size_t usable, const BatchTopology& topology,
                     const OutputRequest& request, std::size_t extra_per_variant) {
    const std::size_t fixed = fixed_run_bytes(topology);
    if (usable <= fixed) {
        return 0;
    }
    return mko::max_variants(usable - fixed, topology, request, MemoryReserve{0, 0.0},
                             extra_per_variant);
}

std::string mib(std::size_t bytes) {
    return std::to_string(static_cast<double>(bytes) / (1024.0 * 1024.0)) + " MiB";
}

// Leert ein Ergebnis; die Kapazität der Vektoren bleibt erhalten.
void clear(SurfaceExtremaResult& e) {
    e.variants = 0;
    e.surfaces = 0;
    e.values.clear();
}
void clear(ResidualResult& r) {
    r.variants = 0;
    r.rays_per_variant = 0;
    r.stride = 0;
    r.residuals.clear();
    r.opd.clear();
    r.status.clear();
    clear(r.extrema);
}
void clear(MeritResult& m) {
    m.variants = 0;
    m.rays_per_variant = 0;
    m.merit.clear();
    m.failed.clear();
    clear(m.extrema);
}

// Kopiert `bytes` Byte vom Gerät nach `dst`.
void copy_to_host(void* dst, const void* src, std::size_t bytes) {
    if (bytes > 0) {
        MKO_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }
}

} // namespace

const char* to_string(TraceMode mode) {
    switch (mode) {
    case TraceMode::Residuals:
        return "RESIDUALS";
    case TraceMode::Merit:
        return "MERIT";
    }
    return "?";
}

TraceEngine::TraceEngine(int device, const EngineOptions& options) : options_(options) {
    const int count = device_count();
    if (count == 0) {
        throw CudaError("TraceEngine: kein CUDA-Gerät gefunden (Treiber installiert?)");
    }
    if (device < 0 || device >= count) {
        throw std::invalid_argument("TraceEngine: Gerät " + std::to_string(device) +
                                    " existiert nicht (vorhanden: 0 … " +
                                    std::to_string(count - 1) + ")");
    }
    // Prüft die Reserve (std::invalid_argument bei fraction außerhalb von [0, 1]).
    (void)usable_bytes(0, options_.reserve);
    cudaDeviceProp prop{};
    MKO_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 6) {
        throw std::invalid_argument("TraceEngine: Gerät " + std::to_string(device) + " (" +
                                    prop.name + ", sm_" + std::to_string(prop.major) +
                                    std::to_string(prop.minor) +
                                    ") wird nicht unterstützt (mindestens sm_60)");
    }
    device_ = device;
    device_name_ = prop.name;
    batch_ = DeviceSystemBatch(device);
}

TraceEngine::~TraceEngine() = default;
TraceEngine::TraceEngine(TraceEngine&&) noexcept = default;
TraceEngine& TraceEngine::operator=(TraceEngine&&) noexcept = default;

void TraceEngine::release() noexcept {
    batch_.release();
    residuals_.release();
    merit_.release();
}

std::size_t TraceEngine::device_bytes() const {
    return batch_.capacity_bytes() + residuals_.capacity_bytes() + merit_.capacity_bytes();
}

std::size_t TraceEngine::available_bytes() const {
    // Eigene Puffer zählen als verfügbar: Ein Lauf verwendet sie wieder oder gibt sie frei.
    std::size_t available = device_memory_info(device_).free_bytes + device_bytes();
    if (options_.memory_limit_bytes) {
        available = std::min(available, *options_.memory_limit_bytes);
    }
    return available;
}

int TraceEngine::max_variants(const BatchTopology& topology, TraceMode mode,
                              const TraceSettings& settings) const {
    check_mode(mode);
    check_settings(settings);
    const ResidualLayout layout(topology, settings.quadrature);
    const std::size_t usable = usable_bytes(available_bytes(), options_.reserve);
    return tranche_capacity(usable, topology, output_request(mode, layout, settings),
                            extra_bytes_per_variant(mode, settings));
}

TraceEngine::TranchePlan TraceEngine::plan_tranches(const BatchTopology& topology, TraceMode mode,
                                                    const ResidualLayout& layout,
                                                    const TraceSettings& settings, int variants,
                                                    int granularity) {
    const bool merit_mode = mode == TraceMode::Merit;
    const OutputRequest request = output_request(mode, layout, settings);
    const std::size_t extra_per_variant = extra_bytes_per_variant(mode, settings);
    const auto g = static_cast<std::size_t>(granularity);
    // Größte Tranche als Vielfaches von granularity
    const auto rounded = [g](int capacity) {
        return static_cast<int>(static_cast<std::size_t>(capacity) / g * g);
    };
    // Der Ausgabepuffer des anderen Modus bleibt erhalten (Wechsel RESIDUALS/MERIT im
    // Optimierer) und zählt dann nicht als verfügbar, solange die Tranche dadurch nicht
    // kleiner wird; sonst wird er freigegeben.
    const std::size_t available = available_bytes();
    const std::size_t other = merit_mode ? residuals_.capacity_bytes() : merit_.capacity_bytes();
    std::size_t usable = usable_bytes(available - std::min(other, available), options_.reserve);
    int capacity = rounded(tranche_capacity(usable, topology, request, extra_per_variant));
    if (other > 0 && capacity < variants) {
        const std::size_t usable_all = usable_bytes(available, options_.reserve);
        const int capacity_all =
            rounded(tranche_capacity(usable_all, topology, request, extra_per_variant));
        if (capacity_all > capacity) {
            if (merit_mode) {
                residuals_.release();
            } else {
                merit_.release();
            }
            usable = usable_all;
            capacity = capacity_all;
        }
    }
    if (capacity < 1) {
        const std::string what =
            granularity == 1 ? std::string("eine Variante passt")
                             : std::to_string(granularity) + " zusammengehörige Varianten passen";
        throw CudaError(
            "TraceEngine: nicht einmal " + what + " in den Gerätespeicher (Gerät " +
            std::to_string(device_) + ", verfügbar " + mib(available) + ", nach Reserve " +
            mib(usable) + ", benötigt " +
            mib(estimate_device_bytes(topology, granularity, request, extra_per_variant) +
                fixed_run_bytes(topology)) +
            ")");
    }
    const int tranche = std::min(capacity, variants);
    // Reichen die vorhandenen Puffer nicht, vorher Batch und Ausgabepuffer des Modus
    // freigeben: Sonst lägen alter und neuer Puffer kurz gleichzeitig im Speicher, und das
    // Budget würde überschritten.
    const std::size_t need_batch = make_batch_layout(topology, tranche).total_bytes;
    const std::size_t need_out =
        make_output_layout(topology, tranche, request, extra_per_variant).total_bytes +
        fixed_run_bytes(topology);
    const std::size_t have_out = merit_mode ? merit_.capacity_bytes() : residuals_.capacity_bytes();
    if (need_batch > batch_.capacity_bytes() || need_out > have_out) {
        batch_.release();
        if (merit_mode) {
            merit_.release();
        } else {
            residuals_.release();
        }
    }

    return TranchePlan{tranche, usable};
}

TraceOutput TraceEngine::run(const SystemBatchHost& batch, TraceMode mode,
                             const TraceSettings& settings) {
    TraceOutput out;
    run(batch, mode, settings, out);
    return out;
}

void TraceEngine::run(const SystemBatchHost& batch, TraceMode mode, const TraceSettings& settings,
                      TraceOutput& out) {
    // ---- Prüfen (Host, vor jeder Gerätearbeit) ----
    check_mode(mode);
    check_settings(settings);
    const BatchTopology& topology = batch.topology();
    ResidualLayout layout(topology, settings.quadrature); // verlangt festgelegte Pupillenarten
    batch.validate_values();                              // inklusive check_pupil_modes
    const int V = batch.variants();
    const bool merit_mode = mode == TraceMode::Merit;
    if (merit_mode && !settings.extra_merit.empty() &&
        settings.extra_merit.size() != static_cast<std::size_t>(V)) {
        throw std::invalid_argument("TraceEngine: extra_merit muss leer sein oder " +
                                    std::to_string(V) + " Werte haben");
    }
    if (merit_mode && V > 0) {
        // OPL und Status eines Tupels liegen im Shared Memory (fragt nur Kernelattribute ab)
        int samples = 0;
        for (const ResidualFieldPlan& p : layout.plan()) {
            samples = std::max(samples, p.samples);
        }
        const detail::DeviceGuard guard(device_);
        const int limit =
            merit_max_samples(settings.extrema, topology.ghost_plates, topology.ghost_terms);
        if (samples > limit) {
            throw std::invalid_argument("TraceEngine: Modus MERIT erlaubt höchstens " +
                                        std::to_string(limit) + " Stützstellen pro Feld (" +
                                        std::to_string(samples) + " angefordert)");
        }
    }

    // ---- Host-Ausgabe anlegen (Ergebnis des anderen Modus leer) ----
    const int S = topology.surfaces;
    const std::int64_t rays = layout.rays_per_variant();
    const std::int64_t stride = rays + settings.extra_residuals;
    const auto rays_total = static_cast<std::size_t>(V) * static_cast<std::size_t>(rays);
    const std::size_t extrema_total =
        settings.extrema ? static_cast<std::size_t>(V) * static_cast<std::size_t>(S) * 2 : 0;
    out.mode = mode;
    out.info = TraceRunInfo{};
    out.info.variants = V;
    ResidualResult& r = out.residuals;
    MeritResult& m = out.merit;
    clear(r);
    clear(m);
    if (merit_mode) {
        m.variants = V;
        m.rays_per_variant = rays;
        m.merit.resize(static_cast<std::size_t>(V));
        m.failed.resize(static_cast<std::size_t>(V));
        m.extrema.variants = settings.extrema ? V : 0;
        m.extrema.surfaces = settings.extrema ? S : 0;
        m.extrema.values.resize(extrema_total);
    } else {
        r.variants = V;
        r.rays_per_variant = rays;
        r.stride = stride;
        r.residuals.resize(static_cast<std::size_t>(V) * static_cast<std::size_t>(stride));
        r.opd.resize(settings.opd ? rays_total : 0);
        r.status.resize(settings.status ? rays_total : 0);
        r.extrema.variants = settings.extrema ? V : 0;
        r.extrema.surfaces = settings.extrema ? S : 0;
        r.extrema.values.resize(extrema_total);
    }
    out.layout = std::move(layout);
    if (V == 0) {
        out.info.device_bytes = device_bytes();
        return;
    }

    // ---- Tranchengröße und Puffer ----
    const TranchePlan plan = plan_tranches(topology, mode, out.layout, settings, V, 1);
    const int tranche = plan.tranche;

    // ---- Tranchen: Upload, Kernel, Download an die richtige Stelle ----
    ResidualRunOptions residual_options;
    residual_options.opd = settings.opd;
    residual_options.extra_residuals = settings.extra_residuals;
    residual_options.extrema = settings.extrema;
    MeritRunOptions merit_options;
    merit_options.extrema = settings.extrema;
    const auto extrema_stride = static_cast<std::size_t>(S) * 2;
    const detail::DeviceGuard guard(device_);
    for (int first = 0; first < V; first += tranche) {
        const int count = std::min(tranche, V - first);
        const auto v0 = static_cast<std::size_t>(first);
        const auto n = static_cast<std::size_t>(count);
        batch_.upload(batch, first, count);
        if (merit_mode) {
            if (!settings.extra_merit.empty()) {
                merit_options.extra_merit = settings.extra_merit.subspan(v0, n);
            }
            merit_.run(batch_, out.layout, merit_options);
            copy_to_host(m.merit.data() + v0, merit_.device_merit(), n * sizeof(double));
            copy_to_host(m.failed.data() + v0, merit_.device_failed(), n * sizeof(std::int64_t));
            if (settings.extrema) {
                copy_to_host(m.extrema.values.data() + v0 * extrema_stride, merit_.device_extrema(),
                             n * extrema_stride * sizeof(double));
            }
        } else {
            residuals_.run(batch_, out.layout, residual_options);
            copy_to_host(r.residuals.data() + v0 * static_cast<std::size_t>(stride),
                         residuals_.device_residuals(),
                         n * static_cast<std::size_t>(stride) * sizeof(double));
            if (settings.status) {
                copy_to_host(r.status.data() + v0 * static_cast<std::size_t>(rays),
                             residuals_.device_status(), n * static_cast<std::size_t>(rays));
            }
            if (settings.opd) {
                copy_to_host(r.opd.data() + v0 * static_cast<std::size_t>(rays),
                             residuals_.device_opd(),
                             n * static_cast<std::size_t>(rays) * sizeof(double));
            }
            if (settings.extrema) {
                copy_to_host(r.extrema.values.data() + v0 * extrema_stride,
                             residuals_.device_extrema(), n * extrema_stride * sizeof(double));
            }
        }
        ++out.info.tranches;
    }
    out.info.tranche_variants = tranche;
    out.info.budget_bytes = plan.usable;
    out.info.device_bytes = device_bytes();
}

TraceRunInfo TraceEngine::run_on_device(const SystemBatchHost& batch, const TraceSettings& settings,
                                        int granularity, const DeviceTrancheFn& consume) {
    // ---- Prüfen (Host, vor jeder Gerätearbeit) ----
    check_settings(settings);
    if (granularity < 1) {
        throw std::invalid_argument("TraceEngine: granularity muss >= 1 sein");
    }
    if (!consume) {
        throw std::invalid_argument("TraceEngine: run_on_device ohne Rückruf");
    }
    const BatchTopology& topology = batch.topology();
    const ResidualLayout layout(topology, settings.quadrature);
    batch.validate_values();
    const int V = batch.variants();
    TraceRunInfo info;
    info.variants = V;
    if (V == 0) {
        info.device_bytes = device_bytes();
        return info;
    }

    // ---- Tranchengröße und Puffer (wie run, Tranchen als Vielfache von granularity) ----
    const TranchePlan plan =
        plan_tranches(topology, TraceMode::Residuals, layout, settings, V, granularity);

    // ---- Tranchen: Upload, Kernel, Rückruf mit den Gerätezeigern ----
    ResidualRunOptions options;
    options.opd = settings.opd;
    options.extra_residuals = settings.extra_residuals;
    options.extrema = settings.extrema;
    const detail::DeviceGuard guard(device_);
    for (int first = 0; first < V; first += plan.tranche) {
        const int count = std::min(plan.tranche, V - first);
        batch_.upload(batch, first, count);
        residuals_.run(batch_, layout, options);
        DeviceTranche tranche;
        tranche.first = first;
        tranche.count = count;
        tranche.rays_per_variant = residuals_.rays_per_variant();
        tranche.stride = residuals_.stride();
        tranche.residuals = residuals_.device_residuals();
        tranche.status = residuals_.device_status();
        tranche.opd = settings.opd ? residuals_.device_opd() : nullptr;
        tranche.extrema = settings.extrema ? residuals_.device_extrema() : nullptr;
        consume(tranche);
        ++info.tranches;
    }
    info.tranche_variants = plan.tranche;
    info.budget_bytes = plan.usable;
    info.device_bytes = device_bytes();
    return info;
}

} // namespace mko::cuda
