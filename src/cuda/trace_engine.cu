// Host-API TraceEngine (M2-T4): Gerätewahl, Prüfung, Tranchen nach freiem Gerätespeicher,
// Rückgabe in Host-Vektoren (docs/architektur.md, Abschnitt "Host-API TraceEngine").

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
        return;
    case TraceMode::Merit:
        throw std::invalid_argument(
            "TraceEngine: Modus MERIT ist noch nicht verfügbar (Kernel folgt mit M2-T3)");
    }
    throw std::invalid_argument("TraceEngine: unbekannter Modus");
}

void check_settings(const TraceSettings& settings) {
    if (settings.extra_residuals < 0) {
        throw std::invalid_argument("TraceEngine: negative Zahl weiterer Residuen");
    }
}

// Ausgaben auf dem Gerät im Modus RESIDUALS, wie DeviceResiduals::run sie anlegt (der Status
// wird immer geschrieben, weil Durchlauf 2 des Kernels ihn liest).
OutputRequest residual_request(const ResidualLayout& layout, const TraceSettings& settings) {
    OutputRequest r;
    r.rays_per_variant = layout.rays_per_variant();
    r.extra_residuals = settings.extra_residuals;
    r.residuals = true;
    r.ray_status = true;
    r.opd = settings.opd;
    r.merit = false;
    r.surface_max = false;
    return r;
}

// Puffer pro Lauf, unabhängig von der Variantenzahl: Feldablage ResidualFieldPlan[F].
std::size_t fixed_run_bytes(const BatchTopology& topology) {
    return align_bytes(static_cast<std::size_t>(topology.fields) * sizeof(ResidualFieldPlan));
}

// Größte Tranche für `usable` Byte nutzbaren Speicher (nach Abzug der Reserve).
int tranche_capacity(std::size_t usable, const BatchTopology& topology,
                     const OutputRequest& request) {
    const std::size_t fixed = fixed_run_bytes(topology);
    if (usable <= fixed) {
        return 0;
    }
    return mko::max_variants(usable - fixed, topology, request, MemoryReserve{0, 0.0});
}

std::string mib(std::size_t bytes) {
    return std::to_string(static_cast<double>(bytes) / (1024.0 * 1024.0)) + " MiB";
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
}

std::size_t TraceEngine::device_bytes() const {
    return batch_.capacity_bytes() + residuals_.capacity_bytes();
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
    return tranche_capacity(usable, topology, residual_request(layout, settings));
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
    const OutputRequest request = residual_request(layout, settings);

    // ---- Host-Ausgabe anlegen ----
    const int V = batch.variants();
    const std::int64_t rays = layout.rays_per_variant();
    const std::int64_t stride = rays + settings.extra_residuals;
    const auto rays_total = static_cast<std::size_t>(V) * static_cast<std::size_t>(rays);
    out.mode = mode;
    out.info = TraceRunInfo{};
    out.info.variants = V;
    ResidualResult& r = out.residuals;
    r.variants = V;
    r.rays_per_variant = rays;
    r.stride = stride;
    r.residuals.resize(static_cast<std::size_t>(V) * static_cast<std::size_t>(stride));
    r.opd.resize(settings.opd ? rays_total : 0);
    r.status.resize(settings.status ? rays_total : 0);
    out.layout = std::move(layout);
    if (V == 0) {
        out.info.device_bytes = device_bytes();
        return;
    }

    // ---- Tranchengröße ----
    const std::size_t available = available_bytes();
    const std::size_t usable = usable_bytes(available, options_.reserve);
    const int capacity = tranche_capacity(usable, topology, request);
    if (capacity < 1) {
        throw CudaError(
            "TraceEngine: nicht einmal eine Variante passt in den Gerätespeicher "
            "(Gerät " +
            std::to_string(device_) + ", verfügbar " + mib(available) + ", nach Reserve " +
            mib(usable) + ", eine Variante braucht " +
            mib(estimate_device_bytes(topology, 1, request) + fixed_run_bytes(topology)) + ")");
    }
    const int tranche = std::min(capacity, V);
    // Reichen die vorhandenen Puffer nicht, vorher beide freigeben: Sonst lägen alter und
    // neuer Puffer kurz gleichzeitig im Speicher, und das Budget würde überschritten.
    const std::size_t need_batch = make_batch_layout(topology, tranche).total_bytes;
    const std::size_t need_out =
        make_output_layout(topology, tranche, request).total_bytes + fixed_run_bytes(topology);
    if (need_batch > batch_.capacity_bytes() || need_out > residuals_.capacity_bytes()) {
        release();
    }

    // ---- Tranchen: Upload, Kernel, Download an die richtige Stelle ----
    ResidualRunOptions run_options;
    run_options.opd = settings.opd;
    run_options.extra_residuals = settings.extra_residuals;
    const detail::DeviceGuard guard(device_);
    for (int first = 0; first < V; first += tranche) {
        const int count = std::min(tranche, V - first);
        batch_.upload(batch, first, count);
        residuals_.run(batch_, out.layout, run_options);
        const auto v0 = static_cast<std::size_t>(first);
        const auto n = static_cast<std::size_t>(count);
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
        ++out.info.tranches;
    }
    out.info.tranche_variants = tranche;
    out.info.budget_bytes = usable;
    out.info.device_bytes = device_bytes();
}

} // namespace mko::cuda
