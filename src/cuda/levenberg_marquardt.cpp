// GPU-Backend des gebündelten Levenberg-Marquardt (M3-T4): Jacobi-Matrizen (finite Differenzen
// oder adjungiert, M5-T3) und Normalgleichungen auf dem Gerät, Merit der Versuchspunkte im Modus
// MERIT der TraceEngine. Steuerlogik gemeinsam mit der CPU (src/optim/levenberg_marquardt.cpp);
// docs/optimierung.md, Abschnitt 4.

#include "mikroptim/cuda/levenberg_marquardt.hpp"

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::cuda {

GpuLmBackend::GpuLmBackend(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                           const LmOptions& options)
    : engine_(engine), set_(set), base_(base), options_(options) {
    if (options.jacobian.extra.count < 0 ||
        (options.jacobian.extra.count > 0 && !options.jacobian.extra.evaluate)) {
        throw std::invalid_argument("Levenberg-Marquardt (GPU): ungültige Zusatzresiduen");
    }
    check_trace_extra_residuals(options.jacobian.trace_extra, "Levenberg-Marquardt (GPU)");
}

int GpuLmBackend::max_jacobian_runs() const {
    const ResidualLayout layout(base_.topology(), options_.jacobian.quadrature);
    const int P = set_.size();
    const JacobianShape one{1, P, layout.trace_rows(), options_.jacobian.extra_count()};
    std::size_t per_run = 0;
    std::size_t held = 0; // schon belegte Puffer, die das Teilstück wiederverwendet
    if (options_.jacobian_method == JacobianMethod::Adjoint) {
        // J, r₀, Status und eine Variante des Batches der Läufe (M5-T3); die Pauschale für
        // Hilfsdaten in buffer_bytes gilt einmal pro Aufruf und ist hier pro Lauf großzügig
        // mitgezählt
        per_run = DeviceAdjointJacobian::buffer_bytes(one, layout.rays_per_variant()) +
                  make_batch_layout(base_.topology(), 1).total_bytes + 7 * kBatchAlignment;
        held = adjoint_.capacity_bytes() + adjoint_.batch_bytes();
    } else {
        const int n = fd_variant_count(options_.jacobian.scheme, P);
        // Ausrichtung der Regionen: großzügig 7 · 256 Byte pro Lauf zusätzlich
        per_run = DeviceFdJacobian::buffer_bytes(one, n) + 7 * kBatchAlignment;
        held = jacobian_.capacity_bytes();
    }
    std::size_t available = device_memory_info(engine_.device()).free_bytes + held;
    if (engine_.options().memory_limit_bytes) {
        available = std::min(available, *engine_.options().memory_limit_bytes);
    }
    const std::size_t budget = usable_bytes(available, engine_.options().reserve) / 2;
    std::size_t runs = budget / per_run;
    if (options_.jacobian_method == JacobianMethod::Adjoint) {
        // ein Kernelstart pro Teilstück: V · F · L Blöcke
        const BatchTopology& t = base_.topology();
        const auto tuples = static_cast<std::size_t>(std::max(1, t.fields * t.wavelengths));
        runs = std::min(runs, static_cast<std::size_t>(kMaxGridBlocks) / tuples);
    }
    return static_cast<int>(std::clamp<std::size_t>(runs, 1, 1u << 30));
}

void GpuLmBackend::normal_equations(std::span<const int> runs, std::span<const double> x,
                                    std::span<double> a, std::span<double> g) {
    const auto P = static_cast<std::size_t>(set_.size());
    const auto count = static_cast<int>(runs.size());
    const int chunk = max_jacobian_runs();
    const bool adjoint = options_.jacobian_method == JacobianMethod::Adjoint;
    ++stats_.jacobian_calls;
    std::vector<double> ca;
    std::vector<double> cg;
    for (int first = 0; first < count; first += chunk) {
        const int m = std::min(chunk, count - first);
        const auto f = static_cast<std::size_t>(first);
        const auto mm = static_cast<std::size_t>(m);
        const std::span<const int> part = runs.subspan(f, mm);
        const SystemBatchHost tmpl = lm_template_batch(base_, part);
        const std::span<const double> xp = x.subspan(f * P, mm * P);
        if (adjoint) {
            // Adjungiert (M5-T3): ein Kernelstart für alle Läufe des Teilstücks, Zusatzzeilen per
            // zentraler FD auf dem Host (Rechentyp FP64 wie der FD-Pfad)
            AdjointJacobianOptions ao = adjoint_options_of(options_.jacobian);
            ao.extra_threads = lm_host_threads(ao.extra_threads,
                                               static_cast<std::int64_t>(m) *
                                                   fd_variant_count(FdScheme::Central, set_.size()),
                                               kLmExtraPerThread);
            adjoint_.compute(engine_, set_, tmpl, xp, ao, Precision::Fp64);
            normal_.compute(adjoint_);
            ++stats_.jacobian_tranches; // ein Kernelstart
        } else {
            FdJacobianOptions fo = options_.jacobian;
            fo.extra_threads = lm_host_threads(fo.extra_threads,
                                               static_cast<std::int64_t>(m) *
                                                   fd_variant_count(fo.scheme, set_.size()),
                                               kLmExtraPerThread);
            jacobian_.compute(engine_, set_, tmpl, xp, fo);
            normal_.compute(jacobian_);
            stats_.jacobian_tranches += jacobian_.run_info().tranches;
        }
        normal_.download(ca, cg);
        std::copy(ca.begin(), ca.end(), a.begin() + static_cast<std::ptrdiff_t>(f * P * P));
        std::copy(cg.begin(), cg.end(), g.begin() + static_cast<std::ptrdiff_t>(f * P));
        ++stats_.jacobian_chunks;
        stats_.jacobian_runs += m;
        stats_.max_runs_per_chunk = std::max(stats_.max_runs_per_chunk, m);
    }
}

void GpuLmBackend::merit(std::span<const int> runs, std::span<const double> x,
                         std::span<double> merit, std::span<std::int64_t> failed,
                         std::span<std::int64_t> hard_failed) {
    const SystemBatchHost batch = lm_point_batch(set_, base_, runs, x);
    const std::vector<double> extra = extra_merit_values(
        options_.jacobian.extra, batch,
        lm_host_threads(options_.jacobian.extra_threads, batch.variants(), kLmExtraPerThread));
    const TraceExtraResiduals& traced = options_.jacobian.trace_extra;
    TraceSettings settings;
    settings.quadrature = options_.jacobian.quadrature;
    settings.extra_merit = extra;
    settings.extrema = traced.count > 0; // Max-Reduktionen für trace-abhängige Residuen
    engine_.run(batch, TraceMode::Merit, settings, output_);
    // Trace-abhängige Residuen aus den Extrema desselben Laufs, zur Merit des Kernels (Strahlen,
    // Strafen, Σ e²) addiert
    const std::vector<double> traced_merit = trace_extra_merit_values(
        traced, batch, output_.merit.extrema,
        lm_host_threads(options_.jacobian.extra_threads, batch.variants(), kLmExtraPerThread));
    for (std::size_t j = 0; j < runs.size(); ++j) {
        merit[j] = output_.merit.merit[j];
        if (!traced_merit.empty()) {
            merit[j] += traced_merit[j];
        }
        failed[j] = output_.merit.failed[j];
        hard_failed[j] = output_.merit.hard_failed[j];
    }
    ++stats_.merit_calls;
    stats_.merit_tranches += output_.info.tranches;
    stats_.merit_points += static_cast<std::int64_t>(runs.size());
}

namespace {

// GPU-Backend einer Runde des Strafverfahrens, das seine Statistik beim Abbau zur Summe aller
// Runden addiert.
class CollectingGpuBackend final : public LmBackend {
public:
    CollectingGpuBackend(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                         const LmOptions& options, LmGpuStats& total)
        : inner_(engine, set, base, options), total_(total) {}
    ~CollectingGpuBackend() override {
        const LmGpuStats& s = inner_.stats();
        total_.jacobian_calls += s.jacobian_calls;
        total_.jacobian_chunks += s.jacobian_chunks;
        total_.jacobian_runs += s.jacobian_runs;
        total_.merit_calls += s.merit_calls;
        total_.merit_points += s.merit_points;
        total_.max_runs_per_chunk = std::max(total_.max_runs_per_chunk, s.max_runs_per_chunk);
        total_.jacobian_tranches += s.jacobian_tranches;
        total_.merit_tranches += s.merit_tranches;
    }
    CollectingGpuBackend(const CollectingGpuBackend&) = delete;
    CollectingGpuBackend& operator=(const CollectingGpuBackend&) = delete;

    void normal_equations(std::span<const int> runs, std::span<const double> x, std::span<double> a,
                          std::span<double> g) override {
        inner_.normal_equations(runs, x, a, g);
    }
    void merit(std::span<const int> runs, std::span<const double> x, std::span<double> merit,
               std::span<std::int64_t> failed, std::span<std::int64_t> hard_failed) override {
        inner_.merit(runs, x, merit, failed, hard_failed);
    }

private:
    GpuLmBackend inner_;
    LmGpuStats& total_;
};

} // namespace

LmResult optimize_lm_gpu(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                         std::span<const double> x0, const LmOptions& options, LmGpuStats* stats) {
    LmGpuStats total{};
    const LmBackendFactory factory = [&](const LmOptions& o, const SystemBatchHost& b) {
        return std::unique_ptr<LmBackend>(
            std::make_unique<CollectingGpuBackend>(engine, set, b, o, total));
    };
    LmResult result = optimize_lm_rounds(factory, set, base, x0, options);
    if (stats != nullptr) {
        *stats = total;
    }
    return result;
}

} // namespace mko::cuda
