// Gebündelter Levenberg-Marquardt auf mehreren GPUs (M7-T3): Aufteilung der Läufe eines
// Aufrufs auf die GpuLmBackends der Slots einer MultiTraceEngine.

#include "mikroptim/cuda/multi_levenberg_marquardt.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mko::cuda {

MultiGpuLmBackend::MultiGpuLmBackend(MultiTraceEngine& engines, const VariableSet& set,
                                     const SystemBatchHost& base, const LmOptions& options)
    : engines_(engines), set_(set) {
    const int n = engines.size();
    backends_.reserve(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        backends_.push_back(std::make_unique<GpuLmBackend>(engines.engine(k), set, base, options));
    }
    jacobian_runs_.assign(static_cast<std::size_t>(n), 0);
    merit_points_.assign(static_cast<std::size_t>(n), 0);
}

MultiGpuLmBackend::~MultiGpuLmBackend() = default;

void MultiGpuLmBackend::normal_equations(std::span<const int> runs, std::span<const double> x,
                                         std::span<double> a, std::span<double> g) {
    const auto P = static_cast<std::size_t>(set_.size());
    const std::vector<DeviceRange> ranges =
        engines_.plan(static_cast<int>(runs.size()), WorkKind::Jacobian, Precision::Fp64);
    const std::vector<double> seconds =
        engines_.run_parallel(ranges, [&](int k, const DeviceRange& r) {
            const auto f = static_cast<std::size_t>(r.first);
            const auto m = static_cast<std::size_t>(r.count);
            backends_[static_cast<std::size_t>(k)]->normal_equations(
                runs.subspan(f, m), x.subspan(f * P, m * P), a.subspan(f * P * P, m * P * P),
                g.subspan(f * P, m * P));
            jacobian_runs_[static_cast<std::size_t>(k)] += r.count;
        });
    engines_.record(WorkKind::Jacobian, Precision::Fp64, ranges, seconds);
}

void MultiGpuLmBackend::merit(std::span<const int> runs, std::span<const double> x,
                              std::span<double> merit, std::span<std::int64_t> failed,
                              std::span<std::int64_t> hard_failed) {
    const auto P = static_cast<std::size_t>(set_.size());
    const std::vector<DeviceRange> ranges =
        engines_.plan(static_cast<int>(runs.size()), WorkKind::Merit, Precision::Fp64);
    const std::vector<double> seconds =
        engines_.run_parallel(ranges, [&](int k, const DeviceRange& r) {
            const auto f = static_cast<std::size_t>(r.first);
            const auto m = static_cast<std::size_t>(r.count);
            backends_[static_cast<std::size_t>(k)]->merit(
                runs.subspan(f, m), x.subspan(f * P, m * P), merit.subspan(f, m),
                failed.subspan(f, m), hard_failed.subspan(f, m));
            merit_points_[static_cast<std::size_t>(k)] += r.count;
        });
    engines_.record(WorkKind::Merit, Precision::Fp64, ranges, seconds);
}

std::vector<LmGpuStats> MultiGpuLmBackend::slot_stats() const {
    std::vector<LmGpuStats> out;
    out.reserve(backends_.size());
    for (const auto& b : backends_) {
        out.push_back(b->stats());
    }
    return out;
}

LmGpuStats MultiGpuLmBackend::stats() const {
    LmGpuStats sum;
    for (const auto& b : backends_) {
        const LmGpuStats& s = b->stats();
        sum.jacobian_calls += s.jacobian_calls;
        sum.jacobian_chunks += s.jacobian_chunks;
        sum.jacobian_runs += s.jacobian_runs;
        sum.merit_calls += s.merit_calls;
        sum.merit_points += s.merit_points;
        sum.max_runs_per_chunk = std::max(sum.max_runs_per_chunk, s.max_runs_per_chunk);
        sum.jacobian_tranches += s.jacobian_tranches;
        sum.merit_tranches += s.merit_tranches;
    }
    return sum;
}

namespace {

// Backend einer Runde des Strafverfahrens (M4-T2), das beim Abbau seine Statistik pro Slot zur
// Summe aller Runden addiert.
class CollectingMultiGpuBackend final : public LmBackend {
public:
    CollectingMultiGpuBackend(MultiTraceEngine& engines, const VariableSet& set,
                              const SystemBatchHost& base, const LmOptions& options,
                              std::vector<LmGpuStats>& total)
        : inner_(engines, set, base, options), total_(total) {}
    ~CollectingMultiGpuBackend() override {
        const std::vector<LmGpuStats> slots = inner_.slot_stats();
        if (total_.size() < slots.size()) {
            total_.resize(slots.size());
        }
        for (std::size_t k = 0; k < slots.size(); ++k) {
            LmGpuStats& t = total_[k];
            const LmGpuStats& s = slots[k];
            t.jacobian_calls += s.jacobian_calls;
            t.jacobian_chunks += s.jacobian_chunks;
            t.jacobian_runs += s.jacobian_runs;
            t.merit_calls += s.merit_calls;
            t.merit_points += s.merit_points;
            t.max_runs_per_chunk = std::max(t.max_runs_per_chunk, s.max_runs_per_chunk);
            t.jacobian_tranches += s.jacobian_tranches;
            t.merit_tranches += s.merit_tranches;
        }
    }
    CollectingMultiGpuBackend(const CollectingMultiGpuBackend&) = delete;
    CollectingMultiGpuBackend& operator=(const CollectingMultiGpuBackend&) = delete;

    void normal_equations(std::span<const int> runs, std::span<const double> x, std::span<double> a,
                          std::span<double> g) override {
        inner_.normal_equations(runs, x, a, g);
    }
    void merit(std::span<const int> runs, std::span<const double> x, std::span<double> merit,
               std::span<std::int64_t> failed, std::span<std::int64_t> hard_failed) override {
        inner_.merit(runs, x, merit, failed, hard_failed);
    }

private:
    MultiGpuLmBackend inner_;
    std::vector<LmGpuStats>& total_;
};

} // namespace

LmResult optimize_lm_multi_gpu(MultiTraceEngine& engines, const VariableSet& set,
                               const SystemBatchHost& base, std::span<const double> x0,
                               const LmOptions& options, std::vector<LmGpuStats>* stats) {
    // Gemeinsamer Ablauf mit dem äußeren Strafverfahren des weichen Versagens (M4-T2)
    std::vector<LmGpuStats> total(static_cast<std::size_t>(engines.size()));
    const LmBackendFactory factory = [&](const LmOptions& o, const SystemBatchHost& b) {
        return std::unique_ptr<LmBackend>(
            std::make_unique<CollectingMultiGpuBackend>(engines, set, b, o, total));
    };
    LmResult result = optimize_lm_rounds(factory, set, base, x0, options);
    if (stats != nullptr) {
        *stats = std::move(total);
    }
    return result;
}

} // namespace mko::cuda
