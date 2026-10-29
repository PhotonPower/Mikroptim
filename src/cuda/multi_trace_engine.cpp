// Mehrere GPUs (M7-T3): Aufteilung eines Batches auf mehrere TraceEngines nach gemessenem
// Durchsatz, ein Host-Thread pro Gerät (docs/architektur.md, Abschnitt "Mehrere GPUs (M7-T3)").
// Reines Host-C++ in Mikroptim::cuda; die Gerätearbeit macht TraceEngine.

#include "mikroptim/cuda/multi_trace_engine.hpp"

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace mko::cuda {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Abbruch eines Slots, weil ein anderer Slot gescheitert ist (run_on_device); wird beim
// Einsammeln der Fehler ignoriert.
struct SlotAbort {};

// Ausnahme aus einem Rückruf des Aufrufers; wird unverändert weitergereicht.
struct CallerError {
    std::exception_ptr error;
};

// Wirft die Ausnahme `error` mit vorangestelltem `prefix` und gleichem Typ (soweit bekannt);
// andere Typen (z. B. std::bad_alloc) unverändert.
[[noreturn]] void rethrow_with_prefix(const std::exception_ptr& error, const std::string& prefix) {
    try {
        std::rethrow_exception(error);
    } catch (const CallerError& e) {
        std::rethrow_exception(e.error);
    } catch (const CudaError& e) {
        throw CudaError(prefix + e.what());
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(prefix + e.what());
    } catch (const std::out_of_range& e) {
        throw std::out_of_range(prefix + e.what());
    } catch (const std::logic_error& e) {
        throw std::logic_error(prefix + e.what());
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(prefix + e.what());
    }
}

std::size_t index_of(WorkKind kind, Precision precision) {
    return static_cast<std::size_t>(kind) * 2 + static_cast<std::size_t>(precision);
}

void check_kind(WorkKind kind) {
    switch (kind) {
    case WorkKind::Residuals:
    case WorkKind::Merit:
    case WorkKind::Prefilter:
    case WorkKind::Jacobian:
        return;
    }
    throw std::invalid_argument("MultiTraceEngine: unbekannte Art der Arbeit");
}

std::vector<double> normalized(std::span<const double> shares) {
    const double sum = detail::checked_share_sum(shares);
    std::vector<double> out(shares.begin(), shares.end());
    for (double& s : out) {
        s /= sum;
    }
    return out;
}

// Prüfungen wie TraceEngine::run vor jeder Gerätearbeit (Meldungen mit Variantennummern des
// ganzen Batches); liefert das Layout.
ResidualLayout check_run(const SystemBatchHost& batch, TraceMode mode,
                         const TraceSettings& settings) {
    if (mode != TraceMode::Residuals && mode != TraceMode::Merit) {
        throw std::invalid_argument("MultiTraceEngine: unbekannter Modus");
    }
    if (settings.extra_residuals < 0) {
        throw std::invalid_argument("MultiTraceEngine: negative Zahl weiterer Residuen");
    }
    check_precision(settings.precision);
    ResidualLayout layout(batch.topology(), settings.quadrature);
    batch.validate_values();
    const int V = batch.variants();
    if (mode == TraceMode::Merit && !settings.extra_merit.empty() &&
        settings.extra_merit.size() != static_cast<std::size_t>(V)) {
        throw std::invalid_argument("MultiTraceEngine: extra_merit muss leer sein oder " +
                                    std::to_string(V) + " Werte haben");
    }
    return layout;
}

// Einstellungen für den Teilbereich `range` (extra_merit nur im Modus Merit, als Teilstück).
TraceSettings part_settings(const TraceSettings& settings, TraceMode mode,
                            const DeviceRange& range) {
    TraceSettings s = settings;
    if (mode == TraceMode::Merit && !settings.extra_merit.empty()) {
        s.extra_merit = settings.extra_merit.subspan(static_cast<std::size_t>(range.first),
                                                     static_cast<std::size_t>(range.count));
    } else {
        s.extra_merit = {};
    }
    return s;
}

// Zahl der Varianten der Kalibrierung: calibration_variants auf ein Vielfaches von granularity
// aufgerundet; 0 (keine Kalibrierung), wenn der Batch kleiner ist (dort ist die Aufteilung
// unwichtig und die Messung vom Startaufwand bestimmt) oder die Kalibrierung abgeschaltet ist.
int calibration_size(int variants, int requested, int granularity) {
    if (requested <= 0 || variants < requested) {
        return 0;
    }
    const std::int64_t g = granularity;
    const std::int64_t c = (static_cast<std::int64_t>(requested) + g - 1) / g * g;
    return static_cast<int>(std::min<std::int64_t>(c, variants));
}

// Die ersten `count` Varianten von `batch`: der Batch selbst, wenn es alle sind, sonst ein
// Teil-Batch in `storage`.
const SystemBatchHost& head(const SystemBatchHost& batch, int count, SystemBatchHost& storage) {
    if (count == batch.variants()) {
        return batch;
    }
    storage.assign_variants(batch, 0, count);
    return storage;
}

// Kopiert die Werte des Teilergebnisses (count Varianten zu je `per_variant` Elementen) an die
// Stelle `first` von `dst`.
template <class T>
void place(const std::vector<T>& src, std::vector<T>& dst, int first, std::size_t per_variant) {
    if (src.empty()) {
        return;
    }
    const std::size_t offset = static_cast<std::size_t>(first) * per_variant;
    if (offset + src.size() > dst.size()) {
        throw std::logic_error("MultiTraceEngine: Teilergebnis passt nicht in die Ausgabe");
    }
    std::copy(src.begin(), src.end(), dst.begin() + static_cast<std::ptrdiff_t>(offset));
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Slots
// ---------------------------------------------------------------------------------------------

struct MultiTraceEngine::Slot {
    Slot(int device, const EngineOptions& options) : engine(device, options) {}

    TraceEngine engine;
    SystemBatchHost part;      // Teil-Batch (Kapazität wiederverwendet)
    TraceOutput output;        // Ergebnis des Teil-Batches
    PrefilterResult prefilter; // Vorfilterung des Teil-Batches
};

MultiTraceEngine::MultiTraceEngine(const MultiEngineOptions& options) : options_(options) {
    const int count = device_count();
    if (count == 0) {
        throw CudaError("MultiTraceEngine: kein CUDA-Gerät gefunden (Treiber installiert?)");
    }
    if (options_.devices.empty()) {
        options_.devices = usable_devices();
        if (options_.devices.empty()) {
            throw CudaError("MultiTraceEngine: kein CUDA-Gerät ab sm_86 gefunden");
        }
    }
    const auto slots = options_.devices.size();
    if (!options_.weights.empty()) {
        if (options_.weights.size() != slots) {
            throw std::invalid_argument(
                "MultiTraceEngine: " + std::to_string(options_.weights.size()) + " Gewichte für " +
                std::to_string(slots) + " Geräte");
        }
        (void)detail::checked_share_sum(options_.weights);
    }
    if (!options_.slot_engines.empty() && options_.slot_engines.size() != slots) {
        throw std::invalid_argument(
            "MultiTraceEngine: " + std::to_string(options_.slot_engines.size()) +
            " Engine-Einstellungen für " + std::to_string(slots) + " Geräte");
    }
    if (options_.calibration_variants < 0) {
        throw std::invalid_argument("MultiTraceEngine: calibration_variants muss >= 0 sein");
    }
    if (!(options_.calibration_seconds >= 0.0) || !std::isfinite(options_.calibration_seconds)) {
        throw std::invalid_argument("MultiTraceEngine: calibration_seconds muss >= 0 sein");
    }
    if (!(options_.smoothing > 0.0 && options_.smoothing <= 1.0)) {
        throw std::invalid_argument("MultiTraceEngine: smoothing muss in (0, 1] liegen");
    }
    if (!(options_.min_measure_seconds >= 0.0) || !std::isfinite(options_.min_measure_seconds)) {
        throw std::invalid_argument("MultiTraceEngine: min_measure_seconds muss >= 0 sein");
    }
    for (const int d : options_.devices) {
        if (d < 0 || d >= count) {
            throw std::invalid_argument("MultiTraceEngine: Gerät " + std::to_string(d) +
                                        " existiert nicht (vorhanden: 0 … " +
                                        std::to_string(count - 1) + ")");
        }
    }
    slots_.reserve(slots);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const int d = options_.devices[slot];
        EngineOptions eo =
            options_.slot_engines.empty() ? options_.engine : options_.slot_engines[slot];
        const auto k = std::count(options_.devices.begin(), options_.devices.end(), d);
        if (k > 1 && !eo.memory_limit_bytes) {
            // Logische Geräte: Slots desselben Geräts teilen sich dessen Speicher
            eo.memory_limit_bytes = device_memory_info(d).free_bytes / static_cast<std::size_t>(k);
        }
        slots_.push_back(std::make_unique<Slot>(d, eo));
    }
}

MultiTraceEngine::~MultiTraceEngine() = default;
MultiTraceEngine::MultiTraceEngine(MultiTraceEngine&&) noexcept = default;
MultiTraceEngine& MultiTraceEngine::operator=(MultiTraceEngine&&) noexcept = default;

int MultiTraceEngine::size() const {
    return static_cast<int>(slots_.size());
}

TraceEngine& MultiTraceEngine::engine(int slot) {
    if (slot < 0 || slot >= size()) {
        throw std::out_of_range("MultiTraceEngine: Slot " + std::to_string(slot) +
                                " existiert nicht");
    }
    return slots_[static_cast<std::size_t>(slot)]->engine;
}

const TraceEngine& MultiTraceEngine::engine(int slot) const {
    if (slot < 0 || slot >= size()) {
        throw std::out_of_range("MultiTraceEngine: Slot " + std::to_string(slot) +
                                " existiert nicht");
    }
    return slots_[static_cast<std::size_t>(slot)]->engine;
}

void MultiTraceEngine::release() noexcept {
    for (const auto& s : slots_) {
        s->engine.release();
        s->part = SystemBatchHost{};
        s->output = TraceOutput{};
        s->prefilter = PrefilterResult{};
    }
}

// ---------------------------------------------------------------------------------------------
// Anteile
// ---------------------------------------------------------------------------------------------

std::vector<double>& MultiTraceEngine::estimate(WorkKind kind, Precision precision) {
    check_kind(kind);
    check_precision(precision);
    return estimates_[index_of(kind, precision)];
}

const std::vector<double>& MultiTraceEngine::estimate(WorkKind kind, Precision precision) const {
    check_kind(kind);
    check_precision(precision);
    return estimates_[index_of(kind, precision)];
}

bool MultiTraceEngine::has_estimate(WorkKind kind, Precision precision) const {
    return !options_.weights.empty() || !estimate(kind, precision).empty();
}

std::vector<double> MultiTraceEngine::shares(WorkKind kind, Precision precision) const {
    if (!options_.weights.empty()) {
        return normalized(options_.weights);
    }
    const std::vector<double>& e = estimate(kind, precision);
    if (!e.empty()) {
        return e;
    }
    if (kind == WorkKind::Jacobian && !estimate(WorkKind::Residuals, precision).empty()) {
        return estimate(WorkKind::Residuals, precision);
    }
    return std::vector<double>(slots_.size(), 1.0 / static_cast<double>(slots_.size()));
}

void MultiTraceEngine::set_shares(WorkKind kind, Precision precision,
                                  std::span<const double> values) {
    if (values.size() != slots_.size()) {
        throw std::invalid_argument("MultiTraceEngine: " + std::to_string(values.size()) +
                                    " Anteile für " + std::to_string(slots_.size()) + " Geräte");
    }
    estimate(kind, precision) = normalized(values);
}

std::vector<DeviceRange> MultiTraceEngine::plan(int units, WorkKind kind,
                                                Precision precision) const {
    return split_variants(units, 1, shares(kind, precision));
}

bool MultiTraceEngine::record(WorkKind kind, Precision precision,
                              std::span<const DeviceRange> ranges,
                              std::span<const double> seconds) {
    if (!options_.weights.empty()) {
        return false;
    }
    // Ohne Schätzung von den verwendeten Anteilen ausgehen; gespeichert wird nur eine
    // tatsächlich aktualisierte Schätzung (sonst bleibt die Kalibrierung möglich).
    std::vector<double> e = shares(kind, precision);
    if (!update_shares(e, ranges, seconds, options_.smoothing, options_.min_measure_seconds)) {
        return false;
    }
    estimate(kind, precision) = std::move(e);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Parallele Ausführung
// ---------------------------------------------------------------------------------------------

std::vector<double> MultiTraceEngine::run_parallel(std::span<const DeviceRange> ranges,
                                                   const SlotWorkFn& work) {
    if (ranges.size() != slots_.size()) {
        throw std::invalid_argument("MultiTraceEngine: " + std::to_string(ranges.size()) +
                                    " Bereiche für " + std::to_string(slots_.size()) + " Geräte");
    }
    const std::size_t n = slots_.size();
    std::vector<double> seconds(n, 0.0);
    std::vector<std::exception_ptr> errors(n);
    const auto task = [&](std::size_t k) {
        const Clock::time_point t0 = Clock::now();
        try {
            work(static_cast<int>(k), ranges[k]);
        } catch (...) {
            errors[k] = std::current_exception();
        }
        seconds[k] = seconds_since(t0);
    };
    std::vector<std::size_t> active;
    for (std::size_t k = 0; k < n; ++k) {
        if (ranges[k].count > 0) {
            active.push_back(k);
        }
    }
    std::vector<std::thread> threads;
    threads.reserve(active.size());
    std::exception_ptr spawn_error;
    try {
        for (std::size_t i = 1; i < active.size(); ++i) {
            threads.emplace_back(task, active[i]);
        }
    } catch (...) {
        spawn_error = std::current_exception();
    }
    if (!spawn_error && !active.empty()) {
        task(active[0]);
    }
    for (std::thread& t : threads) {
        t.join();
    }
    if (spawn_error) {
        std::rethrow_exception(spawn_error);
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (!errors[k]) {
            continue;
        }
        try {
            std::rethrow_exception(errors[k]);
        } catch (const SlotAbort&) {
            continue; // Folge eines Fehlers in einem anderen Slot
        } catch (...) {
        }
        const TraceEngine& e = slots_[k]->engine;
        rethrow_with_prefix(errors[k], "Gerät " + std::to_string(e.device()) + " (" +
                                           e.device_name() + ", Slot " + std::to_string(k) + "): ");
    }
    return seconds;
}

const SystemBatchHost& MultiTraceEngine::slice(int slot, const SystemBatchHost& batch,
                                               const DeviceRange& range) {
    if (range.first == 0 && range.count == batch.variants()) {
        return batch;
    }
    SystemBatchHost& part = slots_[static_cast<std::size_t>(slot)]->part;
    part.assign_variants(batch, range.first, range.count);
    return part;
}

bool MultiTraceEngine::calibrate_kind(WorkKind kind, Precision precision, int sample_variants,
                                      const std::function<void(int slot)>& sample_work) {
    if (sample_variants <= 0 || slots_.size() < 2 || !options_.weights.empty()) {
        return false;
    }
    const std::vector<DeviceRange> all(slots_.size(), DeviceRange{0, sample_variants});
    std::vector<double> rate(slots_.size(), 0.0);
    const double min_seconds = options_.calibration_seconds;
    run_parallel(all, [&](int slot, const DeviceRange&) {
        sample_work(slot); // Aufwärmen: Kontext, Puffer, Kernel laden
        const Clock::time_point t0 = Clock::now();
        int repeats = 0;
        double elapsed = 0.0;
        do {
            sample_work(slot);
            ++repeats;
            elapsed = seconds_since(t0);
        } while (elapsed < min_seconds && repeats < 64);
        rate[static_cast<std::size_t>(slot)] = static_cast<double>(repeats) *
                                               static_cast<double>(sample_variants) /
                                               std::max(elapsed, 1e-9);
    });
    estimate(kind, precision) = normalized(rate);
    return true;
}

void MultiTraceEngine::begin_run(WorkKind kind, Precision precision, int variants,
                                 std::span<const DeviceRange> ranges,
                                 std::span<const double> used_shares) {
    last_.kind = kind;
    last_.precision = precision;
    last_.variants = variants;
    last_.seconds = 0.0;
    last_.calibrated = false;
    last_.shares_updated = false;
    last_.shares.assign(used_shares.begin(), used_shares.end());
    last_.devices.assign(slots_.size(), DeviceRunInfo{});
    for (std::size_t k = 0; k < slots_.size(); ++k) {
        last_.devices[k].device = slots_[k]->engine.device();
        last_.devices[k].range = ranges[k];
    }
}

void MultiTraceEngine::finish_run(std::span<const DeviceRange> ranges,
                                  std::span<const double> seconds, double total_seconds,
                                  TraceRunInfo& summary) {
    last_.seconds = total_seconds;
    const int variants = summary.variants;
    summary = TraceRunInfo{};
    summary.variants = variants;
    for (std::size_t k = 0; k < slots_.size(); ++k) {
        DeviceRunInfo& d = last_.devices[k];
        d.seconds = seconds[k];
        d.throughput = ranges[k].count > 0 && seconds[k] > 0.0
                           ? static_cast<double>(ranges[k].count) / seconds[k]
                           : 0.0;
        summary.tranches += d.info.tranches;
        summary.tranche_variants = std::max(summary.tranche_variants, d.info.tranche_variants);
        summary.budget_bytes += d.info.budget_bytes;
        summary.device_bytes += slots_[k]->engine.device_bytes();
    }
    last_.shares_updated = record(last_.kind, last_.precision, ranges, seconds);
}

// ---------------------------------------------------------------------------------------------
// Läufe
// ---------------------------------------------------------------------------------------------

TraceOutput MultiTraceEngine::run(const SystemBatchHost& batch, TraceMode mode,
                                  const TraceSettings& settings) {
    TraceOutput out;
    run(batch, mode, settings, out);
    return out;
}

void MultiTraceEngine::run(const SystemBatchHost& batch, TraceMode mode,
                           const TraceSettings& settings, TraceOutput& out) {
    ResidualLayout layout = check_run(batch, mode, settings);
    const int V = batch.variants();
    const bool merit_mode = mode == TraceMode::Merit;
    const WorkKind kind = merit_mode ? WorkKind::Merit : WorkKind::Residuals;
    const Precision precision = settings.precision;

    // ---- Kalibrierung (erster Lauf der Art) ----
    bool calibrated = false;
    if (!has_estimate(kind, precision)) {
        const int c = calibration_size(V, options_.calibration_variants, 1);
        if (c > 0 && slots_.size() > 1) {
            const DeviceRange sample_range{0, c};
            SystemBatchHost sample;
            const SystemBatchHost& s = head(batch, c, sample);
            const TraceSettings ss = part_settings(settings, mode, sample_range);
            calibrated = calibrate_kind(kind, precision, c, [&](int slot) {
                Slot& sl = *slots_[static_cast<std::size_t>(slot)];
                sl.engine.run(s, mode, ss, sl.output);
            });
        }
    }

    const std::vector<double> w = shares(kind, precision);
    const std::vector<DeviceRange> ranges = split_variants(V, 1, w);
    begin_run(kind, precision, V, ranges, w);
    last_.calibrated = calibrated;
    const Clock::time_point t0 = Clock::now();

    // ---- Nur ein Slot mit Arbeit (oder V = 0): direkt in `out` ----
    const auto busy = std::count_if(ranges.begin(), ranges.end(),
                                    [](const DeviceRange& r) { return r.count > 0; });
    if (busy <= 1) {
        std::size_t k = 0;
        while (k + 1 < ranges.size() && ranges[k].count == 0) {
            ++k;
        }
        std::vector<double> seconds(slots_.size(), 0.0);
        TraceEngine& e = slots_[k]->engine;
        try {
            e.run(batch, mode, settings, out);
        } catch (...) {
            rethrow_with_prefix(std::current_exception(), "Gerät " + std::to_string(e.device()) +
                                                              " (" + e.device_name() + ", Slot " +
                                                              std::to_string(k) + "): ");
        }
        seconds[k] = seconds_since(t0);
        last_.devices[k].info = out.info;
        finish_run(ranges, seconds, seconds[k], out.info);
        return;
    }

    // ---- Host-Ausgabe anlegen (wie TraceEngine::run) ----
    const int S = batch.topology().surfaces;
    const std::int64_t rays = layout.rays_per_variant();
    const std::int64_t stride = layout.trace_rows() + settings.extra_residuals;
    const auto Vs = static_cast<std::size_t>(V);
    const auto rays_total = Vs * static_cast<std::size_t>(rays);
    const auto extrema_stride = static_cast<std::size_t>(S) * 2;
    const std::size_t extrema_total = settings.extrema ? Vs * extrema_stride : 0;
    out.mode = mode;
    out.info = TraceRunInfo{};
    out.info.variants = V;
    ResidualResult& r = out.residuals;
    MeritResult& m = out.merit;
    r = ResidualResult{};
    m = MeritResult{};
    if (merit_mode) {
        m.variants = V;
        m.rays_per_variant = rays;
        m.merit.resize(Vs);
        m.failed.resize(Vs);
        m.hard_failed.resize(Vs);
        m.extrema.variants = settings.extrema ? V : 0;
        m.extrema.surfaces = settings.extrema ? S : 0;
        m.extrema.values.resize(extrema_total);
    } else {
        r.variants = V;
        r.rays_per_variant = rays;
        r.penalty_rows = layout.penalty_rows();
        r.stride = stride;
        r.residuals.resize(Vs * static_cast<std::size_t>(stride));
        r.opd.resize(settings.opd ? rays_total : 0);
        r.status.resize(settings.status ? rays_total : 0);
        r.extrema.variants = settings.extrema ? V : 0;
        r.extrema.surfaces = settings.extrema ? S : 0;
        r.extrema.values.resize(extrema_total);
    }
    out.layout = std::move(layout);

    // ---- Slots parallel: Teil-Batch, Engine, Ergebnisse an die Stelle first ----
    const std::vector<double> seconds = run_parallel(ranges, [&](int k, const DeviceRange& range) {
        Slot& sl = *slots_[static_cast<std::size_t>(k)];
        const SystemBatchHost& part = slice(k, batch, range);
        sl.engine.run(part, mode, part_settings(settings, mode, range), sl.output);
        const TraceOutput& o = sl.output;
        if (merit_mode) {
            place(o.merit.merit, m.merit, range.first, 1);
            place(o.merit.failed, m.failed, range.first, 1);
            place(o.merit.hard_failed, m.hard_failed, range.first, 1);
            place(o.merit.extrema.values, m.extrema.values, range.first, extrema_stride);
        } else {
            place(o.residuals.residuals, r.residuals, range.first,
                  static_cast<std::size_t>(stride));
            place(o.residuals.opd, r.opd, range.first, static_cast<std::size_t>(rays));
            place(o.residuals.status, r.status, range.first, static_cast<std::size_t>(rays));
            place(o.residuals.extrema.values, r.extrema.values, range.first, extrema_stride);
        }
        last_.devices[static_cast<std::size_t>(k)].info = o.info;
    });
    finish_run(ranges, seconds, seconds_since(t0), out.info);
}

TraceRunInfo MultiTraceEngine::run_on_device(const SystemBatchHost& batch,
                                             const TraceSettings& settings, int granularity,
                                             const MultiDeviceTrancheFn& consume) {
    if (granularity < 1) {
        throw std::invalid_argument("MultiTraceEngine: granularity muss >= 1 sein");
    }
    if (!consume) {
        throw std::invalid_argument("MultiTraceEngine: run_on_device ohne Rückruf");
    }
    TraceSettings checked = settings;
    checked.extra_merit = {};
    (void)check_run(batch, TraceMode::Residuals, checked);
    const int V = batch.variants();
    const WorkKind kind = WorkKind::Residuals;
    const Precision precision = settings.precision;

    // ---- Kalibrierung mit run im Modus Residuals (ohne Rückruf) ----
    bool calibrated = false;
    if (!has_estimate(kind, precision)) {
        const int c = calibration_size(V, options_.calibration_variants, granularity);
        if (c > 0 && slots_.size() > 1) {
            SystemBatchHost sample;
            const SystemBatchHost& s = head(batch, c, sample);
            TraceSettings ss = checked;
            ss.status = false;
            calibrated = calibrate_kind(kind, precision, c, [&](int slot) {
                Slot& sl = *slots_[static_cast<std::size_t>(slot)];
                sl.engine.run(s, TraceMode::Residuals, ss, sl.output);
            });
        }
    }

    const std::vector<double> w = shares(kind, precision);
    const std::vector<DeviceRange> ranges = split_variants(V, granularity, w);
    begin_run(kind, precision, V, ranges, w);
    last_.calibrated = calibrated;
    TraceRunInfo summary;
    summary.variants = V;
    const Clock::time_point t0 = Clock::now();
    if (V == 0) {
        TraceEngine& e = slots_.front()->engine;
        last_.devices.front().info =
            e.run_on_device(batch, checked, granularity, [](const DeviceTranche&) {});
        finish_run(ranges, std::vector<double>(slots_.size(), 0.0), 0.0, summary);
        return summary;
    }

    std::atomic<bool> failed{false};
    const std::vector<double> seconds = run_parallel(ranges, [&](int k, const DeviceRange& range) {
        Slot& sl = *slots_[static_cast<std::size_t>(k)];
        try {
            const SystemBatchHost& part = slice(k, batch, range);
            last_.devices[static_cast<std::size_t>(k)].info = sl.engine.run_on_device(
                part, checked, granularity, [&](const DeviceTranche& tranche) {
                    if (failed.load()) {
                        throw SlotAbort{};
                    }
                    DeviceTranche t = tranche;
                    t.first += range.first;
                    try {
                        consume(k, t);
                    } catch (...) {
                        throw CallerError{std::current_exception()};
                    }
                });
        } catch (...) {
            failed.store(true);
            throw;
        }
    });
    finish_run(ranges, seconds, seconds_since(t0), summary);
    return summary;
}

PrefilterResult MultiTraceEngine::prefilter_variants(const SystemBatchHost& batch,
                                                     const PrefilterSettings& settings) {
    PrefilterResult out;
    prefilter_variants(batch, settings, out);
    return out;
}

TraceRunInfo MultiTraceEngine::prefilter_variants(const SystemBatchHost& batch,
                                                  const PrefilterSettings& settings,
                                                  PrefilterResult& out) {
    // ---- Prüfen wie TraceEngine::prefilter_variants ----
    check_prefilter_options(settings.options);
    const ResidualLayout layout(batch.topology(), settings.quadrature);
    batch.validate_values();
    const int V = batch.variants();
    if (!settings.extra_merit.empty() &&
        settings.extra_merit.size() != static_cast<std::size_t>(V)) {
        throw std::invalid_argument("MultiTraceEngine: extra_merit muss leer sein oder " +
                                    std::to_string(V) + " Werte haben");
    }
    const WorkKind kind = WorkKind::Prefilter;
    const Precision precision = Precision::Fp64;
    PrefilterSettings slice_settings = settings;
    slice_settings.extra_merit = {};

    bool calibrated = false;
    if (!has_estimate(kind, precision)) {
        const int c = calibration_size(V, options_.calibration_variants, 1);
        if (c > 0 && slots_.size() > 1) {
            SystemBatchHost sample;
            const SystemBatchHost& s = head(batch, c, sample);
            calibrated = calibrate_kind(kind, precision, c, [&](int slot) {
                Slot& sl = *slots_[static_cast<std::size_t>(slot)];
                sl.engine.prefilter_variants(s, slice_settings, sl.prefilter);
            });
        }
    }

    const std::vector<double> w = shares(kind, precision);
    const std::vector<DeviceRange> ranges = split_variants(V, 1, w);
    begin_run(kind, precision, V, ranges, w);
    last_.calibrated = calibrated;
    TraceRunInfo summary;
    summary.variants = V;
    const Clock::time_point t0 = Clock::now();

    // ---- Rohdaten pro Slot an die Stelle first ----
    PrefilterStats stats;
    stats.variants = V;
    stats.rays_per_variant = layout.rays_per_variant();
    const auto n = static_cast<std::size_t>(V);
    stats.merit.assign(n, 0.0);
    stats.optical_length.assign(n, 0.0);
    stats.failed.assign(n, 0);
    stats.uncertain_ok.assign(n, 0);
    stats.uncertain_failed.assign(n, 0);
    std::vector<double> seconds(slots_.size(), 0.0);
    if (V > 0) {
        seconds = run_parallel(ranges, [&](int k, const DeviceRange& range) {
            Slot& sl = *slots_[static_cast<std::size_t>(k)];
            const SystemBatchHost& part = slice(k, batch, range);
            last_.devices[static_cast<std::size_t>(k)].info =
                sl.engine.prefilter_variants(part, slice_settings, sl.prefilter);
            const PrefilterStats& p = sl.prefilter.stats;
            place(p.merit, stats.merit, range.first, 1);
            place(p.optical_length, stats.optical_length, range.first, 1);
            place(p.failed, stats.failed, range.first, 1);
            place(p.uncertain_ok, stats.uncertain_ok, range.first, 1);
            place(p.uncertain_failed, stats.uncertain_failed, range.first, 1);
        });
    }
    // ---- Gemeinsame Entscheidung über den ganzen Batch ----
    out = prefilter_decide(std::move(stats), batch, layout, settings.options, settings.extra_merit);
    finish_run(ranges, seconds, seconds_since(t0), summary);
    return summary;
}

void MultiTraceEngine::calibrate(const SystemBatchHost& sample, TraceMode mode,
                                 const TraceSettings& settings) {
    (void)check_run(sample, mode, settings);
    const WorkKind kind = mode == TraceMode::Merit ? WorkKind::Merit : WorkKind::Residuals;
    const int V = sample.variants();
    const int c =
        options_.calibration_variants > 0 ? std::min(V, options_.calibration_variants) : V;
    if (c <= 0 || slots_.size() < 2 || !options_.weights.empty()) {
        return;
    }
    SystemBatchHost part;
    const SystemBatchHost& s = head(sample, c, part);
    const TraceSettings ss = part_settings(settings, mode, DeviceRange{0, c});
    (void)calibrate_kind(kind, settings.precision, c, [&](int slot) {
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        sl.engine.run(s, mode, ss, sl.output);
    });
}

} // namespace mko::cuda
