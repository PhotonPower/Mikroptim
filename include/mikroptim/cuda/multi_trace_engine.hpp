#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Mehrere GPUs (M7-T3): Ein Batch wird in zusammenhängende Variantenbereiche geteilt, die
// parallel auf mehreren Geräten laufen (je eine TraceEngine und ein Host-Thread pro Gerät);
// die Anteile folgen dem gemessenen Durchsatz der Geräte. Ablauf, Bitgleichheit und gemischte
// Architekturen: docs/architektur.md, Abschnitt "Mehrere GPUs (M7-T3)". Die Klasse
// MultiTraceEngine existiert nur in Builds mit MIKROPTIM_ENABLE_CUDA; die Aufteilung
// (split_variants, update_shares) ist inline und auch ohne CUDA nutzbar.

#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/trace_engine.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::cuda {

// Zusammenhängender Bereich [first, first + count) von Varianten (bzw. Läufen) eines Geräts.
struct DeviceRange {
    int first = 0;
    int count = 0;

    friend bool operator==(const DeviceRange&, const DeviceRange&) = default;
};

// Art der Arbeit, für die ein Durchsatz geschätzt wird. Das Verhältnis der Geräte hängt von
// Modus und Zahlentyp ab (z. B. RTX 3090 in df64 etwa 4-mal so schnell wie in FP64, P100
// umgekehrt), daher eine Schätzung pro (Art, Zahlentyp).
enum class WorkKind : std::uint8_t {
    Residuals = 0, // TraceEngine::run im Modus Residuals und run_on_device
    Merit = 1,     // TraceEngine::run im Modus Merit
    Prefilter = 2, // FP32-Vorfilterung (Zahlentyp ohne Bedeutung, geführt als Fp64)
    Jacobian = 3,  // Jacobi-Matrizen und Normalgleichungen des LM (MultiGpuLmBackend)
};
inline constexpr int kWorkKinds = 4;

// Name für Meldungen ("RESIDUALS", "MERIT", "PREFILTER", "JACOBIAN").
inline const char* to_string(WorkKind kind) {
    switch (kind) {
    case WorkKind::Residuals:
        return "RESIDUALS";
    case WorkKind::Merit:
        return "MERIT";
    case WorkKind::Prefilter:
        return "PREFILTER";
    case WorkKind::Jacobian:
        return "JACOBIAN";
    }
    return "?";
}

namespace detail {

// Prüft Anteile: mindestens einer, alle endlich und >= 0, Summe > 0. Rückgabe: Summe.
inline double checked_share_sum(std::span<const double> shares) {
    if (shares.empty()) {
        throw std::invalid_argument("Mehrere GPUs: keine Anteile");
    }
    double sum = 0.0;
    for (const double s : shares) {
        if (!std::isfinite(s) || s < 0.0) {
            throw std::invalid_argument("Mehrere GPUs: Anteile müssen endlich und >= 0 sein");
        }
        sum += s;
    }
    if (!(sum > 0.0)) {
        throw std::invalid_argument("Mehrere GPUs: Summe der Anteile muss > 0 sein");
    }
    return sum;
}

} // namespace detail

// Teilt `variants` Varianten in shares.size() zusammenhängende Bereiche in Slot-Reihenfolge,
// deren Größen proportional zu `shares` sind (nicht normiert). Einheit der Aufteilung sind
// Blöcke aus `granularity` Varianten (z. B. die 2P + 1 Varianten eines FD-Laufs), die nie
// geteilt werden; nur der letzte Block darf kürzer sein. Grenzen: b_k = round(U · Σ_(i<k) w_i /
// Σ w) mit U = ⌈V/g⌉ Blöcken, Bereich k = [b_k · g, min(b_(k+1) · g, V)). Ein Anteil 0 ergibt
// einen leeren Bereich. Deterministisch (nur von den Argumenten abhängig). Wirft
// std::invalid_argument bei variants < 0, granularity < 1 oder ungültigen Anteilen.
inline std::vector<DeviceRange> split_variants(int variants, int granularity,
                                               std::span<const double> shares) {
    if (variants < 0) {
        throw std::invalid_argument("split_variants: negative Variantenzahl");
    }
    if (granularity < 1) {
        throw std::invalid_argument("split_variants: granularity muss >= 1 sein");
    }
    const double total = detail::checked_share_sum(shares);
    const std::int64_t g = granularity;
    const std::int64_t v = variants;
    const std::int64_t units = (v + g - 1) / g;
    std::vector<DeviceRange> ranges(shares.size());
    double cumulative = 0.0;
    std::int64_t previous = 0;
    for (std::size_t i = 0; i < shares.size(); ++i) {
        cumulative += shares[i];
        std::int64_t bound = i + 1 == shares.size()
                                 ? units
                                 : static_cast<std::int64_t>(std::llround(
                                       static_cast<double>(units) * (cumulative / total)));
        bound = bound < previous ? previous : (bound > units ? units : bound);
        const std::int64_t first = previous * g < v ? previous * g : v;
        const std::int64_t end = bound * g < v ? bound * g : v;
        ranges[i] = DeviceRange{static_cast<int>(first), static_cast<int>(end - first)};
        previous = bound;
    }
    return ranges;
}

// Gleitender Mittelwert der Anteile nach einem Lauf. `shares` (Summe 1) wird nur geändert, wenn
// mindestens zwei Slots Arbeit hatten (count > 0) und jeder davon mindestens `min_seconds`
// brauchte (kürzere Zeiten sind vom Startaufwand bestimmt). Mit dem Durchsatz r_i = count_i /
// seconds_i der beteiligten Slots P und ihrem bisherigen Gesamtanteil S_P gilt
//   s_i ← (1 − α) s_i + α S_P r_i / Σ_(j∈P) r_j   (i ∈ P),  übrige Slots unverändert,
// die Summe bleibt also 1. Rückgabe: true, wenn aktualisiert. Wirft std::invalid_argument bei
// unpassenden Größen oder α ∉ (0, 1].
inline bool update_shares(std::vector<double>& shares, std::span<const DeviceRange> ranges,
                          std::span<const double> seconds, double smoothing, double min_seconds) {
    if (ranges.size() != shares.size() || seconds.size() != shares.size()) {
        throw std::invalid_argument("update_shares: unpassende Größen");
    }
    if (!(smoothing > 0.0 && smoothing <= 1.0)) {
        throw std::invalid_argument("update_shares: Glättung muss in (0, 1] liegen");
    }
    int participants = 0;
    double old_share = 0.0;
    double rate_sum = 0.0;
    for (std::size_t i = 0; i < shares.size(); ++i) {
        if (ranges[i].count <= 0) {
            continue;
        }
        if (!(seconds[i] >= min_seconds) || !(seconds[i] > 0.0) || !std::isfinite(seconds[i])) {
            return false;
        }
        ++participants;
        old_share += shares[i];
        rate_sum += static_cast<double>(ranges[i].count) / seconds[i];
    }
    if (participants < 2 || !(rate_sum > 0.0)) {
        return false;
    }
    for (std::size_t i = 0; i < shares.size(); ++i) {
        if (ranges[i].count > 0) {
            const double rate = static_cast<double>(ranges[i].count) / seconds[i];
            shares[i] = (1.0 - smoothing) * shares[i] + smoothing * old_share * rate / rate_sum;
        }
    }
    return true;
}

// Einstellungen der Mehrgeräte-Engine (fest für ihre Lebensdauer).
struct MultiEngineOptions {
    // Geräteindizes, ein Slot pro Eintrag. Leer: alle sichtbaren Geräte ab sm_60
    // (usable_devices). Ein Gerät darf mehrfach vorkommen ("logische Geräte", Tests mit einer
    // GPU): Jeder Eintrag bekommt eine eigene Engine; ohne memory_limit_bytes begrenzt die
    // Engine jeden der k Slots eines Geräts auf 1/k des beim Erzeugen freien Speichers.
    std::vector<int> devices;
    // Einstellungen jeder Engine (Speichergrenze, Reserve).
    EngineOptions engine{};
    // Leer oder ein Eintrag pro Slot, der `engine` für diesen Slot ersetzt (z. B. eine
    // Speichergrenze nur für eine GPU, die auch der Desktop benutzt).
    std::vector<EngineOptions> slot_engines;
    // Feste Anteile (ein Wert >= 0 pro Slot, Summe > 0), z. B. für Tests; ein Anteil 0 schließt
    // den Slot aus. Leer: Anteile nach gemessenem Durchsatz (Kalibrierung und gleitender
    // Mittelwert).
    std::vector<double> weights;
    // Kalibrierung beim ersten Lauf einer Art (WorkKind, Precision) ohne Schätzung, wenn der
    // Batch mindestens calibration_variants Varianten hat: Jeder Slot rechnet die ersten
    // calibration_variants Varianten (auf ein Vielfaches der Granularität aufgerundet) einmal zum
    // Aufwärmen und dann wiederholt, bis calibration_seconds erreicht sind; die Anteile werden
    // proportional zum gemessenen Durchsatz gesetzt, die Ergebnisse verworfen. Kleinere Batches
    // laufen mit gleichen Anteilen (ihre Aufteilung ist unwichtig). 0: keine Kalibrierung, die
    // Schätzung entsteht nur aus dem gleitenden Mittelwert.
    int calibration_variants = 256;
    double calibration_seconds = 0.02;
    // Gewicht α eines neuen Laufs im gleitenden Mittelwert der Anteile, (0, 1].
    double smoothing = 0.3;
    // Läufe, in denen ein beteiligter Slot kürzer rechnete, ändern die Anteile nicht [s].
    double min_measure_seconds = 0.005;
};

// Statistik eines Slots im letzten Lauf.
struct DeviceRunInfo {
    int device = -1;         // CUDA-Geräteindex
    DeviceRange range;       // bearbeitete Varianten (bzw. Läufe beim LM)
    double seconds = 0.0;    // Wandzeit des Slots (Teil-Batch, Engine, Kopie der Ergebnisse)
    double throughput = 0.0; // range.count / seconds [1/s], 0 ohne Arbeit
    TraceRunInfo info;       // Statistik der Engine des Slots
};

// Statistik des letzten Laufs der Mehrgeräte-Engine.
struct MultiRunInfo {
    WorkKind kind = WorkKind::Residuals;
    Precision precision = Precision::Fp64;
    int variants = 0;
    double seconds = 0.0;               // Wandzeit des ganzen Laufs (ohne Kalibrierung)
    bool calibrated = false;            // vor dem Lauf kalibriert
    bool shares_updated = false;        // Lauf ging in den gleitenden Mittelwert ein
    std::vector<double> shares;         // verwendete Anteile pro Slot (Summe 1)
    std::vector<DeviceRunInfo> devices; // pro Slot
};

// Rückruf pro Tranche bei run_on_device: `slot` ist der Slot (Gerät engine(slot).device()), die
// Tranche wie bei TraceEngine::run_on_device mit `first` bezogen auf den ganzen Batch. Läuft
// im Host-Thread des Slots mit dessen Gerät als aktuellem Gerät, für verschiedene Slots
// gleichzeitig: Der Rückruf muss thread-sicher sein (z. B. nur in Ziele schreiben, die zu
// seinem Variantenbereich gehören, oder pro Slot getrennte Puffer).
using MultiDeviceTrancheFn = std::function<void(int slot, const DeviceTranche&)>;

// Arbeit eines Slots für run_parallel.
using SlotWorkFn = std::function<void(int slot, const DeviceRange& range)>;

// Trace-Kernel auf mehreren Geräten. Jeder Slot besitzt eine TraceEngine; ein Lauf teilt den
// Batch mit split_variants nach den Anteilen der Art (WorkKind, Precision), rechnet die
// Bereiche parallel (ein Host-Thread pro Slot mit Arbeit; der erste davon ist der aufrufende
// Thread) und legt die Ergebnisse an dieselben Stellen wie TraceEngine. Da jede Variante ganz
// auf einem Gerät läuft und der Kernel nicht von ihrer Lage im Batch abhängt, sind die
// Ergebnisse pro Variante unabhängig von der Verteilung bitgleich zum Lauf auf einer Engine
// (ADR-0007) – vorausgesetzt, alle Geräte rechnen denselben Code bitgleich (gleiche
// Architektur; zu sm_60 gegen sm_86 siehe docs/architektur.md, "Mehrere GPUs (M7-T3)").
// Nur verschiebbar; nicht gleichzeitig aus mehreren Threads benutzen.
//
// Fehler: Ungültige Argumente (Einstellungen, Werte im Batch) werfen vor jeder Gerätearbeit wie
// TraceEngine. Fehler auf einem Gerät: Die übrigen Slots rechnen ihren Bereich zu Ende (bei
// run_on_device ohne weitere Rückrufe), alle Threads werden beendet, dann wirft die Engine die
// Ausnahme des kleinsten fehlerhaften Slots mit dem gleichen Typ (CudaError,
// std::invalid_argument, …) und vorangestellter Geräteangabe "Gerät d (Name, Slot k): ".
// Ausnahmen aus Rückrufen des Aufrufers werden unverändert weitergereicht. Nach einer Ausnahme
// ist die Engine weiter benutzbar; der Inhalt der Ausgabe ist unbestimmt.
class MultiTraceEngine {
public:
    // Erzeugt die Engines. Wirft CudaError ohne CUDA-Gerät, std::invalid_argument bei
    // ungültigem Geräteindex (wie TraceEngine), falscher Zahl oder ungültigen Werten von
    // `weights` und ungültigen Kalibrier- oder Glättungswerten.
    explicit MultiTraceEngine(const MultiEngineOptions& options = {});
    ~MultiTraceEngine();

    MultiTraceEngine(const MultiTraceEngine&) = delete;
    MultiTraceEngine& operator=(const MultiTraceEngine&) = delete;
    MultiTraceEngine(MultiTraceEngine&&) noexcept;
    MultiTraceEngine& operator=(MultiTraceEngine&&) noexcept;

    // Wie TraceEngine::run: dieselben Prüfungen, dieselbe Ablage in `out` (Größen, Layout,
    // Ergebnisse an der Stelle jeder Variante). out.info fasst die Slots zusammen (Tranchen,
    // Puffer und Budget summiert, tranche_variants als Maximum); die Statistik pro Slot steht in
    // last_run(). Art der Arbeit: Residuals bzw. Merit mit settings.precision.
    void run(const SystemBatchHost& batch, TraceMode mode, const TraceSettings& settings,
             TraceOutput& out);
    TraceOutput run(const SystemBatchHost& batch, TraceMode mode,
                    const TraceSettings& settings = {});

    // Wie TraceEngine::run_on_device auf allen Slots: Die Bereiche sind Vielfache von
    // `granularity` (zusammengehörige Varianten liegen auf demselben Gerät und in derselben
    // Tranche), `consume` wird pro Tranche im Thread ihres Slots aufgerufen (siehe
    // MultiDeviceTrancheFn). Rückgabe: zusammengefasste Statistik wie bei run.
    TraceRunInfo run_on_device(const SystemBatchHost& batch, const TraceSettings& settings,
                               int granularity, const MultiDeviceTrancheFn& consume);

    // FP32-Vorfilterung (wie TraceEngine::prefilter_variants): Rohdaten pro Slot, danach eine
    // gemeinsame Entscheidung prefilter_decide über den ganzen Batch (Kriterium (c) vergleicht
    // mit der besten Variante aller Geräte). Rückgabe: zusammengefasste Statistik.
    TraceRunInfo prefilter_variants(const SystemBatchHost& batch, const PrefilterSettings& settings,
                                    PrefilterResult& out);
    PrefilterResult prefilter_variants(const SystemBatchHost& batch,
                                       const PrefilterSettings& settings = {});

    // Kalibriert die Anteile der Art von (mode, settings.precision) mit den ersten
    // min(V, calibration_variants) Varianten von `sample` (bei calibration_variants = 0 mit
    // allen) wie oben beschrieben; mit festen Gewichten ohne Wirkung. Wirft wie run.
    void calibrate(const SystemBatchHost& sample, TraceMode mode,
                   const TraceSettings& settings = {});

    // Zahl der Slots und ihre Engines.
    int size() const;
    TraceEngine& engine(int slot);
    const TraceEngine& engine(int slot) const;
    int device(int slot) const { return engine(slot).device(); }
    const MultiEngineOptions& options() const { return options_; }

    // Anteile pro Slot (Summe 1) für eine Art: feste Gewichte (normiert), sonst die Schätzung;
    // ohne Schätzung gleiche Anteile (Jacobian: die Schätzung von Residuals, falls vorhanden).
    std::vector<double> shares(WorkKind kind, Precision precision = Precision::Fp64) const;
    // true, wenn feste Gewichte gelten oder für die Art schon eine Schätzung vorliegt.
    bool has_estimate(WorkKind kind, Precision precision = Precision::Fp64) const;
    // Setzt die Schätzung einer Art (z. B. aus einer früheren Sitzung); normiert. Wirft
    // std::invalid_argument bei falscher Länge oder ungültigen Werten.
    void set_shares(WorkKind kind, Precision precision, std::span<const double> shares);

    // ---- Bausteine für Verfahren auf mehreren Geräten (z. B. MultiGpuLmBackend) -------------
    // Bereiche für `units` Einheiten nach shares(kind, precision), granularity 1.
    std::vector<DeviceRange> plan(int units, WorkKind kind, Precision precision) const;
    // Führt work(slot, range) für alle Slots mit range.count > 0 parallel aus (ein Thread pro
    // Slot, der erste im aufrufenden Thread), wartet auf alle und wirft Fehler wie oben
    // beschrieben. Rückgabe: Wandzeit pro Slot [s] (0 ohne Arbeit). `ranges` hat size()
    // Einträge.
    std::vector<double> run_parallel(std::span<const DeviceRange> ranges, const SlotWorkFn& work);
    // Gleitender Mittelwert (update_shares) der Art mit den gemessenen Zeiten eines Laufs;
    // ohne Wirkung bei festen Gewichten. Rückgabe: true, wenn die Anteile geändert wurden.
    bool record(WorkKind kind, Precision precision, std::span<const DeviceRange> ranges,
                std::span<const double> seconds);

    // Statistik des letzten Laufs (run, run_on_device, prefilter_variants).
    const MultiRunInfo& last_run() const { return last_; }

    // Gibt die Gerätepuffer aller Engines und die Host-Puffer der Teil-Batches frei.
    void release() noexcept;

private:
    struct Slot;
    // Kalibriert die Art: sample_work(slot) rechnet die Stichprobe aus `sample_variants`
    // Varianten auf dem Slot (einmal zum Aufwärmen, dann wiederholt). Ohne Wirkung bei festen
    // Gewichten oder nur einem Slot; Rückgabe: true, wenn kalibriert wurde.
    bool calibrate_kind(WorkKind kind, Precision precision, int sample_variants,
                        const std::function<void(int slot)>& sample_work);
    std::vector<double>& estimate(WorkKind kind, Precision precision);
    const std::vector<double>& estimate(WorkKind kind, Precision precision) const;
    // Teil-Batch eines Slots (ohne Kopie, wenn der Bereich der ganze Batch ist).
    const SystemBatchHost& slice(int slot, const SystemBatchHost& batch, const DeviceRange& range);
    void begin_run(WorkKind kind, Precision precision, int variants,
                   std::span<const DeviceRange> ranges, std::span<const double> shares);
    void finish_run(std::span<const DeviceRange> ranges, std::span<const double> seconds,
                    double total_seconds, TraceRunInfo& summary);

    MultiEngineOptions options_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::array<std::vector<double>, static_cast<std::size_t>(kWorkKinds) * 2> estimates_;
    MultiRunInfo last_;
};

} // namespace mko::cuda
