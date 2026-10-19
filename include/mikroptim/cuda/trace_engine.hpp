#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Host-API der Trace-Kernel (M2-T4): Gerätewahl, Prüfung der Eingaben, Aufteilung großer
// Batches in Tranchen nach dem freien Gerätespeicher, Rückgabe in Host-Vektoren. Ablauf,
// Fehler und Thread-Sicherheit: docs/architektur.md, Abschnitt "Host-API TraceEngine".
// Die Implementierung existiert nur in Builds mit MIKROPTIM_ENABLE_CUDA (Bibliothek
// Mikroptim::cuda, Makro MIKROPTIM_HAS_CUDA); Aufrufer ohne CUDA schützen ihren GPU-Pfad mit
// #ifdef MIKROPTIM_HAS_CUDA.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/eval.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/prefilter.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/eval/eval.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>

namespace mko::cuda {

// Kernel-Modus eines Laufs (docs/architektur.md, Abschnitt "Kernel-Modi").
enum class TraceMode {
    Residuals, // Residuenvektor pro Variante, optional OPD und Status pro Strahl (M2-T2)
    Merit,     // Quadratsumme der Residuen und versagte Strahlen pro Variante, ohne den
               // Residuenvektor im Gerätespeicher (M2-T3)
};

// Name des Modus für Meldungen ("RESIDUALS", "MERIT").
const char* to_string(TraceMode mode);

// Einstellungen der Engine (fest für ihre Lebensdauer).
struct EngineOptions {
    // Künstliche Obergrenze für den Speicher, den die Engine als frei ansieht [Byte]
    // (Tests der Tranchen, Teilen einer GPU mit anderen Programmen). Leer: nur cudaMemGetInfo.
    std::optional<std::size_t> memory_limit_bytes;
    // Sicherheitsreserve vom (ggf. begrenzten) freien Speicher, Standard max(256 MiB, 5 %).
    MemoryReserve reserve{};
};

// Einstellungen eines Laufs.
struct TraceSettings {
    ResidualSettings quadrature{};    // Pupillenquadratur N × M (Achsfelder N × 1) und
                                      // Strahlversagen (Standard weich mit Strafresiduen,
                                      // M4-T2; Hard wie in M2)
    bool opd = false;                 // OPD pro Strahl ausgeben (z. B. RMS pro Tupel)
    bool status = true;               // Status pro Strahl ausgeben (TraceStatus, 1 Byte)
    std::int64_t extra_residuals = 0; // weitere Residuen pro Variante nach Strahl- und
                                      // Strafresiduen (Paraxial usw., ab M3); der Lauf setzt
                                      // sie auf 0
    // opd, status und extra_residuals gelten nur im Modus Residuals.
    bool extrema = false; // beide Modi: Max-Reduktionen pro Variante und Fläche (M2-T3,
                          // batch_extrema.hpp), in residuals.extrema bzw. merit.extrema
    // Nur Modus Merit: leer oder ein Wert pro Variante des Batches, den der Kernel zur
    // Quadratsumme addiert (Anteil weiterer Residuen, MeritRunOptions::extra_merit). Die
    // Daten müssen bis zur Rückkehr von run gültig bleiben.
    std::span<const double> extra_merit{};
    // Beide Modi: Rechentyp im Kernel (M7-T1). Df64 (Paar aus zwei float, ca. 48 Bit) ist auf
    // GeForce-Karten (RTX 3090: Kernel ca. 4-mal so schnell) schneller als FP64; OPD und
    // Residuen weichen typisch um einige 1e-11 mm ab, höchstens 1e-9 mm (docs/numerik.md,
    // Abschnitt 12). Batch, Ausgaben (FP64) und Speicherbedarf sind gleich.
    Precision precision = Precision::Fp64;
    // Beide Modi: größte Paketgröße der Kernel in Strahlen (>= 1, M7-T4,
    // ResidualRunOptions::pack_rays). Nur Laufzeit; die Ergebnisse sind für jeden Wert bitgleich.
    int pack_rays = kDefaultPackRays;
};

// Einstellungen der FP32-Vorfilterung (M7-T2, prefilter_variants).
struct PrefilterSettings {
    ResidualSettings quadrature{}; // Pupillenquadratur N × M wie im späteren FP64-Lauf
    PrefilterOptions options{};    // Kriterien und Sicherheitsmargen (batch_prefilter.hpp)
    // Leer oder ein Wert pro Variante (FP64, z. B. Σ e² der paraxialen Nebenbedingungen), der zu
    // beiden Schranken der Merit addiert wird. Muss bis zur Rückkehr gültig bleiben.
    std::span<const double> extra_merit{};
};

// Statistik des letzten Laufs.
struct TraceRunInfo {
    int variants = 0;             // Varianten des Batches
    int tranches = 0;             // Kernelstarts (0 bei leerem Batch)
    int tranche_variants = 0;     // Varianten pro Tranche (die letzte kann kleiner sein)
    std::size_t budget_bytes = 0; // nutzbarer Speicher nach Reserve (usable_bytes)
    std::size_t device_bytes = 0; // von der Engine nach dem Lauf belegter Gerätespeicher
};

// Ergebnis eines Laufs in Host-Vektoren. Wiederverwendbar: run passt die Größen an, die
// Kapazität der Vektoren bleibt erhalten (Schleifen im Optimierer).
struct TraceOutput {
    TraceMode mode = TraceMode::Residuals;
    ResidualLayout layout;    // Ablage der Residuen (aus Topologie und Quadratur)
    ResidualResult residuals; // Modus Residuals: Ablage wie DeviceResiduals::download, OPD,
                              // Status bzw. Extrema leer, wenn nicht angefordert; im Modus
                              // Merit leer
    MeritResult merit;        // Modus Merit: Merit und versagte Strahlen pro Variante, Extrema
                              // auf Anforderung (wie DeviceMerit::download); im Modus
                              // Residuals leer
    TraceRunInfo info;
};

// Ausgaben einer Tranche im Modus Residuals auf dem Gerät (run_on_device). Die Zeiger zeigen in
// die Puffer der Engine, Ablage wie DeviceResiduals (Variante first + j ab j·stride bzw.
// j·rays_per_variant), und gelten nur während des Rückrufs.
struct DeviceTranche {
    int first = 0;                        // erste Variante der Tranche im Batch
    int count = 0;                        // Varianten der Tranche
    std::int64_t rays_per_variant = 0;    // Strahlresiduen pro Variante
    std::int64_t penalty_rows = 0;        // Strafresiduen pro Variante (weich: rays, sonst 0)
    std::int64_t stride = 0;              // rays_per_variant + penalty_rows + extra_residuals
    const double* residuals = nullptr;    // [count · stride], weitere Residuen 0
    const std::uint8_t* status = nullptr; // [count · rays_per_variant], immer vorhanden
    const double* opd = nullptr;          // [count · rays_per_variant], nur mit opd
    const double* extrema = nullptr;      // [count][S][2], nur mit extrema
};

// Rückruf pro Tranche; läuft im Thread des Aufrufers mit dem Gerät der Engine als aktuellem
// Gerät, nachdem der Kernel der Tranche fertig ist. Arbeit, die er im Default-Stream startet,
// ist vor dem Upload der nächsten Tranche beendet.
using DeviceTrancheFn = std::function<void(const DeviceTranche&)>;

// Trace-Kernel auf einem festen CUDA-Gerät. Besitzt die Gerätepuffer (Batch und Ausgaben)
// und verwendet sie über Läufe hinweg wieder; nur verschiebbar. Jeder Aufruf setzt das Gerät
// der Engine und stellt danach das aktuelle Gerät des Aufrufers wieder her. Kein globaler
// Zustand: mehrere Engines (auch auf verschiedenen Geräten, M7) sind unabhängig. Eine Engine
// darf nicht gleichzeitig aus mehreren Threads benutzt werden.
//
// Fehler: CUDA-Fehler und Speichermangel (nicht einmal eine Variante passt) werfen CudaError,
// ungültige Argumente (Geräteindex, Einstellungen, Werte im Batch) std::invalid_argument.
// Nach einer Ausnahme ist die Engine weiter benutzbar; der Inhalt von `out` ist dann
// unbestimmt.
class TraceEngine {
public:
    // Engine für Gerät `device` (0 … device_count() − 1). Wirft CudaError, wenn kein
    // CUDA-Gerät vorhanden ist, std::invalid_argument bei ungültigem Index oder einem Gerät
    // unter sm_60 (keine passende Kernel-Version im Programm).
    explicit TraceEngine(int device, const EngineOptions& options = {});
    ~TraceEngine();

    TraceEngine(const TraceEngine&) = delete;
    TraceEngine& operator=(const TraceEngine&) = delete;
    TraceEngine(TraceEngine&&) noexcept;
    TraceEngine& operator=(TraceEngine&&) noexcept;

    // Traced alle Varianten von `batch` im Modus `mode` und schreibt die Ergebnisse nach
    // `out`. Ablauf: Werte prüfen (validate_values, Pupillenarten), Tranchengröße T aus dem
    // freien Speicher bestimmen (max_variants), dann für jede Tranche [first, first + T)
    // Upload, Kernel, Download an die Stelle `first` der Host-Ausgabe. Die Ergebnisse jeder
    // Variante sind unabhängig von der Aufteilung bitgleich (Kernel ohne Abhängigkeit von der
    // Lage im Batch, ADR-0007). Synchron: Bei der Rückkehr sind alle Daten auf dem Host.
    void run(const SystemBatchHost& batch, TraceMode mode, const TraceSettings& settings,
             TraceOutput& out);
    TraceOutput run(const SystemBatchHost& batch, TraceMode mode,
                    const TraceSettings& settings = {});

    // Modus Residuals ohne Download, für Folgekernel auf dem Gerät (z. B. Differenzen der
    // FD-Jacobi, M3-T2): Ablauf wie run, aber nach dem Kernel jeder Tranche ruft die Engine
    // `consume` mit den Gerätezeigern der Tranche auf, statt die Ausgaben auf den Host zu
    // kopieren. Die Tranchengröße ist ein Vielfaches von `granularity` (≥ 1, z. B. die
    // Varianten eines FD-Laufs), damit zusammengehörige Varianten in derselben Tranche liegen;
    // nur die letzte Tranche darf kleiner sein, wenn V kein Vielfaches ist. settings.status
    // und extra_merit werden ignoriert (der Status liegt immer vor). Speicher, den der
    // Aufrufer vorher selbst belegt, sieht die Engine als belegt (cudaMemGetInfo). Wirft wie
    // run, zusätzlich std::invalid_argument bei granularity < 1 und CudaError, wenn nicht
    // einmal `granularity` Varianten in den Speicher passen; Ausnahmen aus `consume` werden
    // weitergereicht. Rückgabe: Statistik des Laufs.
    TraceRunInfo run_on_device(const SystemBatchHost& batch, const TraceSettings& settings,
                               int granularity, const DeviceTrancheFn& consume);

    // FP32-VORFILTERUNG (M7-T2): bewertet alle Varianten von `batch` grob in float (Kernel der
    // Vorfilterung, Tranchen wie run) und entscheidet konservativ, welche sicher aussortiert
    // werden können (prefilter_decide; Regel und Margen: docs/numerik.md, Abschnitt 12). Die
    // Werte in `out` sind nur Schranken für die Auswahl und dürfen nie in Merit, Jacobi-Matrix
    // oder Optimierung einfließen; behaltene Varianten werden danach mit run in FP64 oder df64
    // bewertet. Der Puffer der Vorfilterung wird beim nächsten run freigegeben, die Puffer von
    // run nur, wenn sie die Tranche verkleinern würden. Wirft wie run, zusätzlich
    // std::invalid_argument bei ungültigen Optionen und zu vielen Stützstellen pro Feld
    // (prefilter_max_samples). Rückgabe: Statistik des Laufs.
    TraceRunInfo prefilter_variants(const SystemBatchHost& batch, const PrefilterSettings& settings,
                                    PrefilterResult& out);
    PrefilterResult prefilter_variants(const SystemBatchHost& batch,
                                       const PrefilterSettings& settings = {});

    // Modus EVAL (M8-T1): wertet alle Varianten von `batch` auf dem dichten kartesischen
    // Pupillenraster aus (OPD nach Kolbenabzug, Spot in der Bildebene, Status, Feldstrahlen;
    // jedes Versagen hart) und schreibt das Ergebnis nach `out` (Ablage wie eval_cpu; Kennzahlen
    // mit eval_statistics). Ablauf wie run: Einstellungen und Werte prüfen, Referenzgeometrie
    // auf dem Host (eval_reference_geometry), Tranchen nach dem freien Speicher
    // (eval_device_bytes), pro Tranche Upload, Kernel, Download an die Stelle first. Der
    // Ausgabepuffer lebt nur während des Aufrufs; die Puffer der anderen Modi werden nur
    // freigegeben, wenn sie die Tranche verkleinern würden. Rechentyp immer FP64. Ergebnisse
    // jeder Variante sind unabhängig von der Aufteilung bitgleich. Wirft wie run, zusätzlich
    // std::invalid_argument bei ungültigen Einstellungen (check_eval_settings) oder einer nicht
    // bestimmbaren Austrittspupille. Rückgabe: Statistik des Laufs.
    TraceRunInfo evaluate(const SystemBatchHost& batch, const EvalSettings& settings,
                          EvalResult& out);
    EvalResult evaluate(const SystemBatchHost& batch, const EvalSettings& settings = {});

    // Größte Variantenzahl pro Tranche für Batches dieser Topologie: max_variants mit dem
    // freien Speicher des Geräts (zuzüglich der von der Engine schon belegten Puffer, die
    // ein Lauf freigibt oder wiederverwendet), höchstens memory_limit_bytes, abzüglich der
    // Reserve und der festen Puffer pro Lauf. 0, wenn keine Variante passt. Wirft wie run
    // (ohne Prüfung der Werte).
    int max_variants(const BatchTopology& topology, TraceMode mode,
                     const TraceSettings& settings = {}) const;

    // Gibt alle Gerätepuffer frei (der nächste Lauf legt sie neu an).
    void release() noexcept;

    int device() const { return device_; }
    const std::string& device_name() const { return device_name_; }
    const EngineOptions& options() const { return options_; }
    // Zurzeit belegter Gerätespeicher (Batch und Ausgaben) [Byte].
    std::size_t device_bytes() const;

private:
    // Speicher, den ein Lauf belegen darf, vor Abzug der Reserve [Byte].
    std::size_t available_bytes() const;
    // Tranchengröße (Vielfaches von granularity, höchstens V) und nutzbarer Speicher; gibt
    // den Puffer des anderen Modus frei, wenn das die Tranche vergrößert, und die Puffer des
    // Modus, wenn sie nicht reichen (docs/architektur.md, „Host-API TraceEngine“).
    struct TranchePlan {
        int tranche = 0;
        std::size_t usable = 0;
    };
    TranchePlan plan_tranches(const BatchTopology& topology, TraceMode mode,
                              const ResidualLayout& layout, const TraceSettings& settings,
                              int variants, int granularity);

    int device_ = -1;
    std::string device_name_;
    EngineOptions options_{};
    DeviceSystemBatch batch_;
    DeviceResiduals residuals_;
    DeviceMerit merit_;
    DevicePrefilter prefilter_; // FP32-Vorfilterung (M7-T2)
};

} // namespace mko::cuda
