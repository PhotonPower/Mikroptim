#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Host-API der Trace-Kernel (M2-T4): Gerätewahl, Prüfung der Eingaben, Aufteilung großer
// Batches in Tranchen nach dem freien Gerätespeicher, Rückgabe in Host-Vektoren. Ablauf,
// Fehler und Thread-Sicherheit: docs/architektur.md, Abschnitt "Host-API TraceEngine".
// Die Implementierung existiert nur in Builds mit MIKROPTIM_ENABLE_CUDA (Bibliothek
// Mikroptim::cuda, Makro MIKROPTIM_HAS_CUDA); Aufrufer ohne CUDA schützen ihren GPU-Pfad mit
// #ifdef MIKROPTIM_HAS_CUDA.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/cuda/residuals.hpp"

#include <cstddef>
#include <cstdint>
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
    ResidualSettings quadrature{};    // Pupillenquadratur N × M (Achsfelder N × 1)
    bool opd = false;                 // OPD pro Strahl ausgeben (z. B. RMS pro Tupel)
    bool status = true;               // Status pro Strahl ausgeben (TraceStatus, 1 Byte)
    std::int64_t extra_residuals = 0; // weitere Residuen pro Variante (Paraxial, Strafen ab
                                      // M3); der Lauf setzt sie auf 0
    // opd, status und extra_residuals gelten nur im Modus Residuals.
    bool extrema = false; // beide Modi: Max-Reduktionen pro Variante und Fläche (M2-T3,
                          // batch_extrema.hpp), in residuals.extrema bzw. merit.extrema
    // Nur Modus Merit: leer oder ein Wert pro Variante des Batches, den der Kernel zur
    // Quadratsumme addiert (Anteil weiterer Residuen, MeritRunOptions::extra_merit). Die
    // Daten müssen bis zur Rückkehr von run gültig bleiben.
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

    int device_ = -1;
    std::string device_name_;
    EngineOptions options_{};
    DeviceSystemBatch batch_;
    DeviceResiduals residuals_;
    DeviceMerit merit_;
};

} // namespace mko::cuda
