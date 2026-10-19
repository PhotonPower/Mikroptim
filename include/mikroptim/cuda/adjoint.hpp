#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// Modus ADJOINT, Grundgerüst (M5-T1): Vorwärtsdurchlauf mit Aufzeichnung pro Strahl im Local
// Memory (AdjointRecord oder CheckpointRecord, include/mikroptim/trace/adjoint_record.hpp), danach
// Lesen der Aufzeichnung in absteigender Reihenfolge der Flächen wie im Rückwärtsdurchlauf
// (M5-T2). Residuen, Status und OPD sind bitgleich zum Kernel RESIDUALS. Der Rückwärtsdurchlauf
// und die Einbindung in die TraceEngine folgen in M5-T2/M5-T3; bis dahin ist dies ein Testzugang
// (Aufzeichnung herunterladen, Prüfsumme, Durchsatz). docs/architektur.md, Abschnitt
// "Modus ADJOINT: Vorwärtsdurchlauf mit Aufzeichnung (M5-T1)".

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/config.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/residuals.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mko::cuda {

// Speicherform der Aufzeichnung pro Strahl.
enum class RecordStorage {
    None = 0,       // keine Aufzeichnung (Vergleich: Vorwärtsdurchlauf allein)
    Full = 1,       // AdjointRecord: alle Zustände (p, d, Zweig) pro Fläche
    Checkpoint = 2, // CheckpointRecord: Zustand alle kCheckpointInterval Flächen, Abschnitte
                    // beim Lesen neu vorwärts gerechnet
};

// Name für Meldungen ("none", "full", "checkpoint").
const char* to_string(RecordStorage storage);

// Threads pro Block (wie RESIDUALS: gleiche Kolbensumme, bitgleiche Residuen).
inline constexpr int kAdjointThreads = kResidualThreads;

// Optionen eines Laufs.
struct AdjointForwardOptions {
    RecordStorage storage = RecordStorage::Full;
    // Kapazität der Aufzeichnung (Flächen): 0 = kleinste Klasse >= S (record_capacity_class:
    // 16, 32, 64), sonst 16, 32 oder 64 und >= S.
    int capacity = 0;
    Precision precision = Precision::Fp64; // Rechentyp (M7-T1); Ausgaben immer FP64
    bool opd = false;                      // rohe OPD pro Strahl zusätzlich ablegen
    // Aufzeichnung pro Strahl in den Gerätespeicher schreiben und herunterladen (nur Tests:
    // 49·S Byte pro Strahl).
    bool states = false;
};

// Ergebnis eines Laufs (Download). Strahl i der Variante v liegt bei v · rays_per_variant + i
// (Ablage von ResidualLayout).
struct AdjointForwardResult {
    ResidualResult residuals; // Residuen, Status (immer) und OPD (mit opts.opd) wie RESIDUALS
    RecordStorage storage = RecordStorage::None;
    int capacity = 0;        // gewählte Kapazitätsklasse (0 ohne Aufzeichnung)
    int surfaces = 0;        // S
    std::vector<double> opl; // [V · rays] OPL bis zur Referenzsphäre (Rechentyp nach double; nur
                             // für Strahlen mit in_piston gültig)
    std::vector<std::uint64_t> checksum; // [V · rays] record_checksum der in absteigender
                                         // Reihenfolge gelesenen Aufzeichnung (0 ohne)
    // Nur mit options.states (sonst leer):
    std::vector<std::int32_t> count;        // [V · rays] gültige Zustände
    std::vector<double> states;             // [V · rays][S][6]: p.x, p.y, p.z, d.x, d.y, d.z
                                            // nach Fläche s (s < count, sonst 0)
    std::vector<std::uint8_t> branch;       // [V · rays][S] Zweig an Fläche s (kBranch…)
    std::vector<double> image_s;            // [V · rays] s_ref der Referenzsphäre
    std::vector<std::uint8_t> image_branch; // [V · rays] Zweig der Referenzsphäre

    std::size_t rays() const { return opl.size(); }
};

// Prüfsumme der Aufzeichnung eines Strahls (Determinismus und Vergleich der Speicherformen ohne
// Download): FNV-1a über 64-Bit-Wörter mit zusätzlicher Durchmischung, Start
// kRecordChecksumSeed. Reihenfolge: für s = count − 1 … 0 die Bitmuster von p.x, p.y, p.z, d.x,
// d.y, d.z (als double) und branch[s], danach image_s (Bitmuster), image_branch, count.
inline constexpr std::uint64_t kRecordChecksumSeed = 0xcbf29ce484222325ull;

MKO_HD constexpr std::uint64_t record_checksum_step(std::uint64_t h, std::uint64_t word) {
    h ^= word;
    h *= 0x100000001b3ull;
    h ^= h >> 29;
    return h;
}

// Ressourcen einer Kernel-Variante (cudaFuncGetAttributes): Register pro Thread, Local Memory
// (Stack) pro Thread, statischer Shared Memory pro Block.
struct AdjointKernelInfo {
    int registers = 0;
    std::size_t local_bytes = 0;
    std::size_t shared_bytes = 0;
    int max_threads_per_block = 0;
};

// Ressourcen der Kernel-Variante für (storage, capacity, precision) auf `device` (< 0: aktuelles
// Gerät). capacity 16, 32 oder 64 (bei None ignoriert). Wirft std::invalid_argument, CudaError.
AdjointKernelInfo adjoint_kernel_info(RecordStorage storage, int capacity,
                                      Precision precision = Precision::Fp64, int device = -1);

// Ausgabepuffer des Grundgerüsts ADJOINT auf dem Gerät (RAII, nur verschiebbar), Aufbau wie
// DeviceResiduals: ein cudaMalloc (Residuen, Status, OPD im Layout make_output_layout, dahinter
// Feldablage, OPL, Prüfsummen und optional die Aufzeichnung), wiederverwendet, wenn er groß genug
// ist. CUDA-Fehler werfen CudaError, falsche Argumente std::invalid_argument; alle Aufrufe stellen
// das aktuelle Gerät des Aufrufers wieder her.
class DeviceAdjointForward {
public:
    DeviceAdjointForward() = default;
    ~DeviceAdjointForward();

    DeviceAdjointForward(const DeviceAdjointForward&) = delete;
    DeviceAdjointForward& operator=(const DeviceAdjointForward&) = delete;
    DeviceAdjointForward(DeviceAdjointForward&& other) noexcept;
    DeviceAdjointForward& operator=(DeviceAdjointForward&& other) noexcept;

    // Startet den Kernel für alle Varianten von `batch` auf dessen Gerät und wartet auf das Ende
    // (ein Block pro (v, f, l), V · F · L <= kMaxGridBlocks). Prüfungen wie DeviceResiduals::run;
    // zusätzlich std::invalid_argument bei unbekannter Speicherform oder unzulässiger Kapazität.
    void run(const DeviceSystemBatch& batch, const ResidualLayout& layout,
             const AdjointForwardOptions& options = {});

    // Lädt die Ergebnisse des letzten Laufs herunter. Wirft std::logic_error ohne Lauf.
    AdjointForwardResult download() const;
    void download(AdjointForwardResult& result) const;

    void release() noexcept;

    int device() const { return device_; }
    int capacity() const { return capacity_class_; }
    std::size_t capacity_bytes() const { return bytes_; }

private:
    struct Regions {
        OutputLayout out{};
        BufferRegion plan, opl, checksum, count, states, branch, image_s, image_branch;
        std::size_t total = 0;
    };

    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t bytes_ = 0;
    bool ran_ = false;
    AdjointForwardOptions options_{};
    int capacity_class_ = 0;
    int variants_ = 0;
    int surfaces_ = 0;
    std::int64_t rays_ = 0;
    std::int64_t penalty_rows_ = 0;
    std::int64_t stride_ = 0;
    Regions regions_{};
};

// Bequemlichkeit: ein Lauf mit eigenem Puffer und Download.
AdjointForwardResult run_adjoint_forward(const DeviceSystemBatch& batch,
                                         const ResidualLayout& layout,
                                         const AdjointForwardOptions& options = {});

} // namespace mko::cuda
