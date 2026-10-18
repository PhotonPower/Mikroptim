#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// SystemBatch im Gerätespeicher (docs/architektur.md, Abschnitt "SystemBatch und Datenlayout").

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/cuda_error.hpp"

#include <array>
#include <cstddef>

namespace mko::cuda {

// Besitzt einen zusammenhängenden Gerätepuffer mit allen Feldern eines Batches im Layout
// make_batch_layout (RAII, nur verschiebbar). Das Gerät wird beim ersten Upload festgelegt
// (Konstruktorargument oder aktuelles Gerät); alle Aufrufe stellen das aktuelle Gerät des
// Aufrufers danach wieder her. CUDA-Fehler werfen CudaError, falsche Argumente
// std::invalid_argument bzw. std::out_of_range. Alle Kopien sind synchron.
class DeviceSystemBatch {
public:
    DeviceSystemBatch() = default;
    // Leerer Container für Gerät `device` (< 0: aktuelles Gerät beim ersten Upload).
    explicit DeviceSystemBatch(int device) : device_(device) {}
    // Legt den Puffer an und lädt alle Varianten von `host` hoch.
    explicit DeviceSystemBatch(const SystemBatchHost& host, int device = -1);
    ~DeviceSystemBatch();

    DeviceSystemBatch(const DeviceSystemBatch&) = delete;
    DeviceSystemBatch& operator=(const DeviceSystemBatch&) = delete;
    DeviceSystemBatch(DeviceSystemBatch&& other) noexcept;
    DeviceSystemBatch& operator=(DeviceSystemBatch&& other) noexcept;

    // Lädt die gemeinsamen Felder und die Varianten [first, first + count) von `host` hoch
    // (count < 0: bis zum Ende); auf dem Gerät tragen sie die Nummern 0 … count − 1 (Tranchen,
    // Aufteilung auf mehrere GPUs). Der Puffer wird wiederverwendet, wenn er groß genug ist.
    // count = 0 ist erlaubt (nur gemeinsame Felder). Prüft vorher check_pupil_modes für den
    // Bereich (std::invalid_argument). Bei einer Ausnahme ist der Container leer.
    void upload(const SystemBatchHost& host, int first = 0, int count = -1);

    // Schreibt alle Gerätevarianten nach `host` an die Stellen first … first + variants() − 1
    // und überschreibt die gemeinsamen Felder und Skalare. Verlangt gleiche Topologie und
    // first + variants() <= host.variants().
    void download(SystemBatchHost& host, int first = 0) const;
    // Neuer Host-Batch mit genau den Gerätevarianten.
    SystemBatchHost download() const;

    // Gibt den Gerätepuffer frei; danach leer (das Gerät bleibt festgelegt).
    void release() noexcept;

    bool empty() const { return !loaded_; }
    int device() const { return device_; }
    int variants() const { return variants_; }
    const BatchTopology& topology() const { return topology_; }
    const SystemBatchLayout& layout() const { return layout_; }
    std::size_t capacity_bytes() const { return capacity_; }

    // Sicht mit Gerätezeigern für Kernel (per Wert übergeben). Gültig bis zum nächsten
    // upload, release oder zur Zerstörung. Wirft std::logic_error, wenn der Container leer ist.
    SystemBatchView<double> view() const;

private:
    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t capacity_ = 0;
    bool loaded_ = false;
    BatchTopology topology_;
    int variants_ = 0;
    SystemBatchLayout layout_{};
    // Skalare des Batches (liegen in der Sicht, nicht im Puffer)
    double reference_wavelength_nm_ = 0.0;
    std::array<double, 3> source_point_{};
    double source_na_ = 0.0;
};

// Größte Variantenzahl, deren Batch und Ausgaben in den freien Speicher von Gerät `device`
// passen (max_variants mit device_memory_info(device).free_bytes). Wirft CudaError.
int max_variants_on_device(const BatchTopology& topology, const OutputRequest& outputs,
                           const MemoryReserve& reserve = {},
                           std::size_t extra_bytes_per_variant = 0, int device = -1);

} // namespace mko::cuda
