#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Normalgleichungen A = JᵀJ und g = Jᵀr₀ mehrerer Läufe auf der GPU (M3-T4) aus den
// Jacobi-Matrizen auf dem Gerät (DeviceFdJacobian, DeviceAdjointJacobian ab M5-T3). Eigener
// deterministischer Kernel statt cublasDgemmStridedBatched: jedes Element ist eine FMA-Kette über
// die Zeilen in aufsteigender Reihenfolge, bitgleich zu normal_equations_host
// (mikroptim/optim/normal_equations.hpp) und unabhängig von der Zahl und Lage der Läufe im Aufruf
// (docs/optimierung.md, Abschnitt 4; ADR-0010). Nur in Builds mit MIKROPTIM_ENABLE_CUDA.

#include "mikroptim/cuda/adjoint_jacobian.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"

#include <cstddef>
#include <vector>

namespace mko::cuda {

// Kachelgröße des Kernels: Block 16 × 16 Threads für 16 × 16 Elemente der Gram-Matrix.
inline constexpr int kGramTile = 16;

// Sicht auf Jacobi-Matrizen im Gerätespeicher (M5-T3): Gerät, Form und die Zeiger auf
// J_i(r, p) bei (i·P + p)·R + r und r₀_i(r) bei i·R + r (Ablage der FD- und der adjungierten
// Jacobi). Die Zeiger gehören dem Erzeuger und müssen während compute gültig sein.
struct DeviceJacobianRef {
    int device = -1;
    JacobianShape shape{};
    const double* jacobian = nullptr;
    const double* residuals = nullptr;
};

// Sichten auf die fertigen Jacobi-Matrizen (wirft std::logic_error, wenn !ready()).
DeviceJacobianRef device_jacobian_ref(const DeviceFdJacobian& jacobian);
DeviceJacobianRef device_jacobian_ref(const DeviceAdjointJacobian& jacobian);

// Gram-Matrizen G_i = [J_i r₀_i]ᵀ[J_i r₀_i] ((P + 1) × (P + 1) spaltenweise pro Lauf, Abstand
// (P + 1)²) im Gerätespeicher: A_i = G_i[0:P, 0:P], g_i = G_i[0:P, P], r₀ᵀr₀ = G_i[P, P]. RAII,
// nur verschiebbar; der Puffer wird wiederverwendet, wenn er groß genug ist. CUDA-Fehler werfen
// CudaError; alle Aufrufe stellen das aktuelle Gerät des Aufrufers wieder her.
class DeviceNormalEquations {
public:
    DeviceNormalEquations() = default;
    ~DeviceNormalEquations();

    DeviceNormalEquations(const DeviceNormalEquations&) = delete;
    DeviceNormalEquations& operator=(const DeviceNormalEquations&) = delete;
    DeviceNormalEquations(DeviceNormalEquations&& other) noexcept;
    DeviceNormalEquations& operator=(DeviceNormalEquations&& other) noexcept;

    // Rechnet G für alle Läufe von `jacobian` (muss ready() sein, sonst std::logic_error) auf
    // dessen Gerät. Ein Block pro (Lauf, Kachelpaar mit Zeilenkachel ≤ Spaltenkachel); Kacheln
    // von 16 Zeilen der Spalten werden in Shared Memory geladen, jeder Thread führt die
    // FMA-Kette seines Elements in aufsteigender Zeilenfolge fort. Synchron.
    void compute(const DeviceFdJacobian& jacobian);
    // Dasselbe für die adjungierte Jacobi (M5-T3) und jede Sicht mit dieser Ablage; dasselbe J
    // ergibt dieselben Bits, gleich welcher Erzeuger. Wirft std::invalid_argument bei
    // Nullzeigern, ungültigem Gerät oder ungültiger Form.
    void compute(const DeviceAdjointJacobian& jacobian);
    void compute(const DeviceJacobianRef& jacobian);

    // A [runs][P][P] (spaltenweise) und g [runs][P] auf den Host.
    void download(std::vector<double>& a, std::vector<double>& g) const;
    // Ganze Gram-Matrizen [runs][P + 1][P + 1].
    std::vector<double> download_gram() const;

    void release() noexcept;

    bool ready() const { return ready_; }
    int runs() const { return runs_; }
    int variables() const { return variables_; }
    const double* device_gram() const {
        return ready_ ? static_cast<const double*>(buffer_) : nullptr;
    }

private:
    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t capacity_ = 0;
    bool ready_ = false;
    int runs_ = 0;
    int variables_ = 0;
};

} // namespace mko::cuda
