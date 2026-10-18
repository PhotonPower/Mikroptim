// Normalgleichungen auf der GPU (M3-T4): Gram-Matrix [J r₀]ᵀ[J r₀] pro Lauf mit fester
// Summationsreihenfolge (FMA-Kette über die Zeilen, aufsteigend), bitgleich zu
// normal_equations_host. docs/optimierung.md, Abschnitt 4; ADR-0010.

#include "mikroptim/cuda/normal_equations.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mko::cuda {

namespace {

struct GramArgs {
    const double* jacobian; // [runs][P][R]
    const double* r0;       // [runs][R]
    std::int64_t rows;      // R
    int variables;          // P
    int tiles;              // T = ⌈(P + 1)/16⌉
    int pairs;              // T(T + 1)/2 Kachelpaare pro Lauf
    double* gram;           // [runs][P + 1][P + 1]
};

// Spalte c des Laufs: c < P eine Spalte von J, c = P der Residuenvektor r₀, sonst nullptr.
__device__ inline const double* gram_column(const GramArgs& a, std::int64_t run, int c) {
    if (c < a.variables) {
        return a.jacobian + (run * a.variables + c) * a.rows;
    }
    if (c == a.variables) {
        return a.r0 + run * a.rows;
    }
    return nullptr;
}

// Block b = Lauf · pairs + Kachelpaar (tp ≤ tq, zeilenweise über das obere Dreieck der
// Kachelmatrix). Thread (ty, tx) rechnet G(p, q) mit p = 16·tp + ty, q = 16·tq + tx:
// acc ← fma(c_p[r], c_q[r], acc) für r = 0 … R − 1 aufsteigend, ab acc = 0. Geladen wird in
// Kacheln von 16 Zeilen (Thread (ty, tx) lädt Zeile r₀ + tx der Spalten 16·tp + ty und
// 16·tq + ty, zusammenhängend über tx). Die letzte Kachel läuft nur bis R − 1, Polster gehen
// nicht in die Summe ein. Keine Abhängigkeit von anderen Läufen oder der Lage im Grid.
__global__ void __launch_bounds__(kGramTile* kGramTile) gram_kernel(const GramArgs a) {
    __shared__ double sp[kGramTile][kGramTile + 1]; // [Zeile in der Kachel][Spalte p]
    __shared__ double sq[kGramTile][kGramTile + 1]; // [Zeile in der Kachel][Spalte q]
    const std::int64_t run = static_cast<std::int64_t>(blockIdx.x) / a.pairs;
    int rem = static_cast<int>(blockIdx.x % static_cast<unsigned>(a.pairs));
    int tp = 0;
    while (rem >= a.tiles - tp) {
        rem -= a.tiles - tp;
        ++tp;
    }
    const int tq = tp + rem;
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const double* load_p = gram_column(a, run, tp * kGramTile + ty);
    const double* load_q = gram_column(a, run, tq * kGramTile + ty);
    double acc = 0.0;
    for (std::int64_t r0 = 0; r0 < a.rows; r0 += kGramTile) {
        const std::int64_t r = r0 + tx;
        sp[tx][ty] = (load_p != nullptr && r < a.rows) ? load_p[r] : 0.0;
        sq[tx][ty] = (load_q != nullptr && r < a.rows) ? load_q[r] : 0.0;
        __syncthreads();
        const std::int64_t left = a.rows - r0;
        const int kmax = left < kGramTile ? static_cast<int>(left) : kGramTile;
        for (int k = 0; k < kmax; ++k) {
            acc = fma(sp[k][ty], sq[k][tx], acc);
        }
        __syncthreads();
    }
    const int n = a.variables + 1;
    const int p = tp * kGramTile + ty;
    const int q = tq * kGramTile + tx;
    if (p < n && q < n && (tp != tq || p <= q)) {
        double* g = a.gram + run * static_cast<std::int64_t>(n) * n;
        g[static_cast<std::int64_t>(q) * n + p] = acc;
        g[static_cast<std::int64_t>(p) * n + q] = acc;
    }
}

} // namespace

DeviceNormalEquations::~DeviceNormalEquations() {
    release();
}

DeviceNormalEquations::DeviceNormalEquations(DeviceNormalEquations&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_),
      ready_(other.ready_), runs_(other.runs_), variables_(other.variables_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.ready_ = false;
}

DeviceNormalEquations& DeviceNormalEquations::operator=(DeviceNormalEquations&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        ready_ = other.ready_;
        runs_ = other.runs_;
        variables_ = other.variables_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.ready_ = false;
    }
    return *this;
}

void DeviceNormalEquations::release() noexcept {
    ready_ = false;
    if (buffer_ == nullptr) {
        return;
    }
    try {
        const detail::DeviceGuard guard(device_);
        cudaFree(buffer_); // Fehler beim Freigeben sind nicht behebbar
    } catch (...) {
        cudaFree(buffer_);
    }
    buffer_ = nullptr;
    capacity_ = 0;
}

void DeviceNormalEquations::compute(const DeviceFdJacobian& jacobian) {
    if (!jacobian.ready()) {
        throw std::logic_error("DeviceNormalEquations::compute: keine Jacobi-Matrix berechnet");
    }
    ready_ = false;
    const JacobianShape& shape = jacobian.shape();
    const int P = shape.variables;
    const int n = P + 1;
    const int tiles = (n + kGramTile - 1) / kGramTile;
    const int pairs = tiles * (tiles + 1) / 2;
    if (static_cast<std::int64_t>(shape.runs) * pairs > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("DeviceNormalEquations: zu viele Läufe für ein Grid");
    }
    const std::size_t bytes = static_cast<std::size_t>(shape.runs) * static_cast<std::size_t>(n) *
                              static_cast<std::size_t>(n) * sizeof(double);
    if (buffer_ != nullptr && device_ != jacobian.device()) {
        release();
    }
    device_ = jacobian.device();
    const detail::DeviceGuard guard(device_);
    if (bytes > capacity_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            capacity_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, bytes));
        capacity_ = bytes;
    }
    GramArgs args{};
    args.jacobian = jacobian.device_jacobian();
    args.r0 = jacobian.device_residuals();
    args.rows = shape.rows();
    args.variables = P;
    args.tiles = tiles;
    args.pairs = pairs;
    args.gram = static_cast<double*>(buffer_);
    const dim3 block(kGramTile, kGramTile);
    const auto blocks = static_cast<unsigned>(static_cast<std::int64_t>(shape.runs) * pairs);
    gram_kernel<<<blocks, block>>>(args);
    MKO_CUDA_CHECK(cudaGetLastError());
    MKO_CUDA_CHECK(cudaDeviceSynchronize());
    runs_ = shape.runs;
    variables_ = P;
    ready_ = true;
}

std::vector<double> DeviceNormalEquations::download_gram() const {
    if (!ready_) {
        throw std::logic_error(
            "DeviceNormalEquations::download: keine Normalgleichungen berechnet");
    }
    const std::size_t n = static_cast<std::size_t>(variables_) + 1;
    std::vector<double> out(static_cast<std::size_t>(runs_) * n * n);
    const detail::DeviceGuard guard(device_);
    MKO_CUDA_CHECK(
        cudaMemcpy(out.data(), buffer_, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return out;
}

void DeviceNormalEquations::download(std::vector<double>& a, std::vector<double>& g) const {
    const std::vector<double> gram = download_gram();
    const auto P = static_cast<std::size_t>(variables_);
    const std::size_t n = P + 1;
    const auto runs = static_cast<std::size_t>(runs_);
    a.resize(runs * P * P);
    g.resize(runs * P);
    for (std::size_t i = 0; i < runs; ++i) {
        const double* gi = gram.data() + i * n * n;
        for (std::size_t q = 0; q < P; ++q) {
            for (std::size_t p = 0; p < P; ++p) {
                a[i * P * P + q * P + p] = gi[q * n + p];
            }
            g[i * P + q] = gi[P * n + q]; // G(q, P)
        }
    }
}

} // namespace mko::cuda
