// Vektorielle PSF auf der GPU (M8-T5): VectorPsfEngine mit cuFFT (Z2Z, FP64). Pro Tranche von T
// Pupillen mit K Eingangsfeldern: Feldkomponenten in der Pupille (Kernel, T·K·3 Felder), FFT
// vorwärts (ein Plan mit Batch T·K·3), Intensität, Komponenten und zentrierte PSF (Kernel), FFT
// rückwärts der Intensität (zweiter Plan mit Batch T und Abstand K·3·M²), MTF-Schnitte (Kernel),
// Download; Kennzahlen auf dem Host (finish_vector_psf). Alle Kernel sind elementweise, ohne
// Reduktionen. Ablauf: docs/architektur.md, Abschnitt "Vektorielle PSF".

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/vector_psf.hpp"
#include "mikroptim/eval/psf_ops.hpp"
#include "mikroptim/eval/vector_psf_ops.hpp"

#include "cuda_check.hpp"
#include "cufft_check.hpp"
#include "device_guard.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::cuda {

namespace {

constexpr int kVectorPsfThreads = 256;

__device__ inline double quiet_nan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

struct VectorPsfKernelArgs {
    int n;          // Rasterkante N
    std::int64_t m; // FFT-Größe M
    int slots;      // T (Plätze der Tranche)
    int pupils;     // gültige Pupillen dieser Tranche (<= T)
    int inputs;     // K Eingangsfelder
    VectorPsfJones jones[kMaxVectorPsfInputs];
    const double* opd;             // [T·N²]
    const VectorPsfParams* params; // [T]
    cufftDoubleComplex* spectrum;  // [T·K·3·M²]
    double* psf;                   // [T·M²]
    double* component;             // [T·3·M²]
    double* mtf;                   // [T·2(N + 1)]
};

unsigned grid_for(std::int64_t total) {
    const std::int64_t blocks = (total + kVectorPsfThreads - 1) / kVectorPsfThreads;
    return static_cast<unsigned>(
        std::min<std::int64_t>(std::max<std::int64_t>(blocks, 1), 1 << 20));
}

// Feldkomponenten G_c der Pupille in die Ecke (0, 0) der M×M-Felder (t, k, c), sonst 0 (auch für
// leere Plätze t >= pupils). Ein Thread pro Pixel und Eingangsfeld schreibt alle drei
// Komponenten.
__global__ void __launch_bounds__(kVectorPsfThreads)
    vector_psf_pupil_kernel(const VectorPsfKernelArgs a) {
    const std::int64_t m2 = a.m * a.m;
    const std::int64_t per_slot = static_cast<std::int64_t>(a.inputs) * m2;
    const std::int64_t total = static_cast<std::int64_t>(a.slots) * per_slot;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const std::int64_t t = idx / per_slot;
        const std::int64_t rest = idx - t * per_slot;
        const std::int64_t k = rest / m2;
        const std::int64_t r = rest - k * m2;
        const std::int64_t j = r / a.m;
        const std::int64_t i = r - j * a.m;
        double re[3] = {0.0, 0.0, 0.0};
        double im[3] = {0.0, 0.0, 0.0};
        if (t < a.pupils && i < a.n && j < a.n) {
            const std::int64_t n = a.n;
            const double x = static_cast<double>(2 * i + 1 - n) / static_cast<double>(n);
            const double y = static_cast<double>(2 * j + 1 - n) / static_cast<double>(n);
            vector_psf_pupil_value(a.params[t], a.jones[k], x, y, a.opd[t * n * n + j * n + i], re,
                                   im);
        }
        const std::int64_t base = ((t * a.inputs + k) * 3) * m2 + r;
        for (int c = 0; c < 3; ++c) {
            a.spectrum[base + c * m2] = make_cuDoubleComplex(re[c], im[c]);
        }
    }
}

// Komponenten Σ_k w_k |E_kc|² / I_0 und Intensität (Summe) an die zentrierte (ggf. gespiegelte)
// Stelle der Ausgaben; die Intensität reell in das Feld (t, 0, 0) (Eingabe der FFT⁻¹).
__global__ void __launch_bounds__(kVectorPsfThreads)
    vector_psf_intensity_kernel(const VectorPsfKernelArgs a) {
    const std::int64_t m2 = a.m * a.m;
    const std::int64_t total = static_cast<std::int64_t>(a.slots) * m2;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t fields = static_cast<std::int64_t>(a.inputs) * 3;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const std::int64_t t = idx / m2;
        const std::int64_t r = idx - t * m2;
        cufftDoubleComplex* slot = a.spectrum + t * fields * m2;
        if (t >= a.pupils) {
            slot[r] = make_cuDoubleComplex(0.0, 0.0);
            continue;
        }
        const VectorPsfParams p = a.params[t];
        double comp[3];
        for (int c = 0; c < 3; ++c) {
            double acc = 0.0;
            for (int k = 0; k < a.inputs; ++k) {
                const cufftDoubleComplex u = slot[(static_cast<std::int64_t>(k) * 3 + c) * m2 + r];
                acc += a.jones[k].weight * psf_norm2(cuCreal(u), cuCimag(u));
            }
            comp[c] = acc * p.scale;
        }
        const double intensity = (comp[0] + comp[1]) + comp[2];
        const std::int64_t qj = r / a.m;
        const std::int64_t qi = r - qj * a.m;
        const std::int64_t o =
            psf_output_index(qj, a.m, p.sign_y) * a.m + psf_output_index(qi, a.m, p.sign_x);
        a.psf[t * m2 + o] = intensity;
        for (int c = 0; c < 3; ++c) {
            a.component[(t * 3 + c) * m2 + o] = comp[c];
        }
        slot[r] = make_cuDoubleComplex(intensity, 0.0);
    }
}

// MTF-Schnitte aus der OTF im Feld (t, 0, 0): sagittal (k, 0), tangential (0, k), k = 0 … N,
// normiert auf OTF(0).
__global__ void __launch_bounds__(kVectorPsfThreads)
    vector_psf_mtf_kernel(const VectorPsfKernelArgs a) {
    const std::int64_t per = static_cast<std::int64_t>(a.n) + 1;
    const std::int64_t total = static_cast<std::int64_t>(a.pupils) * per;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t fields = static_cast<std::int64_t>(a.inputs) * 3;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const std::int64_t t = idx / per;
        const std::int64_t k = idx - t * per;
        const cufftDoubleComplex* otf = a.spectrum + t * fields * a.m * a.m;
        const double otf0 = cuCreal(otf[0]);
        const cufftDoubleComplex s = otf[k];
        const cufftDoubleComplex g = otf[k * a.m];
        double* out = a.mtf + t * 2 * per;
        out[k] = otf0 > 0.0 ? sqrt(psf_norm2(cuCreal(s), cuCimag(s))) / otf0 : quiet_nan();
        out[per + k] = otf0 > 0.0 ? sqrt(psf_norm2(cuCreal(g), cuCimag(g))) / otf0 : quiet_nan();
    }
}

std::string mib(std::size_t bytes) {
    return std::to_string(static_cast<double>(bytes) / (1024.0 * 1024.0)) + " MiB";
}

// Arbeitsspeicher eines Plans M×M mit Batch b und Abstand dist zwischen den Feldern (Abschätzung
// vor dem Anlegen) [Byte].
std::size_t estimate_work(long long m, long long batch, long long dist) {
    cufftHandle h = 0;
    MKO_CUFFT_CHECK(cufftCreate(&h));
    long long n[2] = {m, m};
    long long embed[2] = {m, m};
    std::size_t ws = 0;
    const cufftResult r =
        cufftGetSizeMany64(h, 2, n, embed, 1, dist, embed, 1, dist, CUFFT_Z2Z, batch, &ws);
    cufftDestroy(h);
    if (r != CUFFT_SUCCESS) {
        detail::throw_cufft_error(r, "cufftGetSizeMany64", __FILE__, __LINE__);
    }
    return ws;
}

// Ein cuFFT-Plan ohne automatischen Arbeitsspeicher (der Bereich liegt im Puffer der Engine).
struct Plan {
    cufftHandle handle = 0;
    bool valid = false;
    long long m = 0;
    long long batch = 0;
    long long dist = 0;
    std::size_t work = 0;

    void destroy() noexcept {
        if (valid) {
            cufftDestroy(handle); // Fehler beim Freigeben sind nicht behebbar
            valid = false;
            m = 0;
            batch = 0;
            dist = 0;
            work = 0;
        }
    }

    void make(long long size, long long count, long long distance) {
        if (valid && m == size && batch == count && dist == distance) {
            return;
        }
        destroy();
        MKO_CUFFT_CHECK(cufftCreate(&handle));
        valid = true;
        MKO_CUFFT_CHECK(cufftSetAutoAllocation(handle, 0));
        long long n[2] = {size, size};
        long long embed[2] = {size, size};
        std::size_t ws = 0;
        MKO_CUFFT_CHECK(cufftMakePlanMany64(handle, 2, n, embed, 1, distance, embed, 1, distance,
                                            CUFFT_Z2Z, count, &ws));
        m = size;
        batch = count;
        dist = distance;
        work = ws;
    }
};

} // namespace

// ---------------------------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------------------------

struct VectorPsfEngine::Impl {
    int device = 0;
    EngineOptions options{};
    void* buffer = nullptr;
    std::size_t capacity = 0;
    Plan fields; // Feldkomponenten: Batch T·K·3, Abstand M²
    Plan otf;    // Intensität: Batch T, Abstand K·3·M²
    PsfRunInfo info{};

    ~Impl() { release(); }

    void destroy_plans() noexcept {
        fields.destroy();
        otf.destroy();
    }

    void release() noexcept {
        try {
            const detail::DeviceGuard guard(device);
            destroy_plans();
            if (buffer != nullptr) {
                cudaFree(buffer);
            }
        } catch (...) {
            destroy_plans();
            if (buffer != nullptr) {
                cudaFree(buffer);
            }
        }
        buffer = nullptr;
        capacity = 0;
    }

    std::size_t work_bytes() const { return std::max(fields.work, otf.work); }

    std::size_t available_bytes() const {
        std::size_t available = device_memory_info(device).free_bytes + capacity;
        if (options.memory_limit_bytes) {
            available = std::min(available, *options.memory_limit_bytes);
        }
        return available;
    }
};

// ---------------------------------------------------------------------------------------------
// VectorPsfEngine
// ---------------------------------------------------------------------------------------------

VectorPsfEngine::VectorPsfEngine(int device, const EngineOptions& options)
    : impl_(std::make_unique<Impl>()) {
    const int count = device_count();
    if (count == 0) {
        throw CudaError("VectorPsfEngine: kein CUDA-Gerät gefunden (Treiber installiert?)");
    }
    if (device < 0 || device >= count) {
        throw std::invalid_argument("VectorPsfEngine: Gerät " + std::to_string(device) +
                                    " existiert nicht (vorhanden: 0 … " +
                                    std::to_string(count - 1) + ")");
    }
    (void)usable_bytes(0, options.reserve); // prüft die Reserve
    impl_->device = device;
    impl_->options = options;
}

VectorPsfEngine::~VectorPsfEngine() = default;
VectorPsfEngine::VectorPsfEngine(VectorPsfEngine&&) noexcept = default;
VectorPsfEngine& VectorPsfEngine::operator=(VectorPsfEngine&&) noexcept = default;

const PsfRunInfo& VectorPsfEngine::last_run() const {
    return impl_->info;
}

int VectorPsfEngine::device() const {
    return impl_->device;
}

const EngineOptions& VectorPsfEngine::options() const {
    return impl_->options;
}

std::size_t VectorPsfEngine::device_bytes() const {
    return impl_ ? impl_->capacity : 0;
}

void VectorPsfEngine::release() noexcept {
    if (impl_) {
        impl_->release();
    }
}

std::vector<VectorPsfResult> VectorPsfEngine::compute(const SystemBatchHost& batch,
                                                      const EvalResult& result, int v,
                                                      const PsfSettings& psf_settings,
                                                      const VectorPsfSettings& settings) {
    const std::vector<VectorPsfPupil> pupils =
        make_vector_psf_pupils(batch, result, v, psf_settings);
    return compute(pupils, psf_fft_size(result.grid.size, psf_settings.padding), settings);
}

std::vector<VectorPsfResult> VectorPsfEngine::compute(std::span<const VectorPsfPupil> pupils,
                                                      int fft_size,
                                                      const VectorPsfSettings& settings) {
    Impl& s = *impl_;
    // ---- Prüfen und Host-Ergebnisse anlegen (vor jeder Gerätearbeit) ----
    std::vector<VectorPsfResult> out;
    out.reserve(pupils.size());
    for (const VectorPsfPupil& p : pupils) {
        if (p.pupil.grid != pupils.front().pupil.grid) {
            throw std::invalid_argument(
                "VectorPsfEngine: alle Pupillen brauchen dieselbe Rasterkante");
        }
        out.push_back(prepare_vector_psf_result(p, fft_size, settings));
    }
    s.info = PsfRunInfo{};
    s.info.pupils = static_cast<int>(pupils.size());
    if (pupils.empty()) {
        s.info.device_bytes = s.capacity;
        return out;
    }
    const VectorPsfInputs inputs = vector_psf_inputs(settings.polarization);
    const int n = pupils.front().pupil.grid;
    const long long m = fft_size;
    const long long fields = 3LL * inputs.count;
    const int count = static_cast<int>(pupils.size());

    const detail::DeviceGuard guard(s.device);

    // ---- Tranche T: größte Zahl Pupillen mit Puffern + cuFFT-Arbeitsspeicher <= nutzbar ----
    const std::size_t usable = usable_bytes(s.available_bytes(), s.options.reserve);
    const std::size_t per_pupil =
        make_vector_psf_device_layout(n, fft_size, inputs.count, 1).total_bytes +
        align_bytes(std::max(estimate_work(m, fields, m * m), estimate_work(m, 1, fields * m * m)));
    int tranche = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(count), usable / std::max<std::size_t>(per_pupil, 1)));
    std::size_t need = 0;
    while (tranche >= 1) {
        // Schon passende Pläne bleiben erhalten; sonst neu (Arbeitsspeicher exakt).
        s.fields.make(m, fields * tranche, m * m);
        s.otf.make(m, tranche, fields * m * m);
        need = make_vector_psf_device_layout(n, fft_size, inputs.count, tranche).total_bytes +
               align_bytes(s.work_bytes());
        if (need <= usable) {
            break;
        }
        tranche = tranche > 1 ? std::max(1, tranche * 3 / 4) : 0;
    }
    if (tranche < 1) {
        s.destroy_plans();
        throw CudaError("VectorPsfEngine: nicht eine Pupille passt in den Gerätespeicher (Gerät " +
                        std::to_string(s.device) + ", nach Reserve " + mib(usable) + ", benötigt " +
                        mib(per_pupil) + " für M = " + std::to_string(m) +
                        "); Raster oder Zero-Padding verkleinern");
    }
    const VectorPsfDeviceLayout layout =
        make_vector_psf_device_layout(n, fft_size, inputs.count, tranche);
    if (need > s.capacity) {
        if (s.buffer != nullptr) {
            MKO_CUDA_CHECK(cudaFree(s.buffer));
            s.buffer = nullptr;
            s.capacity = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&s.buffer, need));
        s.capacity = need;
    }
    char* base = static_cast<char*>(s.buffer);
    // Beide Pläne laufen nacheinander im selben Strom und teilen den Arbeitsspeicher.
    if (s.fields.work > 0) {
        MKO_CUFFT_CHECK(cufftSetWorkArea(s.fields.handle, base + layout.total_bytes));
    }
    if (s.otf.work > 0) {
        MKO_CUFFT_CHECK(cufftSetWorkArea(s.otf.handle, base + layout.total_bytes));
    }

    VectorPsfKernelArgs args{};
    args.n = n;
    args.m = m;
    args.slots = tranche;
    args.inputs = inputs.count;
    for (int k = 0; k < kMaxVectorPsfInputs; ++k) {
        args.jones[k] = inputs.jones[k];
    }
    args.opd = reinterpret_cast<const double*>(base + layout.opd.offset);
    args.params = reinterpret_cast<const VectorPsfParams*>(base + layout.params.offset);
    args.spectrum = reinterpret_cast<cufftDoubleComplex*>(base + layout.spectrum.offset);
    args.psf = reinterpret_cast<double*>(base + layout.psf.offset);
    args.component = reinterpret_cast<double*>(base + layout.component.offset);
    args.mtf = reinterpret_cast<double*>(base + layout.mtf.offset);
    const std::int64_t m2i = static_cast<std::int64_t>(m) * m;
    const auto n2 = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const auto m2 = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
    const std::size_t per = static_cast<std::size_t>(n) + 1;

    std::vector<double> opd_stage(static_cast<std::size_t>(tranche) * n2);
    std::vector<VectorPsfParams> params(static_cast<std::size_t>(tranche));
    std::vector<double> mtf_stage(static_cast<std::size_t>(tranche) * 2 * per);

    for (int first = 0; first < count; first += tranche) {
        const int used = std::min(tranche, count - first);
        for (int t = 0; t < used; ++t) {
            const auto q = static_cast<std::size_t>(first + t);
            const VectorPsfPupil& p = pupils[q];
            std::copy(p.pupil.opd.begin(), p.pupil.opd.end(),
                      opd_stage.begin() +
                          static_cast<std::ptrdiff_t>(static_cast<std::size_t>(t) * n2));
            params[static_cast<std::size_t>(t)] =
                make_vector_psf_params(p, settings, out[q].reference_intensity);
        }
        MKO_CUDA_CHECK(cudaMemcpy(base + layout.opd.offset, opd_stage.data(),
                                  static_cast<std::size_t>(used) * n2 * sizeof(double),
                                  cudaMemcpyHostToDevice));
        MKO_CUDA_CHECK(cudaMemcpy(base + layout.params.offset, params.data(),
                                  static_cast<std::size_t>(used) * sizeof(VectorPsfParams),
                                  cudaMemcpyHostToDevice));
        args.pupils = used;

        vector_psf_pupil_kernel<<<grid_for(static_cast<std::int64_t>(tranche) * inputs.count * m2i),
                                  kVectorPsfThreads>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUFFT_CHECK(cufftExecZ2Z(s.fields.handle, args.spectrum, args.spectrum, CUFFT_FORWARD));
        vector_psf_intensity_kernel<<<grid_for(static_cast<std::int64_t>(tranche) * m2i),
                                      kVectorPsfThreads>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUFFT_CHECK(cufftExecZ2Z(s.otf.handle, args.spectrum, args.spectrum, CUFFT_INVERSE));
        vector_psf_mtf_kernel<<<grid_for(static_cast<std::int64_t>(used) *
                                         static_cast<std::int64_t>(per)),
                                kVectorPsfThreads>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUDA_CHECK(cudaDeviceSynchronize());

        MKO_CUDA_CHECK(cudaMemcpy(mtf_stage.data(), base + layout.mtf.offset,
                                  static_cast<std::size_t>(used) * 2 * per * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        for (int t = 0; t < used; ++t) {
            const auto q = static_cast<std::size_t>(first + t);
            VectorPsfResult& r = out[q];
            const auto ts = static_cast<std::size_t>(t);
            MKO_CUDA_CHECK(cudaMemcpy(r.psf.data(),
                                      base + layout.psf.offset + ts * m2 * sizeof(double),
                                      m2 * sizeof(double), cudaMemcpyDeviceToHost));
            for (std::size_t c = 0; c < 3; ++c) {
                MKO_CUDA_CHECK(
                    cudaMemcpy(r.component[c].data(),
                               base + layout.component.offset + (ts * 3 + c) * m2 * sizeof(double),
                               m2 * sizeof(double), cudaMemcpyDeviceToHost));
            }
            const double* mt = mtf_stage.data() + ts * 2 * per;
            std::copy(mt, mt + per, r.mtf_sagittal.begin());
            std::copy(mt + per, mt + 2 * per, r.mtf_tangential.begin());
            finish_vector_psf(r, pupils[q]);
        }
        ++s.info.tranches;
    }
    s.info.tranche = tranche;
    s.info.work_bytes = s.work_bytes();
    s.info.device_bytes = s.capacity;
    s.info.budget_bytes = usable;
    return out;
}

} // namespace mko::cuda
