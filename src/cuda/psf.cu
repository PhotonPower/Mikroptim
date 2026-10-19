// Skalare PSF, MTF und Strehl auf der GPU (M8-T2): PsfEngine mit cuFFT (Z2Z, FP64). Pro Tranche
// von T Pupillen: Pupillenfunktion (Kernel), FFT vorwärts (ein Plan mit Batch T), Intensität
// und zentrierte PSF (Kernel), FFT rückwärts, MTF-Schnitte (Kernel), Download; Kennzahlen auf
// dem Host (finish_psf). Alle Kernel sind elementweise, ohne Reduktionen. Ablauf:
// docs/architektur.md, Abschnitt "PSF, MTF, Strehl".

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/psf.hpp"
#include "mikroptim/eval/psf_ops.hpp"

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

constexpr int kPsfThreads = 256;

__device__ inline double quiet_nan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

struct PsfKernelArgs {
    int n;                        // Rasterkante N
    std::int64_t m;               // FFT-Größe M
    int slots;                    // T (Batch des Plans)
    int pupils;                   // gültige Pupillen dieser Tranche (<= T)
    const double* opd;            // [T·N²]
    const PsfPupilParams* params; // [T]
    cufftDoubleComplex* spectrum; // [T·M²]
    double* psf;                  // [T·M²]
    double* mtf;                  // [T·2(N + 1)]
};

unsigned grid_for(std::int64_t total) {
    const std::int64_t blocks = (total + kPsfThreads - 1) / kPsfThreads;
    return static_cast<unsigned>(
        std::min<std::int64_t>(std::max<std::int64_t>(blocks, 1), 1 << 20));
}

// Pupillenfunktion in die Ecke (0, 0) jedes M×M-Feldes, sonst 0 (auch für leere Plätze t >=
// pupils).
__global__ void __launch_bounds__(kPsfThreads) psf_pupil_kernel(const PsfKernelArgs a) {
    const std::int64_t m2 = a.m * a.m;
    const std::int64_t total = static_cast<std::int64_t>(a.slots) * m2;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const std::int64_t t = idx / m2;
        const std::int64_t r = idx - t * m2;
        const std::int64_t j = r / a.m;
        const std::int64_t i = r - j * a.m;
        double re = 0.0;
        double im = 0.0;
        if (t < a.pupils && i < a.n && j < a.n) {
            const std::int64_t n = a.n;
            psf_pupil_value(a.opd[t * n * n + j * n + i], a.params[t].lambda_mm, re, im);
        }
        a.spectrum[idx] = make_cuDoubleComplex(re, im);
    }
}

// Intensität I = |U|² / n_voll² an die zentrierte (ggf. gespiegelte) Stelle der PSF und reell in
// das Spektrum (Eingabe der FFT⁻¹).
__global__ void __launch_bounds__(kPsfThreads) psf_intensity_kernel(const PsfKernelArgs a) {
    const std::int64_t m2 = a.m * a.m;
    const std::int64_t total = static_cast<std::int64_t>(a.slots) * m2;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const std::int64_t t = idx / m2;
        if (t >= a.pupils) {
            a.spectrum[idx] = make_cuDoubleComplex(0.0, 0.0);
            continue;
        }
        const std::int64_t r = idx - t * m2;
        const std::int64_t qj = r / a.m;
        const std::int64_t qi = r - qj * a.m;
        const PsfPupilParams p = a.params[t];
        const cufftDoubleComplex u = a.spectrum[idx];
        const double intensity = psf_norm2(cuCreal(u), cuCimag(u)) * p.scale;
        const std::int64_t oi = psf_output_index(qi, a.m, p.sign_x);
        const std::int64_t oj = psf_output_index(qj, a.m, p.sign_y);
        a.psf[t * m2 + oj * a.m + oi] = intensity;
        a.spectrum[idx] = make_cuDoubleComplex(intensity, 0.0);
    }
}

// MTF-Schnitte aus der OTF (FFT⁻¹ der Intensität): sagittal (k, 0), tangential (0, k), k = 0 … N,
// normiert auf OTF(0).
__global__ void __launch_bounds__(kPsfThreads) psf_mtf_kernel(const PsfKernelArgs a) {
    const std::int64_t per = static_cast<std::int64_t>(a.n) + 1;
    const std::int64_t total = static_cast<std::int64_t>(a.pupils) * per;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const std::int64_t t = idx / per;
        const std::int64_t k = idx - t * per;
        const cufftDoubleComplex* otf = a.spectrum + t * a.m * a.m;
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

} // namespace

// ---------------------------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------------------------

struct PsfEngine::Impl {
    int device = 0;
    EngineOptions options{};
    void* buffer = nullptr;
    std::size_t capacity = 0;
    cufftHandle plan = 0;
    bool has_plan = false;
    long long plan_m = 0;
    long long plan_batch = 0;
    std::size_t work_bytes = 0;
    PsfRunInfo info{};

    ~Impl() { release(); }

    void destroy_plan() noexcept {
        if (has_plan) {
            cufftDestroy(plan); // Fehler beim Freigeben sind nicht behebbar
            has_plan = false;
            plan_m = 0;
            plan_batch = 0;
            work_bytes = 0;
        }
    }

    void release() noexcept {
        try {
            const detail::DeviceGuard guard(device);
            destroy_plan();
            if (buffer != nullptr) {
                cudaFree(buffer);
            }
        } catch (...) {
            destroy_plan();
            if (buffer != nullptr) {
                cudaFree(buffer);
            }
        }
        buffer = nullptr;
        capacity = 0;
    }

    // Arbeitsspeicher eines Plans M×M mit Batch t (Abschätzung vor dem Anlegen) [Byte].
    static std::size_t estimate_work(long long m, long long t) {
        cufftHandle h = 0;
        MKO_CUFFT_CHECK(cufftCreate(&h));
        long long n[2] = {m, m};
        std::size_t ws = 0;
        const cufftResult r =
            cufftGetSizeMany64(h, 2, n, nullptr, 1, m * m, nullptr, 1, m * m, CUFFT_Z2Z, t, &ws);
        cufftDestroy(h);
        if (r != CUFFT_SUCCESS) {
            detail::throw_cufft_error(r, "cufftGetSizeMany64", __FILE__, __LINE__);
        }
        return ws;
    }

    // Plan M×M mit Batch t ohne automatischen Arbeitsspeicher (wird im Puffer bereitgestellt).
    void make_plan(long long m, long long t) {
        if (has_plan && plan_m == m && plan_batch == t) {
            return;
        }
        destroy_plan();
        MKO_CUFFT_CHECK(cufftCreate(&plan));
        has_plan = true;
        MKO_CUFFT_CHECK(cufftSetAutoAllocation(plan, 0));
        long long n[2] = {m, m};
        std::size_t ws = 0;
        MKO_CUFFT_CHECK(cufftMakePlanMany64(plan, 2, n, nullptr, 1, m * m, nullptr, 1, m * m,
                                            CUFFT_Z2Z, t, &ws));
        plan_m = m;
        plan_batch = t;
        work_bytes = ws;
    }

    std::size_t available_bytes() const {
        std::size_t available = device_memory_info(device).free_bytes + capacity;
        if (options.memory_limit_bytes) {
            available = std::min(available, *options.memory_limit_bytes);
        }
        return available;
    }
};

// ---------------------------------------------------------------------------------------------
// PsfEngine
// ---------------------------------------------------------------------------------------------

PsfEngine::PsfEngine(int device, const EngineOptions& options) : impl_(std::make_unique<Impl>()) {
    const int count = device_count();
    if (count == 0) {
        throw CudaError("PsfEngine: kein CUDA-Gerät gefunden (Treiber installiert?)");
    }
    if (device < 0 || device >= count) {
        throw std::invalid_argument("PsfEngine: Gerät " + std::to_string(device) +
                                    " existiert nicht (vorhanden: 0 … " +
                                    std::to_string(count - 1) + ")");
    }
    (void)usable_bytes(0, options.reserve); // prüft die Reserve
    impl_->device = device;
    impl_->options = options;
}

PsfEngine::~PsfEngine() = default;
PsfEngine::PsfEngine(PsfEngine&&) noexcept = default;
PsfEngine& PsfEngine::operator=(PsfEngine&&) noexcept = default;

const PsfRunInfo& PsfEngine::last_run() const {
    return impl_->info;
}

int PsfEngine::device() const {
    return impl_->device;
}

const EngineOptions& PsfEngine::options() const {
    return impl_->options;
}

std::size_t PsfEngine::device_bytes() const {
    return impl_ ? impl_->capacity : 0;
}

void PsfEngine::release() noexcept {
    if (impl_) {
        impl_->release();
    }
}

std::vector<PsfResult> PsfEngine::compute(const SystemBatchHost& batch, const EvalResult& result,
                                          int v, const PsfSettings& settings) {
    const std::vector<PsfPupil> pupils = make_psf_pupils(batch, result, v, settings);
    return compute(pupils, psf_fft_size(result.grid.size, settings.padding));
}

std::vector<PsfResult> PsfEngine::compute(std::span<const PsfPupil> pupils, int fft_size) {
    Impl& s = *impl_;
    // ---- Prüfen und Host-Ergebnisse anlegen (vor jeder Gerätearbeit) ----
    std::vector<PsfResult> out;
    out.reserve(pupils.size());
    for (const PsfPupil& p : pupils) {
        if (p.grid != pupils.front().grid) {
            throw std::invalid_argument("PsfEngine: alle Pupillen brauchen dieselbe Rasterkante");
        }
        out.push_back(prepare_psf_result(p, fft_size));
    }
    s.info = PsfRunInfo{};
    s.info.pupils = static_cast<int>(pupils.size());
    if (pupils.empty()) {
        s.info.device_bytes = s.capacity;
        return out;
    }
    const int n = pupils.front().grid;
    const long long m = fft_size;
    const int count = static_cast<int>(pupils.size());

    const detail::DeviceGuard guard(s.device);

    // ---- Tranche T: größte Zahl Pupillen mit Puffern + cuFFT-Arbeitsspeicher <= nutzbar ----
    const std::size_t usable = usable_bytes(s.available_bytes(), s.options.reserve);
    const std::size_t per_pupil =
        make_psf_device_layout(n, fft_size, 1).total_bytes + align_bytes(Impl::estimate_work(m, 1));
    int tranche = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(count), usable / std::max<std::size_t>(per_pupil, 1)));
    std::size_t need = 0;
    while (tranche >= 1) {
        // Ein schon passender Plan bleibt erhalten; sonst neu (Arbeitsspeicher exakt).
        s.make_plan(m, tranche);
        need = make_psf_device_layout(n, fft_size, tranche).total_bytes + align_bytes(s.work_bytes);
        if (need <= usable) {
            break;
        }
        tranche = tranche > 1 ? std::max(1, tranche * 3 / 4) : 0;
    }
    if (tranche < 1) {
        s.destroy_plan();
        throw CudaError("PsfEngine: nicht eine Pupille passt in den Gerätespeicher (Gerät " +
                        std::to_string(s.device) + ", nach Reserve " + mib(usable) + ", benötigt " +
                        mib(per_pupil) + " für M = " + std::to_string(m) +
                        "); Raster oder Zero-Padding verkleinern");
    }
    const PsfDeviceLayout layout = make_psf_device_layout(n, fft_size, tranche);
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
    if (s.work_bytes > 0) {
        MKO_CUFFT_CHECK(cufftSetWorkArea(s.plan, base + layout.total_bytes));
    }

    PsfKernelArgs args{};
    args.n = n;
    args.m = m;
    args.slots = tranche;
    args.opd = reinterpret_cast<const double*>(base + layout.opd.offset);
    args.params = reinterpret_cast<const PsfPupilParams*>(base + layout.params.offset);
    args.spectrum = reinterpret_cast<cufftDoubleComplex*>(base + layout.spectrum.offset);
    args.psf = reinterpret_cast<double*>(base + layout.psf.offset);
    args.mtf = reinterpret_cast<double*>(base + layout.mtf.offset);
    const std::int64_t elements = static_cast<std::int64_t>(tranche) * m * m;
    const auto n2 = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const auto m2 = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
    const std::size_t per = static_cast<std::size_t>(n) + 1;

    std::vector<double> opd_stage(static_cast<std::size_t>(tranche) * n2);
    std::vector<PsfPupilParams> params(static_cast<std::size_t>(tranche));
    std::vector<double> mtf_stage(static_cast<std::size_t>(tranche) * 2 * per);

    for (int first = 0; first < count; first += tranche) {
        const int used = std::min(tranche, count - first);
        for (int t = 0; t < used; ++t) {
            const PsfPupil& p = pupils[static_cast<std::size_t>(first + t)];
            std::copy(p.opd.begin(), p.opd.end(),
                      opd_stage.begin() +
                          static_cast<std::ptrdiff_t>(static_cast<std::size_t>(t) * n2));
            const double full = static_cast<double>(p.full_count);
            params[static_cast<std::size_t>(t)] = PsfPupilParams{
                p.wavelength_mm(), 1.0 / (full * full), p.flip_x ? -1 : 1, p.flip_y ? -1 : 1};
        }
        MKO_CUDA_CHECK(cudaMemcpy(base + layout.opd.offset, opd_stage.data(),
                                  static_cast<std::size_t>(used) * n2 * sizeof(double),
                                  cudaMemcpyHostToDevice));
        MKO_CUDA_CHECK(cudaMemcpy(base + layout.params.offset, params.data(),
                                  static_cast<std::size_t>(used) * sizeof(PsfPupilParams),
                                  cudaMemcpyHostToDevice));
        args.pupils = used;

        psf_pupil_kernel<<<grid_for(elements), kPsfThreads>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUFFT_CHECK(cufftExecZ2Z(s.plan, args.spectrum, args.spectrum, CUFFT_FORWARD));
        psf_intensity_kernel<<<grid_for(elements), kPsfThreads>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUFFT_CHECK(cufftExecZ2Z(s.plan, args.spectrum, args.spectrum, CUFFT_INVERSE));
        psf_mtf_kernel<<<grid_for(static_cast<std::int64_t>(used) * static_cast<std::int64_t>(per)),
                         kPsfThreads>>>(args);
        MKO_CUDA_CHECK(cudaGetLastError());
        MKO_CUDA_CHECK(cudaDeviceSynchronize());

        MKO_CUDA_CHECK(cudaMemcpy(mtf_stage.data(), base + layout.mtf.offset,
                                  static_cast<std::size_t>(used) * 2 * per * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        for (int t = 0; t < used; ++t) {
            PsfResult& r = out[static_cast<std::size_t>(first + t)];
            MKO_CUDA_CHECK(cudaMemcpy(r.psf.data(),
                                      base + layout.psf.offset +
                                          static_cast<std::size_t>(t) * m2 * sizeof(double),
                                      m2 * sizeof(double), cudaMemcpyDeviceToHost));
            const double* mt = mtf_stage.data() + static_cast<std::size_t>(t) * 2 * per;
            std::copy(mt, mt + per, r.mtf_sagittal.begin());
            std::copy(mt + per, mt + 2 * per, r.mtf_tangential.begin());
            finish_psf(r, pupils[static_cast<std::size_t>(first + t)]);
        }
        ++s.info.tranches;
    }
    s.info.tranche = tranche;
    s.info.work_bytes = s.work_bytes;
    s.info.device_bytes = s.capacity;
    s.info.budget_bytes = usable;
    return out;
}

} // namespace mko::cuda
