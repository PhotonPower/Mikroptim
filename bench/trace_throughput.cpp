// Benchmark trace_throughput (M2-T6): Durchsatz der Trace-Kernel über die öffentliche
// TraceEngine für synthetische Systeme mit S Flächen und V Varianten. Messgrößen, Optionen und
// Ausgabeformate: bench/README.md, Ergebnisse: docs/benchmarks.md.
//
// Zwei Messungen pro Fall (Zahlentyp, Modus, S, V):
//   - Engine: TraceEngine::run (Prüfung der Werte, Upload, Kernel, Download), synchron;
//   - Kernel: DeviceResiduals::run bzw. DeviceMerit::run auf einem schon hochgeladenen
//     DeviceSystemBatch (Kernelstart(s) mit Feldablage und Synchronisation, ohne Upload des
//     Batches und ohne Download), gemessen mit der Host-Uhr um den synchronen Aufruf.
// Jeweils Median und Minimum aus --repeat Läufen nach --warmup Aufwärmläufen.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_table.hpp"
#include "mikroptim/version.hpp"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using mko::cuda::TraceMode;

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
constexpr int kExitNoGpu = 3; // wie `mikroptim trace --gpu` (SKIP_RETURN_CODE der Smoke-Tests)

constexpr const char* kJsonFormat = "mikroptim.bench.trace_throughput/1";

// Rechentyp der Kernel (TraceSettings::precision, M7-T1). Alle Ausgaben führen ihn als Spalte.
enum class NumberType {
    Fp64,
    Df64,
};

const char* to_string(NumberType type) {
    switch (type) {
    case NumberType::Fp64:
        return "fp64";
    case NumberType::Df64:
        return "df64";
    }
    return "?";
}

mko::cuda::Precision precision_of(NumberType type) {
    return type == NumberType::Df64 ? mko::cuda::Precision::Df64 : mko::cuda::Precision::Fp64;
}

// Stellt den Zahlentyp für einen Lauf ein.
void configure_number_type(NumberType type, mko::cuda::TraceSettings& settings) {
    settings.precision = precision_of(type);
}

// Ein zu messender Modus: Kernel-Modus der Engine, optional mit Max-Reduktionen.
struct ModeSpec {
    TraceMode mode = TraceMode::Merit;
    bool extrema = false;
};

std::string mode_name(const ModeSpec& m) {
    std::string name = m.mode == TraceMode::Merit ? "merit" : "residuals";
    if (m.extrema) {
        name += "+extrema";
    }
    return name;
}

class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Options {
    bool help = false;
    int device = 0;
    std::vector<int> surfaces{10, 20, 30, 40};
    std::vector<int> variants{1, 10, 100, 1000, 10000};
    std::vector<NumberType> types{NumberType::Fp64};
    std::vector<ModeSpec> modes{{TraceMode::Residuals, false}, {TraceMode::Merit, false}};
    int fields = 3;
    int wavelengths = 3;
    int rings = 10;
    int spokes = 8;
    int repeat = 10;
    int warmup = 2;
    bool kernel = true;
    std::uint64_t seed = 20261018;
    std::string csv;  // Dateiname, "-" = Standardausgabe, leer = keine CSV-Ausgabe
    std::string json; // dito
    bool quiet = false;
};

void print_usage() {
    std::printf(
        "Aufruf: trace_throughput [Optionen]\n"
        "\n"
        "Misst den Durchsatz der Trace-Kernel über die TraceEngine (M2-T6) für synthetische\n"
        "Systeme aus schwachen Menisken (S Flächen) mit V leicht gestörten Varianten.\n"
        "\n"
        "Optionen (Listen durch Komma getrennt):\n"
        "  --device N          CUDA-Gerät (Standard 0)\n"
        "  --surfaces LISTE    Flächenzahlen S, je 4 ... %d (Standard 10,20,30,40)\n"
        "  --variants LISTE    Variantenzahlen V >= 1 (Standard 1,10,100,1000,10000)\n"
        "  --mode LISTE        residuals, merit, residuals+extrema, merit+extrema, all\n"
        "                      (Standard residuals,merit; all = alle vier)\n"
        "  --type LISTE        Zahlentyp: fp64 (Standard), df64\n"
        "  --fields F          Felder, Feldwinkel 0 ... 3 Grad (Standard 3)\n"
        "  --wavelengths L     Wellenlängen 486,13 ... 656,27 nm (Standard 3)\n"
        "  --rings N           Ringe der Pupillenquadratur (Standard 10)\n"
        "  --spokes M          Speichen der Pupillenquadratur (Standard 8)\n"
        "  --repeat R          gemessene Läufe pro Fall, Median und Minimum (Standard 10)\n"
        "  --warmup W          Aufwärmläufe pro Fall (Standard 2)\n"
        "  --no-kernel         nur über die Engine messen (ohne Kernelzeit)\n"
        "  --seed N            Seed der Störungen (Standard 20261018)\n"
        "  --csv DATEI         Ergebnisse zusätzlich als CSV (\"-\" = Standardausgabe)\n"
        "  --json DATEI        Ergebnisse zusätzlich als JSON (\"-\" = Standardausgabe)\n"
        "  --quiet             keine Tabelle (nur CSV/JSON)\n"
        "  --help              diese Hilfe\n"
        "\n"
        "Exit-Code: 0 Erfolg, 1 Fehler, 2 Aufruffehler, 3 kein CUDA-Gerät.\n",
        mko::kMaxSurfaces);
}

template <class T>
T parse_number(std::string_view option, std::string_view text) {
    T value{};
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    if (text.empty() || ec != std::errc{} || ptr != last) {
        throw UsageError(std::string(option) + " erwartet eine ganze Zahl, nicht \"" +
                         std::string(text) + "\"");
    }
    return value;
}

std::vector<std::string_view> split_list(std::string_view text) {
    std::vector<std::string_view> items;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        items.push_back(text.substr(start, end - start));
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return items;
}

std::vector<int> parse_int_list(std::string_view option, std::string_view text, int min_value,
                                int max_value) {
    std::vector<int> values;
    for (const std::string_view item : split_list(text)) {
        const int v = parse_number<int>(option, item);
        if (v < min_value || v > max_value) {
            throw UsageError(std::string(option) + ": " + std::to_string(v) + " liegt nicht in [" +
                             std::to_string(min_value) + ", " + std::to_string(max_value) + "]");
        }
        values.push_back(v);
    }
    return values;
}

std::vector<ModeSpec> parse_modes(std::string_view text) {
    std::vector<ModeSpec> modes;
    for (const std::string_view item : split_list(text)) {
        if (item == "residuals") {
            modes.push_back({TraceMode::Residuals, false});
        } else if (item == "merit") {
            modes.push_back({TraceMode::Merit, false});
        } else if (item == "residuals+extrema") {
            modes.push_back({TraceMode::Residuals, true});
        } else if (item == "merit+extrema") {
            modes.push_back({TraceMode::Merit, true});
        } else if (item == "all") {
            modes.push_back({TraceMode::Residuals, false});
            modes.push_back({TraceMode::Merit, false});
            modes.push_back({TraceMode::Residuals, true});
            modes.push_back({TraceMode::Merit, true});
        } else {
            throw UsageError("--mode: unbekannter Modus \"" + std::string(item) + "\"");
        }
    }
    return modes;
}

std::vector<NumberType> parse_types(std::string_view text) {
    std::vector<NumberType> types;
    for (const std::string_view item : split_list(text)) {
        if (item == "fp64") {
            types.push_back(NumberType::Fp64);
        } else if (item == "df64") {
            types.push_back(NumberType::Df64);
        } else {
            throw UsageError("--type: unbekannter Zahlentyp \"" + std::string(item) +
                             "\" (verfügbar: fp64, df64)");
        }
    }
    return types;
}

Options parse_args(int argc, char** argv) {
    Options o;
    constexpr int kIntMax = std::numeric_limits<int>::max();
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                throw UsageError(std::string(arg) + " erwartet einen Wert");
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            o.help = true;
        } else if (arg == "--device") {
            o.device = parse_number<int>(arg, value());
            if (o.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
        } else if (arg == "--surfaces") {
            o.surfaces = parse_int_list(arg, value(), 4, mko::kMaxSurfaces);
        } else if (arg == "--variants") {
            o.variants = parse_int_list(arg, value(), 1, kIntMax);
        } else if (arg == "--mode") {
            o.modes = parse_modes(value());
        } else if (arg == "--type") {
            o.types = parse_types(value());
        } else if (arg == "--fields") {
            o.fields = parse_number<int>(arg, value());
        } else if (arg == "--wavelengths") {
            o.wavelengths = parse_number<int>(arg, value());
        } else if (arg == "--rings") {
            o.rings = parse_number<int>(arg, value());
        } else if (arg == "--spokes") {
            o.spokes = parse_number<int>(arg, value());
        } else if (arg == "--repeat") {
            o.repeat = parse_number<int>(arg, value());
        } else if (arg == "--warmup") {
            o.warmup = parse_number<int>(arg, value());
        } else if (arg == "--no-kernel") {
            o.kernel = false;
        } else if (arg == "--seed") {
            o.seed = parse_number<std::uint64_t>(arg, value());
        } else if (arg == "--csv") {
            o.csv = std::string(value());
        } else if (arg == "--json") {
            o.json = std::string(value());
        } else if (arg == "--quiet") {
            o.quiet = true;
        } else {
            throw UsageError("unbekannte Option " + std::string(arg));
        }
    }
    if (o.fields < 1 || o.wavelengths < 1) {
        throw UsageError("--fields und --wavelengths müssen >= 1 sein");
    }
    if (o.rings < 1 || o.rings > mko::kMaxPupilRings || o.spokes < 1) {
        throw UsageError("--rings muss in [1, " + std::to_string(mko::kMaxPupilRings) +
                         "] liegen, --spokes >= 1");
    }
    if (o.repeat < 1 || o.warmup < 0) {
        throw UsageError("--repeat muss >= 1, --warmup >= 0 sein");
    }
    if (o.surfaces.empty() || o.variants.empty() || o.modes.empty() || o.types.empty()) {
        throw UsageError("leere Liste");
    }
    return o;
}

// ---- Synthetisches System -----------------------------------------------------------------

mko::MaterialSpec buchdahl(double n0, double nu1, double nu2) {
    mko::MaterialSpec m;
    m.kind = mko::MaterialKind::Buchdahl;
    m.n0 = n0;
    m.nu = {nu1, nu2, 0.0};
    return m;
}

mko::Surface sphere(double curvature, double thickness, const mko::MaterialSpec& material) {
    mko::Surface s;
    s.type = mko::SurfaceType::Sphere;
    s.curvature = curvature;
    s.thickness = thickness;
    s.material = material;
    s.semi_diameter = 6.0;
    return s;
}

// Frei erfundenes, physikalisch sinnvolles System mit genau S Flächen (kein reales Objektiv):
// Blende (Radius 2 mm), dann eine Folge schwacher Menisken aus einem Kron- und einem
// Flintglas (Buchdahl-Modelle, abwechselnd, Biegung ±1,5) und, wenn S − 1 ungerade ist, ein
// verkittetes Achromat-ähnliches Glied am Ende. Die dünnen Brechkräfte summieren sich zu
// 1/40 mm⁻¹; die Bildebene liegt im paraxialen Fokus bei λ_ref = 587,56 nm. Felder 0 … 3°,
// Wellenlängen 486,13 … 656,27 nm.
mko::LoadedSystem make_bench_system(int S, int F, int L) {
    const mko::MaterialSpec crown = buchdahl(1.52, -0.08, 0.01);
    const mko::MaterialSpec flint = buchdahl(1.62, -0.12, 0.02);
    const mko::MaterialSpec air{};
    constexpr double kTotalPower = 1.0 / 40.0; // Summe der dünnen Brechkräfte [1/mm]
    constexpr double kGlass = 1.0;             // Mittendicke der Menisken [mm]
    constexpr double kAir = 0.3;               // Luftabstand [mm]
    constexpr double kShape = 1.5;             // Betrag der Biegung X = (c1 + c2)/(c1 − c2)

    const int after_stop = S - 1;
    const bool doublet = after_stop % 2 == 1;
    const int singlets = doublet ? (after_stop - 3) / 2 : after_stop / 2;
    const double phi = kTotalPower / static_cast<double>(singlets + (doublet ? 1 : 0));

    mko::SystemDescription sys;
    sys.name = "Benchmark S=" + std::to_string(S);
    sys.description = "Synthetisches Benchmark-System (M2-T6), frei erfunden";
    sys.spectrum.reference_nm = 587.56;
    for (int l = 0; l < L; ++l) {
        const double t = L == 1 ? 0.0 : static_cast<double>(l) / static_cast<double>(L - 1);
        sys.spectrum.wavelengths_nm.push_back(L == 1 ? 587.56 : 486.13 + t * (656.27 - 486.13));
        sys.spectrum.weights.push_back(1.0);
    }
    sys.source.type = mko::SourceType::Collimated;
    for (int f = 0; f < F; ++f) {
        const double t = F == 1 ? 0.0 : static_cast<double>(f) / static_cast<double>(F - 1);
        sys.source.field_angles_deg.push_back(3.0 * t);
        sys.source.field_weights.push_back(1.0);
    }

    mko::Surface stop;
    stop.type = mko::SurfaceType::Stop;
    stop.thickness = 1.0;
    stop.material = air;
    stop.semi_diameter = 2.0;
    sys.surfaces.push_back(stop);
    for (int i = 0; i < singlets; ++i) {
        const bool use_crown = i % 2 == 0;
        const mko::MaterialSpec& glass = use_crown ? crown : flint;
        const double d = phi / (glass.n0 - 1.0); // c1 − c2
        const double x = i % 4 < 2 ? kShape : -kShape;
        sys.surfaces.push_back(sphere(0.5 * d * (x + 1.0), kGlass, glass));
        sys.surfaces.push_back(sphere(0.5 * d * (x - 1.0), kAir, air));
    }
    if (doublet) {
        // Kron (Brechkraft 1,6 φ, bikonvex) verkittet mit Flint (−0,6 φ)
        const double d1 = 1.6 * phi / (crown.n0 - 1.0);
        const double c1 = 0.5 * d1;
        const double c2 = -0.5 * d1;
        const double c3 = c2 + 0.6 * phi / (flint.n0 - 1.0);
        sys.surfaces.push_back(sphere(c1, 1.2, crown));
        sys.surfaces.push_back(sphere(c2, 0.6, flint));
        sys.surfaces.push_back(sphere(c3, kAir, air));
    }
    if (static_cast<int>(sys.surfaces.size()) != S) {
        throw std::logic_error("make_bench_system: falsche Flächenzahl");
    }

    // Bildebene in den paraxialen Fokus legen
    mko::LoadedSystem loaded = mko::prepare_system(sys, {});
    const mko::ParaxialData p = mko::compute_paraxial(loaded.system, loaded.catalog);
    if (p.afocal || !(p.back_focal_distance > 1.0)) {
        throw std::logic_error("make_bench_system: ungeeignete Schnittweite");
    }
    sys.surfaces.back().thickness = p.back_focal_distance;
    return mko::prepare_system(sys, {});
}

// V Varianten; Variante 0 ungestört, die übrigen mit festem Seed leicht gestört (Krümmungen
// ±2 %, Dicken ±2 %, Arbeitsabstand ±1 %, Brechzahlen der Gläser ±0,002, Pupillenradius ±5 %).
mko::SystemBatchHost make_bench_batch(const mko::LoadedSystem& system, int variants,
                                      std::uint64_t seed) {
    mko::SystemBatchHost b = mko::make_system_batch(system, variants);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int S = b.surfaces();
    for (int v = 1; v < variants; ++v) {
        for (int s = 1; s < S; ++s) {
            b.curvature(v, s) *= 1.0 + 0.02 * u(rng);
        }
        for (int s = 0; s < S - 1; ++s) {
            b.thickness(v, s) *= 1.0 + 0.02 * u(rng);
        }
        b.image_distance(v) *= 1.0 + 0.01 * u(rng);
        for (int s = 0; s < S; ++s) {
            if (b.index(v, s, 0) > 1.0) {
                const double dn = 0.002 * u(rng);
                for (int l = 0; l < b.wavelengths(); ++l) {
                    b.index(v, s, l) += dn;
                }
            }
        }
        b.pupil_radius(v) *= 1.0 + 0.05 * u(rng);
    }
    return b;
}

// ---- Messung ------------------------------------------------------------------------------

struct Timing {
    double median_s = 0.0;
    double min_s = 0.0;
};

template <class Fn>
Timing measure(int warmup, int repeat, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto stop = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double>(stop - start).count());
    }
    std::sort(t.begin(), t.end());
    const std::size_t n = t.size();
    Timing r;
    r.median_s = n % 2 == 1 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    r.min_s = t.front();
    return r;
}

struct Result {
    NumberType type = NumberType::Fp64;
    ModeSpec mode;
    int surfaces = 0;
    int variants = 0;
    std::int64_t rays_per_variant = 0;
    std::int64_t rays = 0; // V · rays_per_variant
    std::int64_t failed = 0;
    int tranches = 0;
    Timing engine;
    std::optional<Timing> kernel; // leer: nicht gemessen (--no-kernel) oder kein Speicher
};

double rays_per_s(const Result& r, double seconds) {
    return static_cast<double>(r.rays) / seconds;
}

Result run_case(mko::cuda::TraceEngine& engine, const Options& o, NumberType type,
                const ModeSpec& mode, const mko::SystemBatchHost& batch) {
    mko::cuda::TraceSettings settings;
    settings.quadrature = mko::ResidualSettings{o.rings, o.spokes};
    settings.status = true;
    settings.extrema = mode.extrema;
    configure_number_type(type, settings);

    Result r;
    r.type = type;
    r.mode = mode;
    r.surfaces = batch.surfaces();
    r.variants = batch.variants();

    mko::cuda::TraceOutput out;
    r.engine = measure(o.warmup, o.repeat, [&] { engine.run(batch, mode.mode, settings, out); });
    r.rays_per_variant = out.layout.rays_per_variant();
    r.rays = static_cast<std::int64_t>(r.variants) * r.rays_per_variant;
    r.tranches = out.info.tranches;
    if (mode.mode == TraceMode::Merit) {
        for (const std::int64_t f : out.merit.failed) {
            r.failed += f;
        }
    } else {
        r.failed = static_cast<std::int64_t>(out.residuals.failed_count());
    }

    if (o.kernel) {
        // Kernel allein über die Klassen, die die Engine kapselt: Batch einmal hochladen, dann
        // nur den synchronen Kernellauf messen. Passt der Batch nicht in einem Stück auf das
        // Gerät, entfällt die Messung (die Engine teilt dann in Tranchen).
        try {
            engine.release(); // Speicher der Engine für diese Messung freigeben
            const mko::cuda::DeviceSystemBatch device_batch(batch, engine.device());
            const mko::ResidualLayout layout(batch.topology(), settings.quadrature);
            if (mode.mode == TraceMode::Merit) {
                mko::cuda::DeviceMerit merit;
                mko::cuda::MeritRunOptions opt;
                opt.extrema = mode.extrema;
                opt.precision = settings.precision;
                r.kernel =
                    measure(o.warmup, o.repeat, [&] { merit.run(device_batch, layout, opt); });
            } else {
                mko::cuda::DeviceResiduals residuals;
                mko::cuda::ResidualRunOptions opt;
                opt.extrema = mode.extrema;
                opt.precision = settings.precision;
                r.kernel =
                    measure(o.warmup, o.repeat, [&] { residuals.run(device_batch, layout, opt); });
            }
        } catch (const mko::cuda::CudaError& e) {
            std::fprintf(stderr, "Hinweis: Kernelzeit für S = %d, V = %d nicht gemessen: %s\n",
                         r.surfaces, r.variants, e.what());
        }
    }
    return r;
}

// ---- Geräteinformation --------------------------------------------------------------------

struct DeviceReport {
    int index = 0;
    std::string name;
    int major = 0;
    int minor = 0;
    int sms = 0;
    int fp32_fp64_ratio = 0;
    double memory_gib = 0.0;
    std::string driver;  // CUDA-Version, die der Treiber unterstützt (cudaDriverGetVersion)
    std::string runtime; // CUDA-Laufzeit des Programms (cudaRuntimeGetVersion)
};

std::string cuda_version_string(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw mko::cuda::CudaError(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

DeviceReport device_report(int device) {
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties");
    int driver = 0;
    int runtime = 0;
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    DeviceReport d;
    d.index = device;
    d.name = p.name;
    d.major = p.major;
    d.minor = p.minor;
    d.sms = p.multiProcessorCount;
    d.fp32_fp64_ratio = p.singleToDoublePrecisionPerfRatio;
    d.memory_gib = static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    d.driver = cuda_version_string(driver);
    d.runtime = cuda_version_string(runtime);
    return d;
}

#ifdef NDEBUG
constexpr const char* kBuildType = "Release";
#else
constexpr const char* kBuildType = "Debug";
#endif

// ---- Ausgabe ------------------------------------------------------------------------------

void print_table_header() {
    std::printf("%-5s %-17s %3s %6s %6s %10s %10s %10s %10s %10s %10s %10s %8s %3s\n", "Typ",
                "Modus", "S", "V", "Str/V", "Engine ms", "Str/s", "StrFl/s", "Kernel ms", "Str/s",
                "StrFl/s", "Engine min", "versagt", "Tr");
}

void print_table_rows(const std::vector<Result>& results) {
    for (const Result& r : results) {
        const double er = rays_per_s(r, r.engine.median_s);
        char kernel_ms[32] = "-";
        char kernel_r[32] = "-";
        char kernel_rs[32] = "-";
        if (r.kernel) {
            const double kr = rays_per_s(r, r.kernel->median_s);
            std::snprintf(kernel_ms, sizeof kernel_ms, "%.4f", 1e3 * r.kernel->median_s);
            std::snprintf(kernel_r, sizeof kernel_r, "%.3e", kr);
            std::snprintf(kernel_rs, sizeof kernel_rs, "%.3e", kr * r.surfaces);
        }
        std::printf("%-5s %-17s %3d %6d %6lld %10.4f %10.3e %10.3e %10s %10s %10s %10.4f %7.3f%% "
                    "%3d\n",
                    to_string(r.type), mode_name(r.mode).c_str(), r.surfaces, r.variants,
                    static_cast<long long>(r.rays_per_variant), 1e3 * r.engine.median_s, er,
                    er * r.surfaces, kernel_ms, kernel_r, kernel_rs, 1e3 * r.engine.min_s,
                    100.0 * static_cast<double>(r.failed) / static_cast<double>(r.rays),
                    r.tranches);
    }
}

void write_csv(std::ostream& os, const DeviceReport& d, const std::vector<Result>& results) {
    os << "gpu,type,mode,surfaces,variants,rays_per_variant,rays,engine_median_s,engine_min_s,"
          "engine_rays_per_s,engine_ray_surfaces_per_s,kernel_median_s,kernel_min_s,"
          "kernel_rays_per_s,kernel_ray_surfaces_per_s,failed_rays,tranches\n";
    char buf[64];
    auto num = [&](double x) {
        std::snprintf(buf, sizeof buf, "%.6e", x);
        return std::string(buf);
    };
    for (const Result& r : results) {
        const double er = rays_per_s(r, r.engine.median_s);
        os << '"' << d.name << "\"," << to_string(r.type) << ',' << mode_name(r.mode) << ','
           << r.surfaces << ',' << r.variants << ',' << r.rays_per_variant << ',' << r.rays << ','
           << num(r.engine.median_s) << ',' << num(r.engine.min_s) << ',' << num(er) << ','
           << num(er * r.surfaces) << ',';
        if (r.kernel) {
            const double kr = rays_per_s(r, r.kernel->median_s);
            os << num(r.kernel->median_s) << ',' << num(r.kernel->min_s) << ',' << num(kr) << ','
               << num(kr * r.surfaces);
        } else {
            os << ",,,";
        }
        os << ',' << r.failed << ',' << r.tranches << '\n';
    }
}

nlohmann::json to_json(const Options& o, const DeviceReport& d,
                       const std::vector<Result>& results) {
    nlohmann::json j;
    j["format"] = kJsonFormat;
    j["mikroptim_version"] = mko::kVersion;
    j["build_type"] = kBuildType;
    j["device"] = {{"index", d.index},
                   {"name", d.name},
                   {"compute_capability", std::to_string(d.major) + "." + std::to_string(d.minor)},
                   {"multiprocessors", d.sms},
                   {"fp32_fp64_ratio", d.fp32_fp64_ratio},
                   {"memory_gib", d.memory_gib},
                   {"driver_cuda_version", d.driver},
                   {"runtime_cuda_version", d.runtime}};
    j["settings"] = {{"fields", o.fields}, {"wavelengths", o.wavelengths},
                     {"rings", o.rings},   {"spokes", o.spokes},
                     {"repeat", o.repeat}, {"warmup", o.warmup},
                     {"seed", o.seed},     {"block_threads", mko::cuda::kResidualThreads}};
    nlohmann::json list = nlohmann::json::array();
    for (const Result& r : results) {
        const double er = rays_per_s(r, r.engine.median_s);
        nlohmann::json e = {{"type", to_string(r.type)},
                            {"mode", mode_name(r.mode)},
                            {"surfaces", r.surfaces},
                            {"variants", r.variants},
                            {"rays_per_variant", r.rays_per_variant},
                            {"rays", r.rays},
                            {"failed_rays", r.failed},
                            {"tranches", r.tranches},
                            {"engine",
                             {{"median_s", r.engine.median_s},
                              {"min_s", r.engine.min_s},
                              {"rays_per_s", er},
                              {"ray_surfaces_per_s", er * r.surfaces}}}};
        if (r.kernel) {
            const double kr = rays_per_s(r, r.kernel->median_s);
            e["kernel"] = {{"median_s", r.kernel->median_s},
                           {"min_s", r.kernel->min_s},
                           {"rays_per_s", kr},
                           {"ray_surfaces_per_s", kr * r.surfaces}};
        } else {
            e["kernel"] = nullptr;
        }
        list.push_back(std::move(e));
    }
    j["results"] = std::move(list);
    return j;
}

// Schreibt über `write` in die Datei `target` ("-" = Standardausgabe).
template <class Fn>
void emit(const std::string& target, Fn&& write) {
    if (target == "-") {
        write(std::cout);
        std::cout.flush();
        return;
    }
    std::ofstream file(target);
    if (!file) {
        throw std::runtime_error("kann " + target + " nicht schreiben");
    }
    write(file);
}

int run(const Options& o) {
    const int count = mko::cuda::device_count();
    if (count == 0) {
        std::fprintf(stderr, "trace_throughput: kein CUDA-Gerät gefunden\n");
        return kExitNoGpu;
    }
    if (o.device >= count) {
        throw UsageError("--device " + std::to_string(o.device) + ": nur " + std::to_string(count) +
                         " Gerät(e) vorhanden");
    }
    const DeviceReport d = device_report(o.device);
    mko::cuda::TraceEngine engine(o.device);
    const bool table = !o.quiet;
    if (table) {
        std::printf("GPU %d: %s (sm_%d%d, %d SMs, %.1f GiB, FP32:FP64 = %d:1), Treiber-CUDA %s, "
                    "Laufzeit-CUDA %s, Build %s\n",
                    d.index, d.name.c_str(), d.major, d.minor, d.sms, d.memory_gib,
                    d.fp32_fp64_ratio, d.driver.c_str(), d.runtime.c_str(), kBuildType);
        std::printf("Quadratur %d x %d, %d Felder, %d Wellenlängen, %d Aufwärmläufe, Median aus %d "
                    "Läufen, %d Threads pro Block\n",
                    o.rings, o.spokes, o.fields, o.wavelengths, o.warmup, o.repeat,
                    mko::cuda::kResidualThreads);
        std::printf("Engine = TraceEngine::run inkl. Prüfung, Upload und Download; Kernel = "
                    "synchroner Kernellauf ohne Upload/Download (Host-Uhr)\n\n");
        print_table_header();
        std::fflush(stdout);
    }
#ifndef NDEBUG
    std::fprintf(stderr, "Warnung: Debug-Build, Messwerte nicht aussagekräftig\n");
#endif

    std::vector<Result> results;
    for (const int S : o.surfaces) {
        const mko::LoadedSystem system = make_bench_system(S, o.fields, o.wavelengths);
        for (const int V : o.variants) {
            const mko::SystemBatchHost batch =
                make_bench_batch(system, V, o.seed + static_cast<std::uint64_t>(S));
            for (const NumberType type : o.types) {
                for (const ModeSpec& mode : o.modes) {
                    results.push_back(run_case(engine, o, type, mode, batch));
                    if (table) {
                        print_table_rows({results.back()});
                        std::fflush(stdout);
                    }
                }
            }
        }
    }
    if (table) {
        std::int64_t failed = 0;
        std::int64_t rays = 0;
        for (const Result& r : results) {
            failed += r.failed;
            rays += r.rays;
        }
        std::printf("\nVersagte Strahlen insgesamt: %lld von %lld\n",
                    static_cast<long long>(failed), static_cast<long long>(rays));
    }
    if (!o.csv.empty()) {
        emit(o.csv, [&](std::ostream& os) { write_csv(os, d, results); });
    }
    if (!o.json.empty()) {
        emit(o.json, [&](std::ostream& os) { os << to_json(o, d, results).dump(2) << '\n'; });
    }
    return kExitOk;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        if (o.help) {
            print_usage();
            return kExitOk;
        }
        return run(o);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "trace_throughput: %s (--help zeigt die Optionen)\n", e.what());
        return kExitUsage;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "trace_throughput: Fehler: %s\n", e.what());
        return kExitError;
    }
}
