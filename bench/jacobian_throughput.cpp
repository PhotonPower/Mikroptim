// Benchmark jacobian_throughput (M5-T3): Laufzeit der Jacobi-Matrix des gebündelten LM auf der
// GPU, finite Differenzen (DeviceFdJacobian) gegen adjungiert (DeviceAdjointJacobian), für
// synthetische Systeme mit S Flächen und P = 2S − 1 Variablen (alle Krümmungen, alle Dicken,
// Arbeitsabstand). Messgrößen, Optionen und Ausgabe: bench/README.md, Ergebnisse:
// docs/benchmarks.md.
//
// Pro Fall (Zahlentyp, S, Zusatzresiduen, Läufe) jeweils Median und Minimum aus --repeat Läufen
// nach --warmup Aufwärmläufen, Host-Uhr um synchrone Aufrufe:
//   - J:      compute der Jacobi-Matrizen aller Läufe (Batch auf dem Host, Upload, Kernel bzw.
//             Engine mit Differenzkernel, Zusatzzeilen); J bleibt auf dem Gerät;
//   - gesamt: J und Normalgleichungen (DeviceNormalEquations::compute und Download von A und g),
//             also die Arbeit von GpuLmBackend::normal_equations pro Teilstück;
//   - adjungiert zusätzlich die Anteile aus DeviceAdjointJacobian::timing() (Zusatzzeilen per
//     finiter Differenzen, Gerät).

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/adjoint_jacobian.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/cuda/normal_equations.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/version.hpp"

#include <cuda_runtime.h>

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
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// FP32:FP64-Verhältnis (cudaDeviceProp::singleToDoublePrecisionPerfRatio gibt es seit CUDA 13
// nicht mehr)
int fp32_fp64_ratio(int device) {
    int ratio = 0;
    (void)cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device);
    return ratio;
}

using mko::cuda::Precision;

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
constexpr int kExitNoGpu = 3; // wie trace_throughput (SKIP_RETURN_CODE des Smoke-Tests)

#ifdef NDEBUG
constexpr const char* kBuildType = "Release";
#else
constexpr const char* kBuildType = "Debug";
#endif

class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Zusatzresiduen eines Falls: keine (nur Strahl- und Strafzeilen, das Maß der Abnahme),
// paraxial (Brennweite als Gleichung, ohne Trace) oder paraxial plus mechanische Bedingungen mit
// Extrema des Traces (trace-abhängig: 2P + 1 MERIT-Varianten pro Lauf auch im adjungierten Pfad).
enum class ExtraKind {
    None,
    Paraxial,
    Mechanical,
};

const char* to_string(ExtraKind k) {
    switch (k) {
    case ExtraKind::None:
        return "none";
    case ExtraKind::Paraxial:
        return "paraxial";
    case ExtraKind::Mechanical:
        return "mechanical";
    }
    return "?";
}

struct Options {
    bool help = false;
    int device = 0;
    std::vector<int> surfaces{31, 41};
    std::vector<int> runs{1, 10, 100};
    std::vector<Precision> types{Precision::Fp64, Precision::Df64};
    std::vector<ExtraKind> extras{ExtraKind::None, ExtraKind::Paraxial, ExtraKind::Mechanical};
    int fields = 3;
    int wavelengths = 3;
    int rings = 10;
    int spokes = 8;
    bool hard_failure = false;
    int extra_threads = 0; // <= 0: wie das GPU-Backend des LM (lm_host_threads)
    int repeat = 5;
    int warmup = 1;
    std::string csv;
    bool quiet = false;
};

template <class T>
T parse_number(std::string_view option, std::string_view text) {
    T value{};
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || ec != std::errc{} || ptr != text.data() + text.size()) {
        throw UsageError(std::string(option) + " erwartet eine Zahl, nicht \"" + std::string(text) +
                         "\"");
    }
    return value;
}

std::vector<std::string_view> split_list(std::string_view text) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = std::min(text.find(',', start), text.size());
        out.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

std::vector<int> parse_int_list(std::string_view option, std::string_view text, int lo, int hi) {
    std::vector<int> out;
    for (const std::string_view item : split_list(text)) {
        const int v = parse_number<int>(option, item);
        if (v < lo || v > hi) {
            throw UsageError(std::string(option) + ": " + std::to_string(v) + " liegt nicht in [" +
                             std::to_string(lo) + ", " + std::to_string(hi) + "]");
        }
        out.push_back(v);
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
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
        } else if (arg == "--runs") {
            o.runs = parse_int_list(arg, value(), 1, std::numeric_limits<int>::max());
        } else if (arg == "--type") {
            o.types.clear();
            for (const std::string_view item : split_list(value())) {
                if (item == "fp64") {
                    o.types.push_back(Precision::Fp64);
                } else if (item == "df64") {
                    o.types.push_back(Precision::Df64);
                } else {
                    throw UsageError("--type: unbekannter Zahlentyp \"" + std::string(item) +
                                     "\" (fp64, df64)");
                }
            }
        } else if (arg == "--extra") {
            o.extras.clear();
            for (const std::string_view item : split_list(value())) {
                if (item == "none") {
                    o.extras.push_back(ExtraKind::None);
                } else if (item == "paraxial") {
                    o.extras.push_back(ExtraKind::Paraxial);
                } else if (item == "mechanical") {
                    o.extras.push_back(ExtraKind::Mechanical);
                } else {
                    throw UsageError("--extra: unbekannte Art \"" + std::string(item) +
                                     "\" (none, paraxial, mechanical)");
                }
            }
        } else if (arg == "--fields") {
            o.fields = parse_number<int>(arg, value());
        } else if (arg == "--wavelengths") {
            o.wavelengths = parse_number<int>(arg, value());
        } else if (arg == "--rings") {
            o.rings = parse_number<int>(arg, value());
        } else if (arg == "--spokes") {
            o.spokes = parse_number<int>(arg, value());
        } else if (arg == "--hard-failure") {
            o.hard_failure = true;
        } else if (arg == "--extra-threads") {
            o.extra_threads = parse_number<int>(arg, value());
        } else if (arg == "--repeat") {
            o.repeat = parse_number<int>(arg, value());
        } else if (arg == "--warmup") {
            o.warmup = parse_number<int>(arg, value());
        } else if (arg == "--csv") {
            o.csv = std::string(value());
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
    if (o.surfaces.empty() || o.runs.empty() || o.types.empty() || o.extras.empty()) {
        throw UsageError("leere Liste");
    }
    return o;
}

void print_usage() {
    std::printf(
        "Aufruf: jacobian_throughput [Optionen]\n"
        "  Laufzeit der Jacobi-Matrix des LM auf der GPU: finite Differenzen gegen adjungiert\n"
        "  (M5-T3), synthetische Systeme mit S Flächen und P = 2S - 1 Variablen.\n"
        "  --device N          CUDA-Gerät (Standard 0)\n"
        "  --surfaces LISTE    Flächenzahlen S, 4 bis 64 (Standard 31,41: P = 61, 81)\n"
        "  --runs LISTE        Läufe pro Aufruf (Standard 1,10,100)\n"
        "  --type LISTE        fp64, df64 (Standard beide)\n"
        "  --extra LISTE       Zusatzresiduen: none, paraxial, mechanical (Standard alle)\n"
        "  --fields F          Felder 0 ... 3 Grad (Standard 3)\n"
        "  --wavelengths L     Wellenlängen (Standard 3)\n"
        "  --rings N, --spokes M  Pupillenquadratur (Standard 10 x 8)\n"
        "  --hard-failure      hartes Strahlversagen (ohne Strafzeilen; Standard weich)\n"
        "  --extra-threads N   Host-Threads der Zusatzresiduen (Standard wie im LM)\n"
        "  --repeat R          gemessene Läufe pro Fall (Standard 5)\n"
        "  --warmup W          Aufwärmläufe pro Fall (Standard 1)\n"
        "  --csv DATEI         Ergebnisse als CSV ('-' = Standardausgabe)\n"
        "  --quiet             keine Tabelle\n"
        "Exit-Code: 0 Erfolg, 1 Fehler, 2 Aufruffehler, 3 kein CUDA-Gerät.\n");
}

// ---- Synthetisches System (wie trace_throughput, mit Variablen und Nebenbedingungen) -------

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

mko::VariableSpec variable(mko::VariableType type, mko::SurfaceSelection selection,
                           std::vector<int> surfaces = {}) {
    mko::VariableSpec v;
    v.type = type;
    v.selection = selection;
    v.surfaces = std::move(surfaces);
    return v;
}

// Frei erfundenes System mit S Flächen (Generator von trace_throughput, kein reales Objektiv):
// Blende, Folge schwacher Menisken aus Kron und Flint (Buchdahl), bei ungeradem S − 1 ein
// verkittetes Glied; Brennweite 40 mm, Bildebene im paraxialen Fokus. Variablen: alle
// Krümmungen (S − 1), alle Dicken außer der letzten (S − 1) und der Arbeitsabstand, ohne
// Schranken (P = 2S − 1). Nebenbedingungen nach `extra`.
mko::LoadedSystem make_system(int S, int F, int L, ExtraKind extra) {
    const mko::MaterialSpec crown = buchdahl(1.52, -0.08, 0.01);
    const mko::MaterialSpec flint = buchdahl(1.62, -0.12, 0.02);
    const mko::MaterialSpec air{};
    constexpr double kTotalPower = 1.0 / 40.0;
    constexpr double kGlass = 1.0;
    constexpr double kAir = 0.3;
    constexpr double kShape = 1.5;

    const int after_stop = S - 1;
    const bool doublet = after_stop % 2 == 1;
    const int singlets = doublet ? (after_stop - 3) / 2 : after_stop / 2;
    const double phi = kTotalPower / static_cast<double>(singlets + (doublet ? 1 : 0));

    mko::SystemDescription sys;
    sys.name = "Jacobi-Benchmark S=" + std::to_string(S);
    sys.description = "Synthetisches Benchmark-System (M5-T3), frei erfunden";
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
        const mko::MaterialSpec& glass = i % 2 == 0 ? crown : flint;
        const double d = phi / (glass.n0 - 1.0);
        const double x = i % 4 < 2 ? kShape : -kShape;
        sys.surfaces.push_back(sphere(0.5 * d * (x + 1.0), kGlass, glass));
        sys.surfaces.push_back(sphere(0.5 * d * (x - 1.0), kAir, air));
    }
    if (doublet) {
        const double d1 = 1.6 * phi / (crown.n0 - 1.0);
        const double c2 = -0.5 * d1;
        sys.surfaces.push_back(sphere(0.5 * d1, 1.2, crown));
        sys.surfaces.push_back(sphere(c2, 0.6, flint));
        sys.surfaces.push_back(sphere(c2 + 0.6 * phi / (flint.n0 - 1.0), kAir, air));
    }
    if (static_cast<int>(sys.surfaces.size()) != S) {
        throw std::logic_error("make_system: falsche Flächenzahl");
    }
    {
        const mko::LoadedSystem first = mko::prepare_system(sys, {});
        const mko::ParaxialData p = mko::compute_paraxial(first.system, first.catalog);
        if (p.afocal || !(p.back_focal_distance > 1.0)) {
            throw std::logic_error("make_system: ungeeignete Schnittweite");
        }
        sys.surfaces.back().thickness = p.back_focal_distance;
        sys.target = mko::Target{};
        sys.target->focal_length = p.focal_length;
    }
    std::vector<int> thick;
    for (int s = 0; s + 1 < S; ++s) {
        thick.push_back(s);
    }
    sys.variables =
        mko::Variables{{variable(mko::VariableType::Curvature, mko::SurfaceSelection::All),
                        variable(mko::VariableType::Thickness, mko::SurfaceSelection::List, thick),
                        variable(mko::VariableType::WorkingDistance, mko::SurfaceSelection::None)}};
    if (extra != ExtraKind::None) {
        mko::Constraints c;
        mko::ConstraintSpec f;
        f.type = mko::ConstraintType::FocalLength;
        f.weight = 1.0;
        c.items.push_back(f);
        if (extra == ExtraKind::Mechanical) {
            // trace-abhängig (Extrema): Strahlhöhe, Glasrand bei freier Öffnung, Einfallswinkel
            mko::ConstraintSpec h;
            h.type = mko::ConstraintType::MaxSemiDiameter;
            h.weight = 1.0;
            c.items.push_back(h);
            mko::ConstraintSpec e;
            e.type = mko::ConstraintType::MinEdgeThickness;
            e.weight = 1.0;
            e.medium = mko::ConstraintMedium::Glass;
            e.thickness = 0.2;
            e.height = mko::EdgeHeight::Rays;
            c.items.push_back(e);
            mko::ConstraintSpec a;
            a.type = mko::ConstraintType::MaxIncidenceAngle;
            a.weight = 1.0;
            a.angle = 60.0;
            c.items.push_back(a);
        }
        sys.constraints = c;
    }
    return mko::prepare_system(sys, {});
}

// ---- Messung ------------------------------------------------------------------------------

struct Timing {
    double median_s = 0.0;
    double min_s = 0.0;
};

Timing summarize(std::vector<double> t) {
    std::sort(t.begin(), t.end());
    const std::size_t n = t.size();
    Timing r;
    r.median_s = n % 2 == 1 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    r.min_s = t.front();
    return r;
}

double seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct Result {
    Precision type = Precision::Fp64;
    ExtraKind extra = ExtraKind::None;
    int surfaces = 0;
    int variables = 0;
    int runs = 0;
    std::int64_t rows = 0; // R pro Lauf
    int extra_rows = 0;
    Timing fd_j, fd_total, ad_j, ad_total, ad_extra, ad_device;
    double gram_deviation = 0.0; // max |A_ad − A_fd| / max |A_fd| (Kontrolle)
};

Result run_case(mko::cuda::TraceEngine& engine, const Options& o, Precision type, ExtraKind extra,
                const mko::LoadedSystem& system, int runs) {
    const mko::SystemBatchHost base = mko::make_system_batch(system, 1);
    const mko::VariableSet set(system.system, base.topology());
    const std::vector<double> x0 = set.extract(base, 0);
    std::vector<double> starts;
    mko::PerturbationOptions po;
    po.seed = 20261019;
    po.first_is_base = true;
    for (int j = 0; j < runs; ++j) {
        const std::vector<double> y =
            mko::perturbed_vector(set, x0, po, static_cast<std::uint64_t>(j));
        starts.insert(starts.end(), y.begin(), y.end());
    }
    mko::FdJacobianOptions fo;
    fo.quadrature = mko::ResidualSettings{o.rings, o.spokes};
    fo.quadrature.failure = o.hard_failure ? mko::FailureMode::Hard : mko::FailureMode::Soft;
    if (extra != ExtraKind::None) {
        mko::MechanicalResiduals m =
            mko::make_mechanical_residuals(mko::make_mechanical_constraints(system.system));
        fo.extra = mko::concat_extra_residuals(
            {mko::make_paraxial_residuals(mko::make_paraxial_constraints(system.system)),
             std::move(m.geometric)});
        fo.trace_extra = std::move(m.traced);
    }
    // Host-Threads der Zusatzresiduen wie im GPU-Backend des LM (lm_host_threads)
    fo.extra_threads = mko::lm_host_threads(o.extra_threads,
                                            static_cast<std::int64_t>(runs) * (2 * set.size() + 1),
                                            mko::kLmExtraPerThread);
    const mko::AdjointJacobianOptions ao = mko::adjoint_options_of(fo);

    mko::cuda::DeviceFdJacobian fd;
    mko::cuda::DeviceAdjointJacobian ad;
    mko::cuda::DeviceNormalEquations normal;
    std::vector<double> a;
    std::vector<double> g;
    std::vector<double> fd_j, fd_total, ad_j, ad_total, ad_extra, ad_device;
    for (int i = 0; i < o.warmup + o.repeat; ++i) {
        const bool record = i >= o.warmup;
        auto t0 = std::chrono::steady_clock::now();
        fd.compute(engine, set, base, starts, fo, type);
        auto t1 = std::chrono::steady_clock::now();
        normal.compute(fd);
        normal.download(a, g);
        auto t2 = std::chrono::steady_clock::now();
        if (record) {
            fd_j.push_back(seconds(t0, t1));
            fd_total.push_back(seconds(t0, t2));
        }
        t0 = std::chrono::steady_clock::now();
        ad.compute(engine, set, base, starts, ao, type);
        t1 = std::chrono::steady_clock::now();
        normal.compute(ad);
        normal.download(a, g);
        t2 = std::chrono::steady_clock::now();
        if (record) {
            ad_j.push_back(seconds(t0, t1));
            ad_total.push_back(seconds(t0, t2));
            ad_extra.push_back(ad.timing().extra_seconds);
            ad_device.push_back(ad.timing().device_seconds);
        }
    }
    // Kontrolle: A aus beiden Verfahren (Fehler der FD-Jacobi)
    std::vector<double> a_fd;
    normal.compute(fd);
    normal.download(a_fd, g);
    normal.compute(ad);
    normal.download(a, g);
    double a_max = 0.0;
    double d_max = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        a_max = std::max(a_max, std::abs(a_fd[k]));
        d_max = std::max(d_max, std::abs(a[k] - a_fd[k]));
    }

    Result r;
    r.type = type;
    r.extra = extra;
    r.surfaces = base.surfaces();
    r.variables = set.size();
    r.runs = runs;
    r.rows = ad.shape().rows();
    r.extra_rows = ad.shape().extra_rows;
    r.fd_j = summarize(fd_j);
    r.fd_total = summarize(fd_total);
    r.ad_j = summarize(ad_j);
    r.ad_total = summarize(ad_total);
    r.ad_extra = summarize(ad_extra);
    r.ad_device = summarize(ad_device);
    r.gram_deviation = a_max > 0.0 ? d_max / a_max : 0.0;
    return r;
}

// ---- Ausgabe ------------------------------------------------------------------------------

std::string ms(double s) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", 1e3 * s);
    return buf;
}

void print_header() {
    std::printf("\n%-5s %-10s %3s %3s %5s %6s %3s | %10s %10s | %10s %10s %9s %9s | %7s %7s | "
                "%8s\n",
                "Typ", "Zusatz", "S", "P", "Läufe", "R", "E", "FD J [ms]", "FD ges.", "Adj J",
                "Adj ges.", "Zusatz", "Gerät", "J x", "ges. x", "|ΔA|");
}

void print_row(const Result& r) {
    std::printf("%-5s %-10s %3d %3d %5d %6lld %3d | %10s %10s | %10s %10s %9s %9s | %7.1f %7.1f "
                "| %8.1e\n",
                mko::cuda::to_string(r.type), to_string(r.extra), r.surfaces, r.variables, r.runs,
                static_cast<long long>(r.rows), r.extra_rows, ms(r.fd_j.median_s).c_str(),
                ms(r.fd_total.median_s).c_str(), ms(r.ad_j.median_s).c_str(),
                ms(r.ad_total.median_s).c_str(), ms(r.ad_extra.median_s).c_str(),
                ms(r.ad_device.median_s).c_str(), r.fd_j.median_s / r.ad_j.median_s,
                r.fd_total.median_s / r.ad_total.median_s, r.gram_deviation);
    std::fflush(stdout);
}

void write_csv(std::ostream& os, const std::string& gpu, const std::vector<Result>& results) {
    os << "gpu,type,extra,surfaces,variables,runs,rows,extra_rows,fd_j_median_s,fd_j_min_s,"
          "fd_total_median_s,fd_total_min_s,adjoint_j_median_s,adjoint_j_min_s,"
          "adjoint_total_median_s,adjoint_total_min_s,adjoint_extra_median_s,"
          "adjoint_device_median_s,speedup_j,speedup_total,gram_deviation\n";
    for (const Result& r : results) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "\"%s\",%s,%s,%d,%d,%d,%lld,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,"
                      "%.6e,%.6e,%.4f,%.4f,%.3e\n",
                      gpu.c_str(), mko::cuda::to_string(r.type), to_string(r.extra), r.surfaces,
                      r.variables, r.runs, static_cast<long long>(r.rows), r.extra_rows,
                      r.fd_j.median_s, r.fd_j.min_s, r.fd_total.median_s, r.fd_total.min_s,
                      r.ad_j.median_s, r.ad_j.min_s, r.ad_total.median_s, r.ad_total.min_s,
                      r.ad_extra.median_s, r.ad_device.median_s, r.fd_j.median_s / r.ad_j.median_s,
                      r.fd_total.median_s / r.ad_total.median_s, r.gram_deviation);
        os << buf;
    }
}

int run(const Options& o) {
    if (mko::cuda::device_count() == 0) {
        std::fprintf(stderr, "jacobian_throughput: kein CUDA-Gerät gefunden\n");
        return kExitNoGpu;
    }
    if (o.device >= mko::cuda::device_count()) {
        throw UsageError("--device " + std::to_string(o.device) + ": Gerät existiert nicht");
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, o.device) != cudaSuccess) {
        throw mko::cuda::CudaError("cudaGetDeviceProperties fehlgeschlagen");
    }
    int runtime = 0;
    cudaRuntimeGetVersion(&runtime);
    const std::string gpu = p.name;
    const bool table = !o.quiet;
#ifndef NDEBUG
    std::fprintf(stderr, "Warnung: Debug-Build, Zeiten nicht aussagekräftig\n");
#endif
    if (table) {
        std::printf("jacobian_throughput (Mikroptim %s, %s): GPU %d %s, sm_%d%d, %d SMs, "
                    "FP32:FP64 = %d:1, CUDA %d.%d\n",
                    mko::kVersion, kBuildType, o.device, gpu.c_str(), p.major, p.minor,
                    p.multiProcessorCount, fp32_fp64_ratio(o.device), runtime / 1000,
                    (runtime % 1000) / 10);
        std::printf("Quadratur %d x %d, %d Felder, %d Wellenlängen, Strahlversagen %s, Median aus "
                    "%d Läufen; Zeiten in ms, x = FD/adjungiert\n",
                    o.rings, o.spokes, o.fields, o.wavelengths, o.hard_failure ? "hart" : "weich",
                    o.repeat);
        print_header();
    }
    mko::cuda::TraceEngine engine(o.device);
    std::vector<Result> results;
    for (const int S : o.surfaces) {
        for (const ExtraKind extra : o.extras) {
            const mko::LoadedSystem system = make_system(S, o.fields, o.wavelengths, extra);
            for (const Precision type : o.types) {
                for (const int runs : o.runs) {
                    results.push_back(run_case(engine, o, type, extra, system, runs));
                    if (table) {
                        print_row(results.back());
                    }
                }
            }
        }
    }
    if (!o.csv.empty()) {
        if (o.csv == "-") {
            write_csv(std::cout, gpu, results);
        } else {
            std::ofstream f(o.csv);
            if (!f) {
                throw std::runtime_error("kann " + o.csv + " nicht schreiben");
            }
            write_csv(f, gpu, results);
        }
    }
    if (table) {
        double worst = std::numeric_limits<double>::infinity();
        for (const Result& r : results) {
            if (r.extra == ExtraKind::None) {
                worst = std::min(worst, r.fd_j.median_s / r.ad_j.median_s);
            }
        }
        if (std::isfinite(worst)) {
            std::printf("\nKleinster Faktor der Jacobi (ohne Zusatzresiduen): %.1f\n", worst);
        }
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
        std::fprintf(stderr, "jacobian_throughput: %s (--help zeigt die Optionen)\n", e.what());
        return kExitUsage;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "jacobian_throughput: Fehler: %s\n", e.what());
        return kExitError;
    }
}
