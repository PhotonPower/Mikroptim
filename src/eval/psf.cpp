// Skalare PSF, MTF und Strehl (M8-T2): Pupillenfunktion aus dem Modus EVAL, CPU-Referenz mit der
// eigenen Radix-2-FFT und die Kennzahlen für CPU- und GPU-Ergebnisse. Formeln: docs/numerik.md,
// Abschnitt "Beugung: PSF, MTF, Strehl (M8-T2)".

#include "mikroptim/eval/psf.hpp"
#include "mikroptim/eval/fft.hpp"
#include "mikroptim/eval/psf_ops.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko {

namespace {

std::size_t to_size(std::int64_t v) {
    return static_cast<std::size_t>(v);
}

// Normierte Pupillenkoordinate der Zellmitte i eines N-Rasters (wie make_eval_grid).
double cell_center(int i, int n) {
    return static_cast<double>(2 * i + 1 - n) / static_cast<double>(n);
}

// |Σ exp(2πi (OPD − t_x x − t_y y)/λ)|² über die transmittierenden Rasterpunkte (zeilenweise in
// aufsteigender Reihenfolge), geteilt durch n_voll².
double direct_strehl(const PsfPupil& p, double tilt_x, double tilt_y) {
    const int n = p.grid;
    const double lambda = p.wavelength_mm();
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (int j = 0; j < n; ++j) {
        const double y = cell_center(j, n);
        for (int i = 0; i < n; ++i) {
            const double opd = p.opd[to_size(j) * to_size(n) + to_size(i)];
            if (!std::isfinite(opd)) {
                continue;
            }
            const double x = cell_center(i, n);
            double re = 0.0;
            double im = 0.0;
            psf_pupil_value(opd - tilt_x * x - tilt_y * y, lambda, re, im);
            sum_re += re;
            sum_im += im;
        }
    }
    const double full = static_cast<double>(p.full_count);
    return (sum_re * sum_re + sum_im * sum_im) / (full * full);
}

// Kleinste-Quadrate-Fit OPD ≈ c₀ + t_x x + t_y y über die transmittierenden Rasterpunkte
// (gleiche Gewichte); false bei weniger als drei Punkten oder singulärer Matrix.
bool fit_tilt(const PsfPupil& p, double& tilt_x, double& tilt_y) {
    const int n = p.grid;
    double g[3][3] = {};
    double h[3] = {};
    for (int j = 0; j < n; ++j) {
        const double y = cell_center(j, n);
        for (int i = 0; i < n; ++i) {
            const double opd = p.opd[to_size(j) * to_size(n) + to_size(i)];
            if (!std::isfinite(opd)) {
                continue;
            }
            const double phi[3] = {1.0, cell_center(i, n), y};
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    g[a][b] += phi[a] * phi[b];
                }
                h[a] += phi[a] * opd;
            }
        }
    }
    // Gauß-Elimination mit Spaltenpivot (3×3)
    for (int c = 0; c < 3; ++c) {
        int piv = c;
        for (int a = c + 1; a < 3; ++a) {
            if (std::abs(g[a][c]) > std::abs(g[piv][c])) {
                piv = a;
            }
        }
        if (!(std::abs(g[piv][c]) > 0.0)) {
            return false;
        }
        for (int b = 0; b < 3; ++b) {
            const double t = g[c][b];
            g[c][b] = g[piv][b];
            g[piv][b] = t;
        }
        const double t = h[c];
        h[c] = h[piv];
        h[piv] = t;
        for (int a = c + 1; a < 3; ++a) {
            const double q = g[a][c] / g[c][c];
            for (int b = c; b < 3; ++b) {
                g[a][b] -= q * g[c][b];
            }
            h[a] -= q * h[c];
        }
    }
    double x[3] = {};
    for (int a = 2; a >= 0; --a) {
        double s = h[a];
        for (int b = a + 1; b < 3; ++b) {
            s -= g[a][b] * x[b];
        }
        x[a] = s / g[a][a];
    }
    tilt_x = x[1];
    tilt_y = x[2];
    return std::isfinite(tilt_x) && std::isfinite(tilt_y);
}

void check_pupil(const PsfPupil& p) {
    if (p.grid < 1 || p.grid > kMaxEvalGrid || p.opd.size() != to_size(p.grid) * to_size(p.grid)) {
        throw std::invalid_argument("PSF: Pupillenkarte hat nicht N·N Werte");
    }
    if (p.full_count < 1) {
        throw std::invalid_argument("PSF: Zahl der Punkte der vollen Pupille muss >= 1 sein");
    }
    if (!(p.wavelength_nm > 0.0) || !std::isfinite(p.wavelength_nm)) {
        throw std::invalid_argument("PSF: Wellenlänge muss positiv sein");
    }
    if (!(p.na_x > 0.0) || !(p.na_y > 0.0) || !std::isfinite(p.na_x) || !std::isfinite(p.na_y)) {
        throw std::invalid_argument("PSF: NA der Pupillenabbildung muss positiv sein");
    }
}

double mtf_at(const std::vector<double>& mtf, double nu) {
    if (mtf.empty() || !(nu >= 0.0)) {
        return PsfResult::kNaN;
    }
    const auto n = static_cast<double>(mtf.size() - 1);
    const double x = nu * n;
    if (x >= n) {
        return nu > 1.0 ? 0.0 : mtf.back();
    }
    const auto k = static_cast<std::size_t>(x);
    const double t = x - static_cast<double>(k);
    return (1.0 - t) * mtf[k] + t * mtf[k + 1];
}

} // namespace

double PsfResult::mtf_sagittal_at(double nu) const {
    return mtf_at(mtf_sagittal, nu);
}

double PsfResult::mtf_tangential_at(double nu) const {
    return mtf_at(mtf_tangential, nu);
}

void check_psf_settings(const PsfSettings& s) {
    if (!(s.padding >= 2.0) || !std::isfinite(s.padding)) {
        throw std::invalid_argument("PSF: Zero-Padding-Faktor muss >= 2 sein");
    }
    if (!(s.numerical_aperture >= 0.0) || !std::isfinite(s.numerical_aperture)) {
        throw std::invalid_argument("PSF: vorgegebene NA muss >= 0 sein (0: aus den Feldstrahlen)");
    }
}

int psf_fft_size(int grid, double padding) {
    if (grid < 1 || grid > kMaxEvalGrid) {
        throw std::invalid_argument("PSF: Rasterkante N = " + std::to_string(grid) +
                                    " außerhalb von [1, " + std::to_string(kMaxEvalGrid) + "]");
    }
    PsfSettings s;
    s.padding = padding;
    check_psf_settings(s);
    const double target = std::ceil(padding * static_cast<double>(grid));
    const double minimum = std::max(2.0 * static_cast<double>(grid), target);
    if (minimum > static_cast<double>(kMaxPsfFft)) {
        throw std::invalid_argument("PSF: FFT-Größe " + std::to_string(minimum) +
                                    " überschreitet " + std::to_string(kMaxPsfFft) +
                                    "; Raster oder Zero-Padding verkleinern");
    }
    int m = 1;
    while (static_cast<double>(m) < minimum) {
        m *= 2;
    }
    return m;
}

PsfPupil make_psf_pupil(const SystemBatchHost& batch, const EvalResult& result, int v, int f, int l,
                        const PsfSettings& settings) {
    check_psf_settings(settings);
    if (batch.variants() != result.variants || batch.fields() != result.fields ||
        batch.wavelengths() != result.wavelengths) {
        throw std::invalid_argument("PSF: Batch passt nicht zum EVAL-Ergebnis");
    }
    const EvalMap map = eval_opd_map(result, v, f, l, OpdRemoval::Piston); // prüft v, f, l
    PsfPupil p;
    p.field = f;
    p.wavelength = l;
    p.grid = map.size;
    p.full_count = result.grid.full_count;
    p.wavelength_nm = batch.wavelength_nm(l);
    p.opd = map.values;

    // Ableitungen der Bildraumrichtung nach den normierten Pupillenkoordinaten aus den
    // Feldstrahlen: a_y = n' (d_y(0, δ) − d_y(0, −δ))/(2δ), a_x = n' (d_x(δ, 0) − d_x(0, 0))/δ.
    const double n_image = batch.index(v, batch.surfaces() - 1, l);
    const double delta = result.settings.field_ray_offset;
    const auto ok = [&](int j) { return result.field_ray_status(v, f, l, j) == TraceStatus::Ok; };
    double a_x = PsfResult::kNaN;
    double a_y = PsfResult::kNaN;
    if (ok(kEvalChiefRay) && ok(kEvalSagittal)) {
        a_x = n_image *
              (result.field_ray(v, f, l, kEvalSagittal, 2) -
               result.field_ray(v, f, l, kEvalChiefRay, 2)) /
              delta;
    }
    if (ok(kEvalTangentialPlus) && ok(kEvalTangentialMinus)) {
        a_y = n_image *
              (result.field_ray(v, f, l, kEvalTangentialPlus, 3) -
               result.field_ray(v, f, l, kEvalTangentialMinus, 3)) /
              (2.0 * delta);
    }
    p.flip_x = std::isfinite(a_x) && a_x > 0.0;
    p.flip_y = std::isfinite(a_y) && a_y > 0.0;
    if (settings.numerical_aperture > 0.0) {
        p.na_x = settings.numerical_aperture;
        p.na_y = settings.numerical_aperture;
    } else {
        if (!std::isfinite(a_x) || !std::isfinite(a_y) || a_x == 0.0 || a_y == 0.0) {
            throw std::invalid_argument(
                "PSF: NA der Pupillenabbildung für Feld " + std::to_string(f) + ", Wellenlänge " +
                std::to_string(l) +
                " nicht bestimmbar (Feldstrahl versagt oder afokal); NA vorgeben");
        }
        p.na_x = std::abs(a_x);
        p.na_y = std::abs(a_y);
    }
    return p;
}

std::vector<PsfPupil> make_psf_pupils(const SystemBatchHost& batch, const EvalResult& result, int v,
                                      const PsfSettings& settings) {
    std::vector<PsfPupil> out;
    out.reserve(static_cast<std::size_t>(result.fields) *
                static_cast<std::size_t>(result.wavelengths));
    for (int f = 0; f < result.fields; ++f) {
        for (int l = 0; l < result.wavelengths; ++l) {
            out.push_back(make_psf_pupil(batch, result, v, f, l, settings));
        }
    }
    return out;
}

PsfResult prepare_psf_result(const PsfPupil& pupil, int fft_size) {
    check_pupil(pupil);
    if (!is_power_of_two(fft_size) || fft_size < 2 * pupil.grid || fft_size > kMaxPsfFft) {
        throw std::invalid_argument(
            "PSF: FFT-Größe " + std::to_string(fft_size) + " muss eine Zweierpotenz in [2N, " +
            std::to_string(kMaxPsfFft) + "] sein (N = " + std::to_string(pupil.grid) + ")");
    }
    PsfResult r;
    r.field = pupil.field;
    r.wavelength = pupil.wavelength;
    r.grid = pupil.grid;
    r.fft_size = fft_size;
    r.wavelength_nm = pupil.wavelength_nm;
    r.na_x = pupil.na_x;
    r.na_y = pupil.na_y;
    const double lambda = pupil.wavelength_mm();
    const double scale = lambda * static_cast<double>(pupil.grid) / (2.0 * fft_size);
    r.dx = scale / pupil.na_x;
    r.dy = scale / pupil.na_y;
    r.airy_radius =
        kBesselJ1FirstZero / (2.0 * std::numbers::pi) * lambda / std::sqrt(pupil.na_x * pupil.na_y);
    r.psf.assign(to_size(fft_size) * to_size(fft_size), 0.0);
    r.mtf_sagittal.assign(to_size(pupil.grid) + 1, 0.0);
    r.mtf_tangential.assign(to_size(pupil.grid) + 1, 0.0);
    return r;
}

void finish_psf(PsfResult& r, const PsfPupil& p) {
    check_pupil(p);
    const auto m = to_size(r.fft_size);
    if (r.grid != p.grid || r.psf.size() != m * m) {
        throw std::invalid_argument("PSF: Ergebnis passt nicht zur Pupille");
    }
    r.transmitted = 0;
    for (const double v : p.opd) {
        r.transmitted += std::isfinite(v) ? 1 : 0;
    }
    double sum = 0.0;
    std::size_t peak_index = 0;
    for (std::size_t k = 0; k < r.psf.size(); ++k) {
        sum += r.psf[k];
        if (r.psf[k] > r.psf[peak_index]) {
            peak_index = k;
        }
    }
    const double md = static_cast<double>(m);
    r.energy = sum * static_cast<double>(p.full_count) / (md * md);
    r.strehl_fft = r.psf[(m / 2) * m + m / 2];
    r.strehl_origin = direct_strehl(p, 0.0, 0.0);
    double tilt_x = 0.0;
    double tilt_y = 0.0;
    if (fit_tilt(p, tilt_x, tilt_y)) {
        r.strehl = direct_strehl(p, tilt_x, tilt_y);
        // Kippung t (OPD pro normierter Pupillenkoordinate) verschiebt die PSF um t/NA,
        // gespiegelt bei divergenter Welle (numerik.md).
        r.reference_x = (p.flip_x ? -tilt_x : tilt_x) / p.na_x;
        r.reference_y = (p.flip_y ? -tilt_y : tilt_y) / p.na_y;
    } else {
        r.strehl = PsfResult::kNaN;
        r.reference_x = PsfResult::kNaN;
        r.reference_y = PsfResult::kNaN;
    }
    r.peak = r.psf[peak_index];
    r.peak_x = (static_cast<double>(peak_index % m) - md / 2.0) * r.dx;
    r.peak_y = (static_cast<double>(peak_index / m) - md / 2.0) * r.dy;
}

PsfResult psf_cpu(const PsfPupil& pupil, int fft_size) {
    PsfResult r = prepare_psf_result(pupil, fft_size);
    const int n = pupil.grid;
    const auto m = to_size(fft_size);
    const double lambda = pupil.wavelength_mm();
    std::vector<std::complex<double>> buf(m * m, std::complex<double>(0.0, 0.0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double re = 0.0;
            double im = 0.0;
            psf_pupil_value(pupil.opd[to_size(j) * to_size(n) + to_size(i)], lambda, re, im);
            buf[to_size(j) * m + to_size(i)] = {re, im};
        }
    }
    fft2d(buf, fft_size, fft_size, FftDirection::Forward);

    // Intensität normiert auf die ungestörte volle Pupille (|U(0)| = n_voll), zentriert und bei
    // divergenter Welle gespiegelt; dasselbe Feld danach reell für die MTF.
    const double full = static_cast<double>(pupil.full_count);
    const double scale = 1.0 / (full * full);
    const int sx = pupil.flip_x ? -1 : 1;
    const int sy = pupil.flip_y ? -1 : 1;
    const auto mi = static_cast<std::int64_t>(m);
    for (std::size_t qj = 0; qj < m; ++qj) {
        const auto oj = to_size(psf_output_index(static_cast<std::int64_t>(qj), mi, sy));
        for (std::size_t qi = 0; qi < m; ++qi) {
            std::complex<double>& u = buf[qj * m + qi];
            const double intensity = psf_norm2(u.real(), u.imag()) * scale;
            const auto oi = to_size(psf_output_index(static_cast<std::int64_t>(qi), mi, sx));
            r.psf[oj * m + oi] = intensity;
            u = {intensity, 0.0};
        }
    }
    fft2d(buf, fft_size, fft_size, FftDirection::Inverse);

    // OTF = FFT⁻¹(I) ∝ Autokorrelation der Pupille; MTF = |OTF| / OTF(0)
    const double otf0 = buf[0].real();
    for (int k = 0; k <= n; ++k) {
        const std::complex<double> s = buf[to_size(k)];
        const std::complex<double> t = buf[to_size(k) * m];
        r.mtf_sagittal[to_size(k)] =
            otf0 > 0.0 ? msqrt(psf_norm2(s.real(), s.imag())) / otf0 : PsfResult::kNaN;
        r.mtf_tangential[to_size(k)] =
            otf0 > 0.0 ? msqrt(psf_norm2(t.real(), t.imag())) / otf0 : PsfResult::kNaN;
    }
    finish_psf(r, pupil);
    return r;
}

PsfDeviceLayout make_psf_device_layout(int grid, int fft_size, int pupils) {
    if (grid < 1 || fft_size < 2 * grid || pupils < 1) {
        throw std::invalid_argument("PSF: ungültige Größen für die Gerätepuffer");
    }
    const auto t = static_cast<std::size_t>(pupils);
    const std::size_t m2 = to_size(fft_size) * to_size(fft_size);
    const std::size_t n2 = to_size(grid) * to_size(grid);
    PsfDeviceLayout l;
    std::size_t offset = 0;
    const auto place = [&offset](std::size_t bytes) {
        const BufferRegion r{offset, bytes};
        offset += align_bytes(bytes);
        return r;
    };
    l.spectrum = place(t * m2 * 2 * sizeof(double));
    l.psf = place(t * m2 * sizeof(double));
    l.opd = place(t * n2 * sizeof(double));
    l.params = place(t * sizeof(PsfPupilParams));
    l.mtf = place(t * 2 * (to_size(grid) + 1) * sizeof(double));
    l.total_bytes = offset;
    return l;
}

std::vector<PsfResult> psf_cpu(const SystemBatchHost& batch, const EvalResult& result, int v,
                               const PsfSettings& settings) {
    const std::vector<PsfPupil> pupils = make_psf_pupils(batch, result, v, settings);
    const int m = psf_fft_size(result.grid.size, settings.padding);
    std::vector<PsfResult> out;
    out.reserve(pupils.size());
    for (const PsfPupil& p : pupils) {
        out.push_back(psf_cpu(p, m));
    }
    return out;
}

} // namespace mko
