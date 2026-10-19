// Vektorielle PSF (M8-T5, Debye-Wolf/Richards-Wolf): Pupille aus dem Modus EVAL, Bezugsintensität
// der perfekten Pupille, CPU-Referenz mit der eigenen Radix-2-FFT, direkte Auswertung des
// Debye-Integrals und die Kennzahlen für CPU- und GPU-Ergebnisse. Formeln: docs/numerik.md,
// Abschnitt "Vektorielle PSF (M8-T5)".

#include "mikroptim/eval/vector_psf.hpp"
#include "mikroptim/eval/fft.hpp"
#include "mikroptim/eval/psf_ops.hpp"
#include "mikroptim/eval/vector_psf_ops.hpp"

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

using Complex = std::complex<double>;

std::size_t to_size(std::int64_t v) {
    return static_cast<std::size_t>(v);
}

// Normierte Pupillenkoordinate der Zellmitte i eines N-Rasters (wie make_eval_grid).
double cell_center(int i, int n) {
    return static_cast<double>(2 * i + 1 - n) / static_cast<double>(n);
}

// Punkt (i, j) gehört zur vollen Pupille (ganzzahlig, wie make_eval_grid).
bool in_pupil(int i, int j, int n) {
    const long long a = 2LL * i + 1 - n;
    const long long b = 2LL * j + 1 - n;
    return a * a + b * b <= static_cast<long long>(n) * n;
}

// Produkt ohne die Sonderbehandlung von NaN/∞ (wie in fft.cpp).
Complex mul(Complex a, Complex b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

Complex unit_phase(double phase) {
    return {std::cos(phase), std::sin(phase)};
}

// Pupillenfelder G aller transmittierenden Rasterpunkte (Eingangsfeld k, Komponente c) für die
// direkte Auswertung des Debye-Integrals, zeilenweise in aufsteigender Reihenfolge.
struct FieldTable {
    int grid = 0;
    int inputs = 0;
    double weight[kMaxVectorPsfInputs] = {};
    double alpha = 0.0; // 2π a_x / λ [1/mm]
    double beta = 0.0;  // 2π a_y / λ
    std::vector<int> column;
    std::vector<int> row;
    std::vector<Complex> g; // [Punkt · K · 3 + k · 3 + c]
};

FieldTable make_field_table(const VectorPsfPupil& vp, const VectorPsfSettings& settings) {
    const PsfPupil& p = vp.pupil;
    const VectorPsfParams params = make_vector_psf_params(vp, settings, 1.0);
    const VectorPsfInputs in = vector_psf_inputs(settings.polarization);
    FieldTable t;
    t.grid = p.grid;
    t.inputs = in.count;
    for (int k = 0; k < in.count; ++k) {
        t.weight[k] = in.jones[k].weight;
    }
    t.alpha = kPsfTwoPi * params.a_x / params.lambda_mm;
    t.beta = kPsfTwoPi * params.a_y / params.lambda_mm;
    const int n = p.grid;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const double opd = p.opd[to_size(j) * to_size(n) + to_size(i)];
            double re[kMaxVectorPsfInputs][3];
            double im[kMaxVectorPsfInputs][3];
            bool ok = true;
            for (int k = 0; k < in.count && ok; ++k) {
                ok = vector_psf_pupil_value(params, in.jones[k], cell_center(i, n),
                                            cell_center(j, n), opd, re[k], im[k]);
            }
            if (!ok) {
                continue;
            }
            t.column.push_back(i);
            t.row.push_back(j);
            for (int k = 0; k < in.count; ++k) {
                for (int c = 0; c < 3; ++c) {
                    t.g.emplace_back(re[k][c], im[k][c]);
                }
            }
        }
    }
    return t;
}

// Phasenfaktoren exp(i·f·x_i) der N Zellmitten.
std::vector<Complex> phase_factors(int n, double f, double position) {
    std::vector<Complex> out(to_size(n));
    for (int i = 0; i < n; ++i) {
        out[to_size(i)] = unit_phase(f * cell_center(i, n) * position);
    }
    return out;
}

// Intensität Σ_k w_k Σ_c |E_kc|² aus den Feldsummen e[k·3 + c] (nicht normiert).
double intensity_of(const FieldTable& t, const Complex* e, double* components) {
    double comp[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < 3; ++c) {
        double acc = 0.0;
        for (int k = 0; k < t.inputs; ++k) {
            const Complex v = e[k * 3 + c];
            acc += t.weight[k] * psf_norm2(v.real(), v.imag());
        }
        comp[c] = acc;
    }
    if (components != nullptr) {
        for (int c = 0; c < 3; ++c) {
            components[c] = comp[c];
        }
    }
    return (comp[0] + comp[1]) + comp[2];
}

// Direkte Auswertung im Punkt (x, y) [mm] (nicht normiert).
double table_intensity(const FieldTable& t, double x, double y, double* components) {
    const std::vector<Complex> col = phase_factors(t.grid, t.alpha, x);
    const std::vector<Complex> row = phase_factors(t.grid, t.beta, y);
    Complex e[kMaxVectorPsfInputs * 3] = {};
    const auto width = to_size(t.inputs) * 3;
    for (std::size_t q = 0; q < t.column.size(); ++q) {
        const Complex f = mul(row[to_size(t.row[q])], col[to_size(t.column[q])]);
        for (std::size_t m = 0; m < width; ++m) {
            e[m] += mul(f, t.g[q * width + m]);
        }
    }
    return intensity_of(t, e, components);
}

// Intensität entlang einer Geraden parallel zu x (axis = 0, feste Koordinate y0) oder zu y
// (axis = 1, fest x0): Summe über die Zeilen bzw. Spalten einmal vorab, dann O(N) pro Punkt.
struct LineEvaluator {
    const FieldTable* table = nullptr;
    int axis = 0;
    std::vector<Complex> h; // [Spalte bzw. Zeile · K·3 + m]

    LineEvaluator(const FieldTable& t, int line_axis, double fixed) : table(&t), axis(line_axis) {
        const auto width = to_size(t.inputs) * 3;
        h.assign(to_size(t.grid) * width, Complex(0.0, 0.0));
        const std::vector<Complex> other =
            phase_factors(t.grid, axis == 0 ? t.beta : t.alpha, fixed);
        for (std::size_t q = 0; q < t.column.size(); ++q) {
            const int along = axis == 0 ? t.column[q] : t.row[q];
            const int across = axis == 0 ? t.row[q] : t.column[q];
            const Complex f = other[to_size(across)];
            for (std::size_t m = 0; m < width; ++m) {
                h[to_size(along) * width + m] += mul(f, t.g[q * width + m]);
            }
        }
    }

    double operator()(double position) const {
        const FieldTable& t = *table;
        const auto width = to_size(t.inputs) * 3;
        const std::vector<Complex> f =
            phase_factors(t.grid, axis == 0 ? t.alpha : t.beta, position);
        Complex e[kMaxVectorPsfInputs * 3] = {};
        for (int i = 0; i < t.grid; ++i) {
            for (std::size_t m = 0; m < width; ++m) {
                e[m] += mul(f[to_size(i)], h[to_size(i) * width + m]);
            }
        }
        return intensity_of(t, e, nullptr);
    }
};

// Halbwertsbreite entlang einer Achse durch den Abtastpunkt (pi, pj) des Maximums: Abtastwerte
// bis unter die halbe Höhe, dann Bisektion auf dem direkt ausgewerteten Feld. NaN, wenn die
// halbe Höhe innerhalb des Bildfensters nicht unterschritten wird.
double full_width_half_maximum(const VectorPsfResult& r, const FieldTable& t, int axis, int pi,
                               int pj) {
    const int m = r.fft_size;
    const double step = axis == 0 ? r.dx : r.dy;
    const double x0 = (static_cast<double>(pi) - 0.5 * m) * r.dx;
    const double y0 = (static_cast<double>(pj) - 0.5 * m) * r.dy;
    const LineEvaluator line(t, axis, axis == 0 ? y0 : x0);
    const double center = axis == 0 ? x0 : y0;
    const double half = 0.5 * line(center) / r.reference_intensity;
    double edge[2] = {PsfResult::kNaN, PsfResult::kNaN};
    for (int side = 0; side < 2; ++side) {
        const int dir = side == 0 ? 1 : -1;
        for (int k = 1; k < m; ++k) {
            const int ii = axis == 0 ? pi + dir * k : pi;
            const int jj = axis == 0 ? pj : pj + dir * k;
            if (ii < 0 || jj < 0 || ii >= m || jj >= m) {
                break;
            }
            if (r.psf_at(ii, jj) < half) {
                double lo = static_cast<double>(k - 1);
                double hi = static_cast<double>(k);
                for (int it = 0; it < 60; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    if (line(center + dir * mid * step) / r.reference_intensity >= half) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                edge[side] = 0.5 * (lo + hi) * step;
                break;
            }
        }
    }
    return edge[0] + edge[1];
}

void check_vector_pupil(const VectorPsfPupil& vp) {
    if (!(vp.index > 0.0) || !std::isfinite(vp.index)) {
        throw std::invalid_argument("Vektorielle PSF: Brechzahl des Bildraums muss positiv sein");
    }
    if (!std::isfinite(vp.chief_x) || !std::isfinite(vp.chief_y) ||
        vp.chief_x * vp.chief_x + vp.chief_y * vp.chief_y >= 1.0) {
        throw std::invalid_argument("Vektorielle PSF: ungültige Richtung des Hauptstrahls");
    }
    const double u2 = vp.input_y * vp.input_y + vp.input_z * vp.input_z;
    if (!(vp.input_z > 0.0) || !(std::abs(u2 - 1.0) < 1e-12)) {
        throw std::invalid_argument(
            "Vektorielle PSF: Richtung des Eingangsbündels muss ein Einheitsvektor mit u_z > 0 "
            "sein");
    }
}

} // namespace

const char* polarization_name(Polarization polarization) {
    switch (polarization) {
    case Polarization::X:
        return "x";
    case Polarization::Y:
        return "y";
    case Polarization::Circular:
        return "circular";
    case Polarization::Unpolarized:
        return "unpolarized";
    }
    return "?";
}

Polarization parse_polarization(std::string_view name) {
    for (const Polarization p :
         {Polarization::X, Polarization::Y, Polarization::Circular, Polarization::Unpolarized}) {
        if (name == polarization_name(p)) {
            return p;
        }
    }
    throw std::invalid_argument("Polarisation '" + std::string(name) +
                                "' unbekannt (x, y, circular, unpolarized)");
}

VectorPsfInputs vector_psf_inputs(Polarization polarization) {
    VectorPsfInputs in;
    const VectorPsfJones x{{1.0, 0.0}, {0.0, 0.0}, 1.0};
    const VectorPsfJones y{{0.0, 1.0}, {0.0, 0.0}, 1.0};
    switch (polarization) {
    case Polarization::X:
        in.count = 1;
        in.jones[0] = x;
        break;
    case Polarization::Y:
        in.count = 1;
        in.jones[0] = y;
        break;
    case Polarization::Circular: {
        const double h = std::numbers::sqrt2 / 2.0;
        in.count = 1;
        in.jones[0] = VectorPsfJones{{h, 0.0}, {0.0, h}, 1.0};
        break;
    }
    case Polarization::Unpolarized:
        in.count = 2;
        in.jones[0] = x;
        in.jones[1] = y;
        in.jones[0].weight = 0.5;
        in.jones[1].weight = 0.5;
        break;
    }
    return in;
}

void check_vector_psf_settings(const VectorPsfSettings& settings) {
    if (!std::isfinite(settings.defocus)) {
        throw std::invalid_argument("Vektorielle PSF: Defokus muss endlich sein");
    }
    (void)polarization_name(settings.polarization);
}

VectorPsfPupil make_vector_psf_pupil(const SystemBatchHost& batch, const EvalResult& result, int v,
                                     int f, int l, const PsfSettings& settings) {
    VectorPsfPupil vp;
    vp.pupil = make_psf_pupil(batch, result, v, f, l, settings);
    vp.index = batch.index(v, batch.surfaces() - 1, l);
    if (result.field_ray_status(v, f, l, kEvalChiefRay) == TraceStatus::Ok) {
        vp.chief_x = result.field_ray(v, f, l, kEvalChiefRay, 2);
        vp.chief_y = result.field_ray(v, f, l, kEvalChiefRay, 3);
    }
    if (batch.source() == SourceKind::Collimated) {
        const double theta = batch.field_angle(v, f);
        vp.input_y = std::sin(theta);
        vp.input_z = std::cos(theta);
    }
    check_vector_pupil(vp);
    return vp;
}

std::vector<VectorPsfPupil> make_vector_psf_pupils(const SystemBatchHost& batch,
                                                   const EvalResult& result, int v,
                                                   const PsfSettings& settings) {
    std::vector<VectorPsfPupil> out;
    out.reserve(static_cast<std::size_t>(result.fields) *
                static_cast<std::size_t>(result.wavelengths));
    for (int f = 0; f < result.fields; ++f) {
        for (int l = 0; l < result.wavelengths; ++l) {
            out.push_back(make_vector_psf_pupil(batch, result, v, f, l, settings));
        }
    }
    return out;
}

VectorPsfParams make_vector_psf_params(const VectorPsfPupil& vp, const VectorPsfSettings& settings,
                                       double reference_intensity) {
    const PsfPupil& p = vp.pupil;
    VectorPsfParams q{};
    q.lambda_mm = p.wavelength_mm();
    q.scale = 1.0 / reference_intensity;
    q.a_x = p.flip_x ? p.na_x : -p.na_x;
    q.a_y = p.flip_y ? p.na_y : -p.na_y;
    q.index = vp.index;
    q.chief_x = vp.chief_x;
    q.chief_y = vp.chief_y;
    q.input_y = vp.input_y;
    q.input_z = vp.input_z;
    q.defocus = settings.defocus;
    q.sign_x = p.flip_x ? -1 : 1;
    q.sign_y = p.flip_y ? -1 : 1;
    return q;
}

double vector_psf_reference_intensity(const VectorPsfPupil& vp, Polarization polarization) {
    const PsfPupil& p = vp.pupil;
    if (p.grid < 1 || p.opd.size() != to_size(p.grid) * to_size(p.grid)) {
        throw std::invalid_argument("Vektorielle PSF: Pupillenkarte hat nicht N·N Werte");
    }
    check_vector_pupil(vp);
    VectorPsfSettings s;
    s.polarization = polarization;
    const VectorPsfParams params = make_vector_psf_params(vp, s, 1.0);
    const VectorPsfInputs in = vector_psf_inputs(polarization);
    const int n = p.grid;
    Complex e[kMaxVectorPsfInputs * 3] = {};
    bool any = false;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (!in_pupil(i, j, n)) {
                continue;
            }
            for (int k = 0; k < in.count; ++k) {
                double re[3];
                double im[3];
                if (vector_psf_pupil_value(params, in.jones[k], cell_center(i, n),
                                           cell_center(j, n), 0.0, re, im)) {
                    any = true;
                    for (int c = 0; c < 3; ++c) {
                        e[k * 3 + c] += Complex(re[c], im[c]);
                    }
                }
            }
        }
    }
    double total = 0.0;
    for (int c = 0; c < 3; ++c) {
        double acc = 0.0;
        for (int k = 0; k < in.count; ++k) {
            acc += in.jones[k].weight * psf_norm2(e[k * 3 + c].real(), e[k * 3 + c].imag());
        }
        total += acc;
    }
    if (!any || !(total > 0.0)) {
        throw std::invalid_argument(
            "Vektorielle PSF: kein ausbreitungsfähiger Rasterpunkt (|s_⊥| >= 1 überall)");
    }
    return total;
}

VectorPsfResult prepare_vector_psf_result(const VectorPsfPupil& vp, int fft_size,
                                          const VectorPsfSettings& settings) {
    check_vector_psf_settings(settings);
    VectorPsfResult r;
    static_cast<PsfResult&>(r) = prepare_psf_result(vp.pupil, fft_size);
    r.polarization = settings.polarization;
    r.defocus = settings.defocus;
    r.reference_intensity = vector_psf_reference_intensity(vp, settings.polarization);
    for (std::vector<double>& c : r.component) {
        c.assign(r.psf.size(), 0.0);
    }
    return r;
}

double vector_psf_intensity_at(const VectorPsfPupil& vp, const VectorPsfSettings& settings,
                               double reference_intensity, double x, double y,
                               std::array<double, 3>* components) {
    check_vector_psf_settings(settings);
    check_vector_pupil(vp);
    const FieldTable t = make_field_table(vp, settings);
    double comp[3];
    const double total = table_intensity(t, x, y, comp);
    if (components != nullptr) {
        for (int c = 0; c < 3; ++c) {
            (*components)[static_cast<std::size_t>(c)] = comp[c] / reference_intensity;
        }
    }
    return total / reference_intensity;
}

void finish_vector_psf(VectorPsfResult& r, const VectorPsfPupil& vp) {
    const PsfPupil& p = vp.pupil;
    const auto m = to_size(r.fft_size);
    if (r.grid != p.grid || r.psf.size() != m * m) {
        throw std::invalid_argument("Vektorielle PSF: Ergebnis passt nicht zur Pupille");
    }
    for (const std::vector<double>& c : r.component) {
        if (c.size() != m * m) {
            throw std::invalid_argument("Vektorielle PSF: Komponenten fehlen");
        }
    }
    VectorPsfSettings settings;
    settings.polarization = r.polarization;
    settings.defocus = r.defocus;
    const VectorPsfParams params = make_vector_psf_params(vp, settings, 1.0);
    const int n = p.grid;

    // Transmission, ausbreitungsfähige Punkte und Σ W² der vollen Pupille
    double weight2_full = 0.0;
    r.transmitted = 0;
    r.evanescent = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (!in_pupil(i, j, n)) {
                continue;
            }
            double ex[3];
            double ey[3];
            double w = 0.0;
            double dz = 0.0;
            if (!vector_psf_basis(params, cell_center(i, n), cell_center(j, n), ex, ey, w, dz)) {
                ++r.evanescent;
                continue;
            }
            weight2_full += w * w;
            if (std::isfinite(p.opd[to_size(j) * to_size(n) + to_size(i)])) {
                ++r.transmitted;
            }
        }
    }

    // Summen über die Bildebene (sequenziell), Maxima
    double sum = 0.0;
    double sum_c[3] = {0.0, 0.0, 0.0};
    std::size_t peak_index = 0;
    for (std::size_t k = 0; k < r.psf.size(); ++k) {
        sum += r.psf[k];
        if (r.psf[k] > r.psf[peak_index]) {
            peak_index = k;
        }
    }
    for (std::size_t c = 0; c < 3; ++c) {
        double peak_c = r.component[c][0];
        for (const double v : r.component[c]) {
            sum_c[c] += v;
            peak_c = std::max(peak_c, v);
        }
        r.component_peak[c] = peak_c;
        r.energy_fraction[c] = sum > 0.0 ? sum_c[c] / sum : PsfResult::kNaN;
    }
    const double md = static_cast<double>(m);
    // Parseval: Σ_pq |E|² = M² Σ_k w_k Σ_p W_p² |J_k|² (Rotation normerhaltend, Σ w_k |J_k|² = 1)
    r.energy = sum * r.reference_intensity / (md * md * weight2_full);
    r.strehl_fft = r.psf[(m / 2) * m + m / 2];
    r.peak = r.psf[peak_index];
    r.peak_x = (static_cast<double>(peak_index % m) - md / 2.0) * r.dx;
    r.peak_y = (static_cast<double>(peak_index / m) - md / 2.0) * r.dy;

    const FieldTable t = make_field_table(vp, settings);
    r.strehl_origin = table_intensity(t, 0.0, 0.0, nullptr) / r.reference_intensity;
    double tilt_x = 0.0;
    double tilt_y = 0.0;
    if (psf_tilt_fit(p, tilt_x, tilt_y)) {
        r.reference_x = (p.flip_x ? -tilt_x : tilt_x) / p.na_x;
        r.reference_y = (p.flip_y ? -tilt_y : tilt_y) / p.na_y;
        r.strehl =
            table_intensity(t, r.reference_x, r.reference_y, nullptr) / r.reference_intensity;
    } else {
        r.strehl = PsfResult::kNaN;
        r.reference_x = PsfResult::kNaN;
        r.reference_y = PsfResult::kNaN;
    }
    const int pi = static_cast<int>(peak_index % m);
    const int pj = static_cast<int>(peak_index / m);
    r.fwhm_x = full_width_half_maximum(r, t, 0, pi, pj);
    r.fwhm_y = full_width_half_maximum(r, t, 1, pi, pj);
}

VectorPsfResult vector_psf_cpu(const VectorPsfPupil& vp, int fft_size,
                               const VectorPsfSettings& settings) {
    VectorPsfResult r = prepare_vector_psf_result(vp, fft_size, settings);
    const PsfPupil& pupil = vp.pupil;
    const int n = pupil.grid;
    const auto m = to_size(fft_size);
    const auto mi = static_cast<std::int64_t>(m);
    const VectorPsfParams params = make_vector_psf_params(vp, settings, r.reference_intensity);
    const VectorPsfInputs in = vector_psf_inputs(settings.polarization);
    const int sx = params.sign_x;
    const int sy = params.sign_y;

    // acc[c][q]: Σ_k w_k |E_kc|² am FFT-Index q (k aufsteigend)
    std::vector<std::vector<double>> acc(3, std::vector<double>(m * m, 0.0));
    std::vector<std::vector<Complex>> buf(3);
    for (int k = 0; k < in.count; ++k) {
        for (std::vector<Complex>& b : buf) {
            b.assign(m * m, Complex(0.0, 0.0));
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                double re[3];
                double im[3];
                vector_psf_pupil_value(params, in.jones[k], cell_center(i, n), cell_center(j, n),
                                       pupil.opd[to_size(j) * to_size(n) + to_size(i)], re, im);
                for (std::size_t c = 0; c < 3; ++c) {
                    buf[c][to_size(j) * m + to_size(i)] = {re[c], im[c]};
                }
            }
        }
        for (std::size_t c = 0; c < 3; ++c) {
            fft2d(buf[c], fft_size, fft_size, FftDirection::Forward);
            const double w = in.jones[k].weight;
            for (std::size_t q = 0; q < m * m; ++q) {
                acc[c][q] += w * psf_norm2(buf[c][q].real(), buf[c][q].imag());
            }
        }
    }

    // Normierung, Zentrierung (Spiegelung bei divergenter Welle), Intensität reell für die MTF
    std::vector<Complex>& otf = buf[0];
    for (std::size_t qj = 0; qj < m; ++qj) {
        const auto oj = to_size(psf_output_index(static_cast<std::int64_t>(qj), mi, sy));
        for (std::size_t qi = 0; qi < m; ++qi) {
            const std::size_t q = qj * m + qi;
            const auto oi = to_size(psf_output_index(static_cast<std::int64_t>(qi), mi, sx));
            const double cx = acc[0][q] * params.scale;
            const double cy = acc[1][q] * params.scale;
            const double cz = acc[2][q] * params.scale;
            const double total = (cx + cy) + cz;
            const std::size_t o = oj * m + oi;
            r.component[0][o] = cx;
            r.component[1][o] = cy;
            r.component[2][o] = cz;
            r.psf[o] = total;
            otf[q] = {total, 0.0};
        }
    }
    fft2d(otf, fft_size, fft_size, FftDirection::Inverse);
    const double otf0 = otf[0].real();
    for (int k = 0; k <= n; ++k) {
        const Complex s = otf[to_size(k)];
        const Complex t = otf[to_size(k) * m];
        r.mtf_sagittal[to_size(k)] =
            otf0 > 0.0 ? msqrt(psf_norm2(s.real(), s.imag())) / otf0 : PsfResult::kNaN;
        r.mtf_tangential[to_size(k)] =
            otf0 > 0.0 ? msqrt(psf_norm2(t.real(), t.imag())) / otf0 : PsfResult::kNaN;
    }
    finish_vector_psf(r, vp);
    return r;
}

std::vector<VectorPsfResult> vector_psf_cpu(const SystemBatchHost& batch, const EvalResult& result,
                                            int v, const PsfSettings& psf_settings,
                                            const VectorPsfSettings& settings) {
    const std::vector<VectorPsfPupil> pupils =
        make_vector_psf_pupils(batch, result, v, psf_settings);
    const int m = psf_fft_size(result.grid.size, psf_settings.padding);
    std::vector<VectorPsfResult> out;
    out.reserve(pupils.size());
    for (const VectorPsfPupil& p : pupils) {
        out.push_back(vector_psf_cpu(p, m, settings));
    }
    return out;
}

VectorPsfDeviceLayout make_vector_psf_device_layout(int grid, int fft_size, int inputs,
                                                    int pupils) {
    if (grid < 1 || fft_size < 2 * grid || inputs < 1 || inputs > kMaxVectorPsfInputs ||
        pupils < 1) {
        throw std::invalid_argument("Vektorielle PSF: ungültige Größen für die Gerätepuffer");
    }
    const auto t = static_cast<std::size_t>(pupils);
    const std::size_t m2 = to_size(fft_size) * to_size(fft_size);
    const std::size_t n2 = to_size(grid) * to_size(grid);
    VectorPsfDeviceLayout l;
    std::size_t offset = 0;
    const auto place = [&offset](std::size_t bytes) {
        const BufferRegion r{offset, bytes};
        offset += align_bytes(bytes);
        return r;
    };
    l.spectrum = place(t * static_cast<std::size_t>(inputs) * 3 * m2 * 2 * sizeof(double));
    l.psf = place(t * m2 * sizeof(double));
    l.component = place(t * 3 * m2 * sizeof(double));
    l.opd = place(t * n2 * sizeof(double));
    l.params = place(t * sizeof(VectorPsfParams));
    l.mtf = place(t * 2 * (to_size(grid) + 1) * sizeof(double));
    l.total_bytes = offset;
    return l;
}

} // namespace mko
