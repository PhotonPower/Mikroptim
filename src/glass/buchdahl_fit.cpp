// Kleinste-Quadrate-Fit des Buchdahl-Modells (docs/numerik.md, Abschnitt 10).

#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/glass/dispersion.hpp"

#include "format_util.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>

namespace mko {

namespace {

void validate_options(const BuchdahlFitOptions& o) {
    const double lo = o.lambda_min_nm;
    const double hi = o.lambda_max_nm;
    if (!(std::isfinite(lo) && std::isfinite(hi) && lo < hi)) {
        throw GlassError("Buchdahl-Fit: ungültiger Bereich [" + detail::format_number(lo) + ", " +
                         detail::format_number(hi) + "] nm (endlich, min < max erwartet)");
    }
    if (!(buchdahl_denominator(nm_to_um(lo)) > 0.0)) {
        throw GlassError("Buchdahl-Fit: Buchdahl-Koordinate ist bei " + detail::format_number(lo) +
                         " nm nicht definiert (erfordert lambda > 146,074 nm)");
    }
    if (o.samples < 2) {
        throw GlassError("Buchdahl-Fit: mindestens 2 Stützstellen erwartet, nicht " +
                         std::to_string(o.samples));
    }
    if (o.check_samples < 2) {
        throw GlassError("Buchdahl-Fit: Prüfgitter braucht mindestens 2 Punkte, nicht " +
                         std::to_string(o.check_samples));
    }
}

double checked_index(const IndexFunction& index_nm, double lambda_nm) {
    const double n = index_nm(lambda_nm);
    if (!std::isfinite(n)) {
        throw GlassError("Buchdahl-Fit: Brechzahl bei " + detail::format_number(lambda_nm) +
                         " nm ist nicht endlich");
    }
    return n;
}

// Löst min ||A x - b||_2 per Householder-QR. A ist m x p, spaltenweise abgelegt
// (a[j * m + i]), m >= p. A und b werden überschrieben; das Ergebnis steht in x[0..p).
void householder_least_squares(std::vector<double>& a, std::vector<double>& b, int m, int p,
                               double* x) {
    const auto mm = static_cast<std::size_t>(m);
    auto at = [&](int i, int j) -> double& {
        return a[static_cast<std::size_t>(j) * mm + static_cast<std::size_t>(i)];
    };
    for (int k = 0; k < p; ++k) {
        double norm2 = 0.0;
        for (int i = k; i < m; ++i) {
            norm2 += at(i, k) * at(i, k);
        }
        const double norm = std::sqrt(norm2);
        if (!(norm > 0.0)) {
            throw GlassError("Buchdahl-Fit: Ausgleichsproblem hat keinen vollen Rang");
        }
        // Spiegelung von Spalte k auf alpha e_k; Vorzeichen gegen Auslöschung in v_k.
        const double alpha = at(k, k) > 0.0 ? -norm : norm;
        at(k, k) -= alpha; // v = (a_kk - alpha, a_k+1,k, ..., a_m-1,k)
        double vtv = 0.0;
        for (int i = k; i < m; ++i) {
            vtv += at(i, k) * at(i, k);
        }
        // H = I - 2 v v^T / (v^T v) auf die restlichen Spalten und auf b anwenden.
        for (int j = k + 1; j < p; ++j) {
            double s = 0.0;
            for (int i = k; i < m; ++i) {
                s += at(i, k) * at(i, j);
            }
            const double f = 2.0 * s / vtv;
            for (int i = k; i < m; ++i) {
                at(i, j) -= f * at(i, k);
            }
        }
        double s = 0.0;
        for (int i = k; i < m; ++i) {
            s += at(i, k) * b[static_cast<std::size_t>(i)];
        }
        const double f = 2.0 * s / vtv;
        for (int i = k; i < m; ++i) {
            b[static_cast<std::size_t>(i)] -= f * at(i, k);
        }
        at(k, k) = alpha; // Diagonale von R
    }
    // Rückwärtseinsetzen R x = (Q^T b)[0..p).
    for (int k = p - 1; k >= 0; --k) {
        double s = b[static_cast<std::size_t>(k)];
        for (int j = k + 1; j < p; ++j) {
            s -= at(k, j) * x[j];
        }
        x[k] = s / at(k, k);
    }
}

} // namespace

double BuchdahlFit::index(double lambda_nm) const {
    return buchdahl_index(n0, nu[0], nu[1], nu[2], nm_to_um(lambda_nm));
}

std::vector<double> buchdahl_fit_nodes_nm(const BuchdahlFitOptions& options) {
    validate_options(options);
    const int n = options.samples;
    const double lo = options.lambda_min_nm;
    const double hi = options.lambda_max_nm;
    std::vector<double> nodes(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        double lambda = 0.0;
        if (options.nodes == BuchdahlFitNodes::Chebyshev) {
            // lambda_k = m - h cos(pi (k + 1/2) / n), aufsteigend, ohne Randpunkte.
            const double mid = 0.5 * (lo + hi);
            const double half = 0.5 * (hi - lo);
            lambda = mid - half * std::cos(std::numbers::pi * (k + 0.5) / n);
        } else {
            lambda = k == n - 1 ? hi : lo + (hi - lo) * k / (n - 1);
        }
        nodes[static_cast<std::size_t>(k)] = lambda;
    }
    return nodes;
}

std::vector<double> buchdahl_check_grid_nm(const BuchdahlFitOptions& options) {
    validate_options(options);
    const int n = options.check_samples;
    const double lo = options.lambda_min_nm;
    const double hi = options.lambda_max_nm;
    std::vector<double> grid(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        grid[static_cast<std::size_t>(i)] = i == n - 1 ? hi : lo + (hi - lo) * i / (n - 1);
    }
    return grid;
}

BuchdahlFit fit_buchdahl(const IndexFunction& index_nm, int dispersion_terms,
                         const BuchdahlFitOptions& options) {
    if (dispersion_terms != 2 && dispersion_terms != 3) {
        throw GlassError("Buchdahl-Fit: 2 oder 3 Dispersionsterme erwartet, nicht " +
                         std::to_string(dispersion_terms));
    }
    if (!index_nm) {
        throw GlassError("Buchdahl-Fit: keine Brechzahlfunktion angegeben");
    }
    const int p = dispersion_terms + 1; // Unbekannte n0, nu1, ..., nu_terms
    const std::vector<double> nodes = buchdahl_fit_nodes_nm(options);
    const int m = options.samples;
    if (m < p) {
        throw GlassError("Buchdahl-Fit: " + std::to_string(m) + " Stützstellen reichen nicht für " +
                         std::to_string(p) + " Unbekannte");
    }

    // Entwurfsmatrix A[i][j] = w_i^j (Vandermonde in w), rechte Seite b_i = n(lambda_i).
    std::vector<double> a(static_cast<std::size_t>(m) * static_cast<std::size_t>(p));
    std::vector<double> b(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        const double lambda_nm = nodes[static_cast<std::size_t>(i)];
        const double w = buchdahl_omega(nm_to_um(lambda_nm));
        double wj = 1.0;
        for (int j = 0; j < p; ++j) {
            a[static_cast<std::size_t>(j) * static_cast<std::size_t>(m) +
              static_cast<std::size_t>(i)] = wj;
            wj *= w;
        }
        b[static_cast<std::size_t>(i)] = checked_index(index_nm, lambda_nm);
    }
    double x[4] = {0.0, 0.0, 0.0, 0.0};
    householder_least_squares(a, b, m, p, x);

    BuchdahlFit fit;
    fit.dispersion_terms = dispersion_terms;
    fit.n0 = x[0];
    fit.nu = {x[1], x[2], x[3]};
    fit.options = options;

    // Fehler auf dem Prüfgitter mit derselben Auswertung wie material_index (buchdahl_index).
    double max_err = 0.0;
    double sum_sq = 0.0;
    const std::vector<double> grid = buchdahl_check_grid_nm(options);
    for (const double lambda_nm : grid) {
        const double e = fit.index(lambda_nm) - checked_index(index_nm, lambda_nm);
        max_err = std::fmax(max_err, std::fabs(e));
        sum_sq += e * e;
    }
    fit.max_abs_error = max_err;
    fit.rms_error = std::sqrt(sum_sq / static_cast<double>(grid.size()));
    return fit;
}

BuchdahlFit fit_buchdahl(const Glass& glass, int dispersion_terms,
                         const BuchdahlFitOptions& options) {
    const std::string prefix = "Glas " + glass.name + ": ";
    if (!(glass.in_range(options.lambda_min_nm) && glass.in_range(options.lambda_max_nm))) {
        throw GlassError(prefix + "Fitbereich [" + detail::format_number(options.lambda_min_nm) +
                         ", " + detail::format_number(options.lambda_max_nm) +
                         "] nm liegt nicht im Gültigkeitsbereich [" +
                         detail::format_number(glass.range_nm[0]) + ", " +
                         detail::format_number(glass.range_nm[1]) + "] nm");
    }
    try {
        return fit_buchdahl([&glass](double lambda_nm) { return glass.index(lambda_nm); },
                            dispersion_terms, options);
    } catch (const GlassError& e) {
        throw GlassError(prefix + e.what());
    }
}

MaterialSpec to_buchdahl_material(const BuchdahlFit& fit) {
    MaterialSpec m;
    m.kind = MaterialKind::Buchdahl;
    m.n0 = fit.n0;
    m.nu = fit.nu;
    return m;
}

} // namespace mko
