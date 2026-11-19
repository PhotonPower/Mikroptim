// CMA-ES für die memetische Suche (M11-T7, docs/optimierung.md, Abschnitt 15). Nur Operationen,
// die IEEE 754 korrekt gerundet festlegt (+, −, ·, /, √) und exakte Hilfsfunktionen (floor,
// frexp, ldexp); keine Funktionen der Mathematikbibliothek mit plattformabhängiger Rundung.

#include "mikroptim/optim/cma_es.hpp"

#include "mikroptim/optim/variant_generator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("CMA-ES: " + message);
}

// ln 2 = kLn2Hi + kLn2Lo; kLn2Hi hat 32 Mantissenbits, k·kLn2Hi ist für |k| ≤ 2¹⁰ exakt
// (Cody-Waite-Reduktion, Werte wie fdlibm).
constexpr double kLn2Hi = 6.93147180369123816490e-01;
constexpr double kLn2Lo = 1.90821492927058770002e-10;
constexpr double kLog2E = 1.44269504088896338700e+00;
constexpr double kSqrtHalf = 0.70710678118654752440;

std::size_t idx(int i, int j, int n) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i);
}

} // namespace

double portable_exp(double x) {
    if (std::isnan(x)) {
        return x;
    }
    if (x > 709.782712893384) {
        return std::numeric_limits<double>::infinity();
    }
    if (x < -745.2) {
        return 0.0;
    }
    const double k = std::floor(x * kLog2E + 0.5);
    const double r = (x - k * kLn2Hi) - k * kLn2Lo;
    // e^r = 1 + r (1 + r/2 (1 + r/3 (… (1 + r/20)))), |r| ≤ 0,35: Restglied < 10⁻²¹
    double p = 1.0;
    for (int j = 20; j >= 1; --j) {
        p = 1.0 + (r / static_cast<double>(j)) * p;
    }
    return std::ldexp(p, static_cast<int>(k));
}

double portable_log(double x) {
    if (std::isnan(x) || x < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (std::isinf(x)) {
        return x;
    }
    int e = 0;
    double m = std::frexp(x, &e); // x = m·2^e, m ∈ [0,5; 1)
    if (m < kSqrtHalf) {
        m *= 2.0;
        e -= 1;
    }
    // ln m = 2 artanh(s) = 2 (s + s³/3 + s⁵/5 + …), |s| ≤ 0,1716, s² ≤ 0,0295
    const double s = (m - 1.0) / (m + 1.0);
    const double s2 = s * s;
    double t = 1.0 / 31.0;
    for (int j = 29; j >= 1; j -= 2) {
        t = 1.0 / static_cast<double>(j) + s2 * t;
    }
    const double lm = 2.0 * s * t;
    const double ed = static_cast<double>(e);
    return ed * kLn2Hi + (ed * kLn2Lo + lm);
}

void symmetric_eigen(int n, std::span<double> a, std::span<double> values,
                     std::span<double> vectors) {
    if (n < 1) {
        invalid("symmetric_eigen: n muss >= 1 sein");
    }
    const auto nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() != nn || vectors.size() != nn || values.size() != static_cast<std::size_t>(n)) {
        invalid("symmetric_eigen: falsche Längen");
    }
    std::fill(vectors.begin(), vectors.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        vectors[idx(i, i, n)] = 1.0;
    }
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                off += std::abs(a[idx(p, q, n)]);
            }
        }
        if (off == 0.0) {
            break;
        }
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[idx(p, q, n)];
                if (apq == 0.0) {
                    continue;
                }
                const double app = a[idx(p, p, n)];
                const double aqq = a[idx(q, q, n)];
                // Nach einigen Durchgängen: vernachlässigbare Elemente (unter der Rundung der
                // Diagonale) zu 0 setzen (Rutishauser)
                const double g = 100.0 * std::abs(apq);
                if (sweep > 3 && std::abs(app) + g == std::abs(app) &&
                    std::abs(aqq) + g == std::abs(aqq)) {
                    a[idx(p, q, n)] = 0.0;
                    a[idx(q, p, n)] = 0.0;
                    continue;
                }
                // t = tan φ als betragskleinere Wurzel von t² + 2θt − 1 = 0
                const double theta = (aqq - app) / (2.0 * apq);
                double t = 0.0;
                if (std::abs(theta) > 1e150) {
                    t = 0.5 / theta;
                } else {
                    t = 1.0 / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                    if (theta < 0.0) {
                        t = -t;
                    }
                }
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                // A ← A J (Spalten p, q), dann A ← Jᵀ A (Zeilen p, q), V ← V J
                for (int k = 0; k < n; ++k) {
                    const double akp = a[idx(k, p, n)];
                    const double akq = a[idx(k, q, n)];
                    a[idx(k, p, n)] = c * akp - s * akq;
                    a[idx(k, q, n)] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = a[idx(p, k, n)];
                    const double aqk = a[idx(q, k, n)];
                    a[idx(p, k, n)] = c * apk - s * aqk;
                    a[idx(q, k, n)] = s * apk + c * aqk;
                }
                a[idx(p, q, n)] = 0.0;
                a[idx(q, p, n)] = 0.0;
                for (int k = 0; k < n; ++k) {
                    const double vkp = vectors[idx(k, p, n)];
                    const double vkq = vectors[idx(k, q, n)];
                    vectors[idx(k, p, n)] = c * vkp - s * vkq;
                    vectors[idx(k, q, n)] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        values[static_cast<std::size_t>(i)] = a[idx(i, i, n)];
    }
}

int cma_default_lambda(int dimension) {
    if (dimension < 1) {
        invalid("Dimension muss >= 1 sein");
    }
    return 4 + static_cast<int>(std::floor(3.0 * portable_log(static_cast<double>(dimension))));
}

CmaParameters cma_parameters(int dimension, int lambda) {
    if (dimension < 1) {
        invalid("Dimension muss >= 1 sein");
    }
    if (lambda == 0) {
        lambda = cma_default_lambda(dimension);
    }
    if (lambda < 2) {
        invalid("Populationsgröße lambda muss >= 2 sein (0: Standard)");
    }
    CmaParameters p;
    const double n = static_cast<double>(dimension);
    p.dimension = dimension;
    p.lambda = lambda;
    p.mu = lambda / 2;
    const double head = portable_log((static_cast<double>(lambda) + 1.0) / 2.0);
    p.weights.resize(static_cast<std::size_t>(p.mu));
    double sum = 0.0;
    for (int i = 0; i < p.mu; ++i) {
        const double w = head - portable_log(static_cast<double>(i + 1));
        p.weights[static_cast<std::size_t>(i)] = w;
        sum += w;
    }
    double sum2 = 0.0;
    for (double& w : p.weights) {
        w /= sum;
        sum2 += w * w;
    }
    const double me = 1.0 / sum2;
    p.mu_eff = me;
    p.c_sigma = (me + 2.0) / (n + me + 5.0);
    p.d_sigma = 1.0 + 2.0 * std::max(0.0, std::sqrt((me - 1.0) / (n + 1.0)) - 1.0) + p.c_sigma;
    p.c_c = (4.0 + me / n) / (n + 4.0 + 2.0 * me / n);
    p.c_1 = 2.0 / ((n + 1.3) * (n + 1.3) + me);
    p.c_mu = std::min(1.0 - p.c_1, 2.0 * (me - 2.0 + 1.0 / me) / ((n + 2.0) * (n + 2.0) + me));
    p.chi_n = std::sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
    p.clip_norm = std::sqrt(n) + 2.0 * n / (n + 2.0);
    return p;
}

CmaEs::CmaEs(std::span<const double> mean, double sigma, int lambda)
    : params_(cma_parameters(static_cast<int>(mean.size()), lambda)), sigma_(sigma),
      mean_(mean.begin(), mean.end()) {
    for (const double m : mean_) {
        if (!std::isfinite(m)) {
            invalid("Mittelwert nicht endlich");
        }
    }
    if (!(std::isfinite(sigma) && sigma > 0.0)) {
        invalid("sigma muss endlich und > 0 sein");
    }
    const int n = params_.dimension;
    const auto nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    c_.assign(nn, 0.0);
    b_.assign(nn, 0.0);
    for (int i = 0; i < n; ++i) {
        c_[idx(i, i, n)] = 1.0;
        b_[idx(i, i, n)] = 1.0;
    }
    d_.assign(static_cast<std::size_t>(n), 1.0);
    p_sigma_.assign(static_cast<std::size_t>(n), 0.0);
    p_c_.assign(static_cast<std::size_t>(n), 0.0);
}

void CmaEs::sample(std::uint64_t seed, std::uint64_t i, std::span<double> out) const {
    const int n = params_.dimension;
    if (out.size() != static_cast<std::size_t>(n)) {
        invalid("sample: Ausgabe hat falsche Länge");
    }
    std::mt19937_64 gen(
        variant_seed(variant_seed(seed, static_cast<std::uint64_t>(generation_)), i));
    std::vector<double> z(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        double s = 0.0;
        for (int j = 0; j < 12; ++j) {
            s += unit_uniform(gen());
        }
        z[static_cast<std::size_t>(k)] = (s - 6.0) * d_[static_cast<std::size_t>(k)];
    }
    for (int r = 0; r < n; ++r) {
        double y = 0.0;
        for (int k = 0; k < n; ++k) {
            y += b_[idx(r, k, n)] * z[static_cast<std::size_t>(k)];
        }
        out[static_cast<std::size_t>(r)] = mean_[static_cast<std::size_t>(r)] + sigma_ * y;
    }
}

double CmaEs::mahalanobis_norm(std::span<const double> v) const {
    const int n = params_.dimension;
    if (v.size() != static_cast<std::size_t>(n)) {
        invalid("mahalanobis_norm: falsche Länge");
    }
    // ‖C^{−1/2} v‖ = ‖D⁻¹ Bᵀ v‖ (B orthonormal)
    double s = 0.0;
    for (int k = 0; k < n; ++k) {
        double u = 0.0;
        for (int r = 0; r < n; ++r) {
            u += b_[idx(r, k, n)] * v[static_cast<std::size_t>(r)];
        }
        u /= d_[static_cast<std::size_t>(k)];
        s += u * u;
    }
    return std::sqrt(s);
}

void CmaEs::update(std::span<const double> points, std::span<const int> ranking,
                   std::span<const double> sigma_points) {
    const int n = params_.dimension;
    const int lambda = params_.lambda;
    const int mu = params_.mu;
    const auto un = static_cast<std::size_t>(n);
    if (points.size() != static_cast<std::size_t>(lambda) * un) {
        invalid("update: " + std::to_string(points.size()) +
                " Werte statt lambda·n = " + std::to_string(static_cast<std::size_t>(lambda) * un));
    }
    if (ranking.size() < static_cast<std::size_t>(mu) ||
        ranking.size() > static_cast<std::size_t>(lambda)) {
        invalid("update: Ranking braucht mu bis lambda Einträge");
    }
    std::vector<char> seen(static_cast<std::size_t>(lambda), 0);
    for (const int r : ranking) {
        if (r < 0 || r >= lambda || seen[static_cast<std::size_t>(r)] != 0) {
            invalid("update: ungültiger oder doppelter Index im Ranking");
        }
        seen[static_cast<std::size_t>(r)] = 1;
    }
    for (const double x : points) {
        if (!std::isfinite(x)) {
            invalid("update: Punkt nicht endlich");
        }
    }
    if (!sigma_points.empty() && sigma_points.size() != points.size()) {
        invalid("update: sigma_points muss leer sein oder lambda·n Werte haben");
    }
    for (const double x : sigma_points) {
        if (!std::isfinite(x)) {
            invalid("update: Punkt nicht endlich");
        }
    }

    // Schritte der μ besten Punkte, auf die Mahalanobis-Länge clip_norm begrenzt, und ihre
    // gewichtete Summe
    const auto steps = [&](std::span<const double> pts, std::vector<double>& y,
                           std::vector<double>& yw) {
        y.assign(static_cast<std::size_t>(mu) * un, 0.0);
        yw.assign(un, 0.0);
        for (int l = 0; l < mu; ++l) {
            const auto src = static_cast<std::size_t>(ranking[static_cast<std::size_t>(l)]) * un;
            const std::span<double> yl(y.data() + static_cast<std::size_t>(l) * un, un);
            for (std::size_t k = 0; k < un; ++k) {
                yl[k] = (pts[src + k] - mean_[k]) / sigma_;
            }
            const double norm = mahalanobis_norm(yl);
            if (norm > params_.clip_norm) {
                const double f = params_.clip_norm / norm;
                for (double& v : yl) {
                    v *= f;
                }
            }
        }
        for (int l = 0; l < mu; ++l) {
            const double w = params_.weights[static_cast<std::size_t>(l)];
            for (std::size_t k = 0; k < un; ++k) {
                yw[k] += w * y[static_cast<std::size_t>(l) * un + k];
            }
        }
    };
    std::vector<double> y;
    std::vector<double> yw;
    steps(points, y, yw);
    // Schrittweitenpfad aus eigenen Punkten (z. B. den gezogenen Starts), sonst aus `points`
    std::vector<double> ys;
    std::vector<double> yw_sigma;
    if (!sigma_points.empty()) {
        steps(sigma_points, ys, yw_sigma);
    } else {
        yw_sigma = yw;
    }
    // C^{−1/2} y_w = B D⁻¹ Bᵀ y_w (mit B, D vor der Fortschreibung)
    std::vector<double> u(un);
    for (int k = 0; k < n; ++k) {
        double s = 0.0;
        for (int r = 0; r < n; ++r) {
            s += b_[idx(r, k, n)] * yw_sigma[static_cast<std::size_t>(r)];
        }
        u[static_cast<std::size_t>(k)] = s / d_[static_cast<std::size_t>(k)];
    }
    for (std::size_t k = 0; k < un; ++k) {
        mean_[k] += sigma_ * yw[k];
    }
    const double cs = params_.c_sigma;
    const double ps_factor = std::sqrt(cs * (2.0 - cs) * params_.mu_eff);
    double ps_norm2 = 0.0;
    for (int r = 0; r < n; ++r) {
        double s = 0.0;
        for (int k = 0; k < n; ++k) {
            s += b_[idx(r, k, n)] * u[static_cast<std::size_t>(k)];
        }
        double& ps = p_sigma_[static_cast<std::size_t>(r)];
        ps = (1.0 - cs) * ps + ps_factor * s;
        ps_norm2 += ps * ps;
    }
    const double ps_norm = std::sqrt(ps_norm2);
    sigma_decay_ *= (1.0 - cs) * (1.0 - cs);
    const bool h_sigma = ps_norm / std::sqrt(1.0 - sigma_decay_) <
                         (1.4 + 2.0 / (static_cast<double>(n) + 1.0)) * params_.chi_n;
    const double cc = params_.c_c;
    const double pc_factor = h_sigma ? std::sqrt(cc * (2.0 - cc) * params_.mu_eff) : 0.0;
    for (std::size_t k = 0; k < un; ++k) {
        p_c_[k] = (1.0 - cc) * p_c_[k] + pc_factor * yw[k];
    }
    const double delta = h_sigma ? 0.0 : cc * (2.0 - cc);
    const double c1 = params_.c_1;
    const double cmu = params_.c_mu;
    const double keep = 1.0 + c1 * delta - c1 - cmu;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double rank_mu = 0.0;
            for (int l = 0; l < mu; ++l) {
                const std::size_t o = static_cast<std::size_t>(l) * un;
                rank_mu += params_.weights[static_cast<std::size_t>(l)] *
                           y[o + static_cast<std::size_t>(i)] * y[o + static_cast<std::size_t>(j)];
            }
            double& cij = c_[idx(i, j, n)];
            cij = keep * cij +
                  c1 * p_c_[static_cast<std::size_t>(i)] * p_c_[static_cast<std::size_t>(j)] +
                  cmu * rank_mu;
        }
    }
    const double log_change =
        std::min(1.0, (cs / params_.d_sigma) * (ps_norm / params_.chi_n - 1.0));
    sigma_ *= portable_exp(log_change);
    ++generation_;
    decompose();
}

void CmaEs::decompose() {
    const int n = params_.dimension;
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i < n; ++i) {
            const double s = 0.5 * (c_[idx(i, j, n)] + c_[idx(j, i, n)]);
            c_[idx(i, j, n)] = s;
            c_[idx(j, i, n)] = s;
        }
    }
    std::vector<double> a = c_;
    std::vector<double> values(static_cast<std::size_t>(n));
    symmetric_eigen(n, a, values, b_);
    double vmax = 0.0;
    bool finite = true;
    for (const double v : values) {
        finite = finite && std::isfinite(v);
        vmax = std::max(vmax, v);
    }
    if (!finite || !(vmax > 0.0)) {
        // Entartete Kovarianz (sollte nicht vorkommen): zurück auf C = I
        std::fill(c_.begin(), c_.end(), 0.0);
        std::fill(b_.begin(), b_.end(), 0.0);
        for (int i = 0; i < n; ++i) {
            c_[idx(i, i, n)] = 1.0;
            b_[idx(i, i, n)] = 1.0;
        }
        std::fill(d_.begin(), d_.end(), 1.0);
        return;
    }
    // Konditionszahl von C höchstens 10¹⁴
    const double floor_value = vmax * 1e-14;
    for (int k = 0; k < n; ++k) {
        d_[static_cast<std::size_t>(k)] =
            std::sqrt(std::max(values[static_cast<std::size_t>(k)], floor_value));
    }
}

} // namespace mko
