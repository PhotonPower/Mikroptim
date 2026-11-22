#include "mikroptim/glass/glass_preselection.hpp"

#include "mikroptim/glass/dispersion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace mko {

GlassDispersion glass_dispersion(std::string name, const BuchdahlFit& fit) {
    if (!(fit.n0 > 1.0) || !std::isfinite(fit.n0)) {
        throw std::invalid_argument("Glasvorauswahl: n0 <= 1 bei " + name);
    }
    GlassDispersion d;
    d.name = std::move(name);
    d.n0 = fit.n0;
    for (std::size_t j = 0; j < 3; ++j) {
        d.eta[j] = fit.nu[j] / (fit.n0 - 1.0);
    }
    return d;
}

std::vector<GlassDispersion> catalog_dispersions(const GlassCatalog& catalog,
                                                 const BuchdahlFitOptions& options,
                                                 std::vector<std::string>* skipped) {
    std::vector<GlassDispersion> out;
    for (const Glass& g : catalog.glasses()) {
        try {
            out.push_back(glass_dispersion(g.name, fit_buchdahl(g, 3, options)));
        } catch (const std::exception&) {
            if (skipped != nullptr) {
                skipped->push_back(g.name);
            }
        }
    }
    return out;
}

namespace {

// Löst das (k+1)×(k+1)-System Σ w_i φ_i = φ, Σ w_i φ_i η_ij = 0 (j = 1 … k) per Gauß mit
// Pivotsuche; false bei (relativ) singulärer Matrix.
bool solve_powers(std::span<const GlassDispersion* const> g, std::span<const double> w, int k,
                  double phi, double tol, std::vector<double>& out) {
    const int m = k + 1;
    std::array<std::array<double, 4>, 3> a{}; // std::array: GCC 13 mit -march=x86-64-v3 meldete
                                              // beim C-Feld fälschlich maybe-uninitialized
    for (int i = 0; i < m; ++i) {
        a[0][i] = w[static_cast<std::size_t>(i)];
        for (int j = 1; j <= k; ++j) {
            a[j][i] = w[static_cast<std::size_t>(i)] *
                      g[static_cast<std::size_t>(i)]->eta[static_cast<std::size_t>(j - 1)];
        }
    }
    a[0][m] = phi;
    // Zeilennormen für das relative Kriterium
    double scale = 1.0;
    for (int r = 0; r < m; ++r) {
        double s = 0.0;
        for (int c = 0; c < m; ++c) {
            s += a[r][c] * a[r][c];
        }
        scale *= std::sqrt(s);
    }
    if (!(scale > 0.0)) {
        return false;
    }
    double det = 1.0;
    for (int c = 0; c < m; ++c) {
        int piv = c;
        for (int r = c + 1; r < m; ++r) {
            if (std::abs(a[r][c]) > std::abs(a[piv][c])) {
                piv = r;
            }
        }
        if (a[piv][c] == 0.0) {
            return false;
        }
        if (piv != c) {
            for (int q = 0; q <= m; ++q) {
                std::swap(a[c][q], a[piv][q]);
            }
            det = -det;
        }
        det *= a[c][c];
        for (int r = c + 1; r < m; ++r) {
            const double f = a[r][c] / a[c][c];
            for (int q = c; q <= m; ++q) {
                a[r][q] -= f * a[c][q];
            }
        }
    }
    if (std::abs(det) <= tol * scale) {
        return false;
    }
    out.assign(static_cast<std::size_t>(m), 0.0);
    for (int r = m - 1; r >= 0; --r) {
        double s = a[r][m];
        for (int q = r + 1; q < m; ++q) {
            s -= a[r][q] * out[static_cast<std::size_t>(q)];
        }
        out[static_cast<std::size_t>(r)] = s / a[r][r];
    }
    return std::all_of(out.begin(), out.end(), [](double x) { return std::isfinite(x); });
}

void check(const GlassPreselectionOptions& o, std::size_t n) {
    if (o.order != 1 && o.order != 2) {
        throw std::invalid_argument("Glasvorauswahl: order muss 1 oder 2 sein");
    }
    const auto m = static_cast<std::size_t>(o.order + 1);
    if (!o.weights.empty() && o.weights.size() != m) {
        throw std::invalid_argument("Glasvorauswahl: weights braucht order + 1 Einträge");
    }
    for (double w : o.weights) {
        if (!(w > 0.0) || !std::isfinite(w)) {
            throw std::invalid_argument("Glasvorauswahl: Gewichte müssen endlich und > 0 sein");
        }
    }
    if (!(o.power != 0.0) || !std::isfinite(o.power) || !(o.max_power_sum > 0.0) ||
        o.max_results < 1 || !(o.singular_tolerance >= 0.0)) {
        throw std::invalid_argument("Glasvorauswahl: ungültige Optionen");
    }
    if (o.fixed.size() > m) {
        throw std::invalid_argument("Glasvorauswahl: zu viele feste Gläser");
    }
    if (o.wavelengths_nm.size() == 1) {
        throw std::invalid_argument(
            "Glasvorauswahl: Spektrum braucht mindestens zwei Wellenlängen");
    }
    for (double l : o.wavelengths_nm) {
        if (!std::isfinite(l) || !(l > 1000.0 * (kBuchdahlLambda0Um - 1.0 / kBuchdahlAlpha))) {
            throw std::invalid_argument("Glasvorauswahl: Wellenlänge außerhalb der "
                                        "Buchdahl-Koordinate");
        }
    }
    for (std::size_t i = 0; i < o.fixed.size(); ++i) {
        if (o.fixed[i] < 0 || static_cast<std::size_t>(o.fixed[i]) >= n) {
            throw std::invalid_argument("Glasvorauswahl: festes Glas außerhalb der Liste");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (o.fixed[i] == o.fixed[j]) {
                throw std::invalid_argument("Glasvorauswahl: festes Glas doppelt");
            }
        }
    }
}

} // namespace

std::vector<PreselectedGlasses> preselect_glasses(std::span<const GlassDispersion> glasses,
                                                  const GlassPreselectionOptions& o) {
    check(o, glasses.size());
    const int k = o.order;
    const auto m = static_cast<std::size_t>(k + 1);
    const std::vector<double> w = o.weights.empty() ? std::vector<double>(m, 1.0) : o.weights;
    const auto n = static_cast<int>(glasses.size());
    std::vector<PreselectedGlasses> all;

    // Kombination: feste Gläser zuerst, freie aufsteigend ohne Wiederholung
    std::vector<int> free;
    for (int i = 0; i < n; ++i) {
        if (std::find(o.fixed.begin(), o.fixed.end(), i) == o.fixed.end()) {
            free.push_back(i);
        }
    }
    const std::size_t nfree = m - o.fixed.size();
    std::vector<std::size_t> pick(nfree);
    for (std::size_t i = 0; i < nfree; ++i) {
        pick[i] = i;
    }
    std::vector<int> idx(m);
    std::vector<const GlassDispersion*> g(m);
    std::vector<double> phi;
    // Buchdahl-Koordinaten des Spektrums für den spektralen Rest
    std::vector<double> omega;
    for (double l : o.wavelengths_nm) {
        omega.push_back(buchdahl_omega(l * 1e-3));
    }
    const bool spectral = !omega.empty();
    std::vector<double> rho(omega.size());
    const auto evaluate = [&]() {
        for (std::size_t i = 0; i < o.fixed.size(); ++i) {
            idx[i] = o.fixed[i];
        }
        for (std::size_t i = 0; i < nfree; ++i) {
            idx[o.fixed.size() + i] = free[pick[i]];
        }
        for (std::size_t i = 0; i < m; ++i) {
            g[i] = &glasses[static_cast<std::size_t>(idx[i])];
        }
        if (!solve_powers(g, w, k, o.power, o.singular_tolerance, phi)) {
            return;
        }
        double sum = 0.0;
        double res = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            sum += std::abs(w[i] * phi[i]);
            res += w[i] * phi[i] * g[i]->eta[static_cast<std::size_t>(k)];
        }
        sum /= std::abs(o.power);
        res = std::abs(res) / std::abs(o.power);
        if (!(sum <= o.max_power_sum)) {
            return;
        }
        double spread = std::numeric_limits<double>::quiet_NaN();
        if (spectral) {
            // ρ(ω) = Σ w_i φ_i Σ_{j>k} η_ij ω^j / φ; Streuung um den Mittelwert
            double mean = 0.0;
            for (std::size_t l = 0; l < omega.size(); ++l) {
                double r = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    double p = 0.0;
                    for (int j = 3; j > k; --j) {
                        p = (p + g[i]->eta[static_cast<std::size_t>(j - 1)]) * omega[l];
                    }
                    for (int j = k; j > 0; --j) {
                        p *= omega[l];
                    }
                    r += w[i] * phi[i] * p;
                }
                rho[l] = r / o.power;
                mean += rho[l];
            }
            mean /= static_cast<double>(omega.size());
            double var = 0.0;
            for (double r : rho) {
                var += (r - mean) * (r - mean);
            }
            spread = std::sqrt(var / static_cast<double>(omega.size()));
        }
        all.push_back(PreselectedGlasses{idx, phi, sum, res, spread});
    };
    if (nfree == 0) {
        evaluate();
    } else if (free.size() >= nfree) {
        for (;;) {
            evaluate();
            // nächste Kombination (lexikografisch)
            std::size_t i = nfree;
            while (i > 0 && pick[i - 1] == free.size() - nfree + (i - 1)) {
                --i;
            }
            if (i == 0) {
                break;
            }
            ++pick[i - 1];
            for (std::size_t j = i; j < nfree; ++j) {
                pick[j] = pick[j - 1] + 1;
            }
        }
    }
    const auto better = [spectral](const PreselectedGlasses& a, const PreselectedGlasses& b) {
        if (spectral && a.spectral_residual != b.spectral_residual) {
            return a.spectral_residual < b.spectral_residual;
        }
        if (a.residual != b.residual) {
            return a.residual < b.residual;
        }
        if (a.power_sum != b.power_sum) {
            return a.power_sum < b.power_sum;
        }
        return a.glasses < b.glasses;
    };
    const auto keep = std::min(all.size(), static_cast<std::size_t>(o.max_results));
    std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(keep), all.end(),
                      better);
    all.resize(keep);
    return all;
}

std::vector<GlassDispersion> filter_dispersions(std::span<const GlassDispersion> glasses,
                                                const GlassCatalog& catalog,
                                                const GlassCandidateFilter& filter,
                                                std::span<const std::string> keep) {
    const bool ranges = filter.properties.n_d || filter.properties.v_d || filter.properties.p_gF ||
                        filter.properties.dp_gF;
    const auto contains = [](const auto& list, const std::string& name) {
        return std::find(list.begin(), list.end(), name) != list.end();
    };
    std::vector<GlassDispersion> out;
    for (const GlassDispersion& d : glasses) {
        if (contains(keep, d.name)) {
            out.push_back(d);
            continue;
        }
        if (contains(filter.exclude, d.name) ||
            (!filter.only.empty() && !contains(filter.only, d.name))) {
            continue;
        }
        if (ranges) {
            const Glass& g = catalog.at(d.name);
            if (!g.in_range(kLineG) || !g.in_range(kLineC) ||
                !filter.properties.matches(glass_properties(g))) {
                continue;
            }
        }
        out.push_back(d);
    }
    return out;
}

namespace {

void check_member(std::span<const double> n, std::span<const double> x, std::size_t extra,
                  const char* what) {
    if (n.empty() || x.size() != n.size() + extra) {
        throw std::invalid_argument(std::string("Startradien: ") + what +
                                    " passt nicht zur Zahl der Brechzahlen");
    }
    for (double v : n) {
        if (!(v > 1.0) || !std::isfinite(v)) {
            throw std::invalid_argument("Startradien: Brechzahl muss endlich und > 1 sein");
        }
    }
    for (double v : x) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument(std::string("Startradien: ") + what + " nicht endlich");
        }
    }
}

// Beitrag eines dünnen Elements (y = 1, H = 1) mit Brechzahl n, Brechkraft K, Produkt
// BK = (n − 1)(c_vorn + c_hinten) und CK = u + u′ (Welford, Kap. 7).
std::array<double, 2> element_seidel(double n, double k, double bk, double ck) {
    const double e0 = (n / (n - 1.0)) * (n / (n - 1.0)) * k * k;
    const double e1 = (n + 2.0) / (n * (n - 1.0) * (n - 1.0));
    const double e2 = 4.0 * (n + 1.0) / (n * (n - 1.0));
    const double e3 = (3.0 * n + 2.0) / n;
    const double si = 0.25 * k * (e0 + e1 * bk * bk + e2 * bk * ck + e3 * ck * ck);
    const double sii = 0.5 * k * ((n + 1.0) / (n * (n - 1.0)) * bk + (2.0 * n + 1.0) / n * ck);
    return {si, sii};
}

} // namespace

std::array<double, 2> thin_member_seidel(std::span<const double> indices,
                                         std::span<const double> curvatures,
                                         double incoming_slope) {
    check_member(indices, curvatures, 1, "Krümmungsliste");
    std::array<double, 2> s{0.0, 0.0};
    double u = incoming_slope; // y = 1; 0 = kollimiert
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const double n = indices[i];
        const double k = (n - 1.0) * (curvatures[i] - curvatures[i + 1]);
        const double bk = (n - 1.0) * (curvatures[i] + curvatures[i + 1]);
        const std::array<double, 2> e = element_seidel(n, k, bk, 2.0 * u - k);
        s[0] += e[0];
        s[1] += e[1];
        u -= k;
    }
    return s;
}

ThinMemberShape thin_member_shape(std::span<const double> indices, std::span<const double> powers,
                                  MemberBending bending, double incoming_slope) {
    check_member(indices, powers, 0, "Brechkraftliste");
    // c_i = c₁ − D_i, D_i = Σ_{j<i} K_j/(n_j − 1); BK_i = (n_i − 1)(c_i + c_{i+1}) = α_i c₁ + β_i
    // mit α_i = 2(n_i − 1), β_i = −2(n_i − 1) D_i − K_i; CK_i = 2u_i − K_i hängt nicht von c₁ ab.
    ThinMemberShape r;
    auto& [a, b, d] = r.spherical_coefficients;
    auto& [s, t] = r.coma_coefficients;
    double offset = 0.0;
    double u = incoming_slope;
    std::vector<double> step(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const double n = indices[i];
        const double k = powers[i];
        step[i] = k / (n - 1.0);
        const double alpha = 2.0 * (n - 1.0);
        const double beta = -2.0 * (n - 1.0) * offset - k;
        const double ck = 2.0 * u - k;
        const double e1 = (n + 2.0) / (n * (n - 1.0) * (n - 1.0));
        const double e2 = 4.0 * (n + 1.0) / (n * (n - 1.0));
        // S_I,i = (K/4)[e₀ + e₁ BK² + e₂ BK CK + e₃ CK²] = a_i c₁² + b_i c₁ + d_i
        a += 0.25 * k * e1 * alpha * alpha;
        b += 0.25 * k * (2.0 * e1 * alpha * beta + e2 * alpha * ck);
        d += element_seidel(n, k, beta, ck)[0];
        // S_II,i = (K/2)[(n + 1)/(n(n − 1)) BK + (2n + 1)/n CK] = s_i c₁ + t_i
        s += 0.5 * k * (n + 1.0) / (n * (n - 1.0)) * alpha;
        t += element_seidel(n, k, beta, ck)[1];
        offset += step[i];
        u -= k;
    }
    // Σ c_i² der Flächen bei Biegung c₁ (flachere Form)
    const auto steepness = [&](double c) {
        double sum = c * c;
        for (double st : step) {
            c -= st;
            sum += c * c;
        }
        return sum;
    };
    double c1 = 0.0;
    if (bending == MemberBending::ZeroComa) {
        if (s == 0.0) {
            throw std::invalid_argument("Startradien: Koma hängt nicht von der Biegung ab");
        }
        c1 = -t / s;
    } else if (a != 0.0) {
        const double disc = b * b - 4.0 * a * d;
        if (disc >= 0.0) {
            // stabile Form der Nullstellen
            const double q = -0.5 * (b + std::copysign(std::sqrt(disc), b));
            const double r1 = q / a;
            const double r2 = q != 0.0 ? d / q : r1;
            c1 = steepness(r1) <= steepness(r2) ? r1 : r2;
            r.spherical_zero = true;
        } else {
            c1 = -b / (2.0 * a);
        }
    } else if (b != 0.0) {
        c1 = -d / b;
        r.spherical_zero = true;
    }
    r.curvatures.resize(indices.size() + 1);
    r.curvatures[0] = c1;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        r.curvatures[i + 1] = r.curvatures[i] - step[i];
    }
    const std::array<double, 2> sum = thin_member_seidel(indices, r.curvatures, incoming_slope);
    r.spherical = sum[0];
    r.coma = sum[1];
    return r;
}

} // namespace mko
