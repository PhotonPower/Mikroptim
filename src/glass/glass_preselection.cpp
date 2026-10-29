#include "mikroptim/glass/glass_preselection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

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
        all.push_back(PreselectedGlasses{idx, phi, sum, res});
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
    const auto better = [](const PreselectedGlasses& a, const PreselectedGlasses& b) {
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

} // namespace mko
