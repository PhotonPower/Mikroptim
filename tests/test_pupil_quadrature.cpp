// Pupillenquadratur (M1-T4): analytische Orakel.
//
// Exaktheitsbereich (docs/numerik.md, Abschnitt 8). Die Quadratur ist ein Tensorprodukt:
// radial Gauß-Legendre in u = ρ², exakt für Polynome in u bis Grad 2N−1; azimutal die
// Mittelpunktsregel, exakt für cos(mφ) mit m ≤ 2M−1 (halbe Pupille) bzw. cos(mφ), sin(mφ)
// mit m ≤ M−1 (volle Pupille). Für die Zernike-Polynome folgt:
//
//  - Mittelwert von R_n^0 (n gerade): R_n^0 ist ein Polynom vom Grad n/2 in u, also exakt
//    für n ≤ 4N−2. Erwartet: 1 für n = 0, sonst 0.
//  - Orthogonalität von Z_a = R_n^m cos(mφ) und Z_b = R_n'^m' cos(m'φ):
//    Z_a Z_b = ½ R_n^m R_n'^m' [cos((m+m')φ) + cos((m−m')φ)].
//    Azimutal exakt, wenn m + m' ≤ 2M−1 (halbe Pupille).
//    Radial: Ist m + m' gerade, so ist R_n^m R_n'^m' = u^((m+m')/2)·Polynom in u vom
//    Gesamtgrad (n+n')/2, also exakt für n + n' ≤ 4N−2. Ist m + m' ungerade, so ist der
//    radiale Faktor kein Polynom in u, aber beide Kosinusterme haben ungerade Ordnung
//    und werden azimutal exakt zu 0 integriert; das Ergebnis 0 = δ_ab ist dann ebenfalls
//    exakt.
//    Die Formulierung im Plan „bis n = 2N−1“ ist der Spezialfall n, n' ≤ 2N−1
//    (⇒ n + n' ≤ 4N−2); getestet wird der volle garantierte Bereich n + n' ≤ 4N−2.
//  - Normierung: ⟨(R_n^m cos mφ)²⟩ = (1 + δ_m0) / (2(n+1)), auf halber und voller Pupille.
//
// Die Zernike-Radialpolynome sind hier unabhängig vom Produktionscode implementiert:
// explizite Summenformel (nur für kleine n numerisch brauchbar, Auslöschung) und die
// stabile Rekursion nach Prata und Rusch, die gegen die Summenformel geprüft wird.

#include "mikroptim/pupil/quadrature.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

using mko::PupilMode;
using mko::PupilQuadrature;
using mko::PupilSample;
using mko::test::kQuadrature;
using mko::test::kQuadratureNode;

namespace {

constexpr PupilMode kModes[] = {PupilMode::Half, PupilMode::Full, PupilMode::Axis};

// ---------------------------------------------------------------------------
// Unabhängige Orakel
// ---------------------------------------------------------------------------

double factorial(int n) {
    double f = 1.0;
    for (int k = 2; k <= n; ++k) {
        f *= k;
    }
    return f;
}

// Explizite Summenformel
// R_n^m(ρ) = Σ_k (−1)^k (n−k)! / (k! ((n+m)/2−k)! ((n−m)/2−k)!) ρ^(n−2k).
// Wegen Auslöschung nur für kleine n genau.
double zernike_radial_explicit(int n, int m, double rho) {
    double sum = 0.0;
    for (int k = 0; k <= (n - m) / 2; ++k) {
        const double c = factorial(n - k) /
                         (factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k));
        sum += ((k % 2 == 0) ? c : -c) * std::pow(rho, n - 2 * k);
    }
    return sum;
}

// Alle R_n^m(ρ) für 0 ≤ m ≤ n ≤ nmax, n − m gerade, über die Rekursion nach Prata und Rusch
// (Appl. Opt. 28, 1989): R_n^n = ρ^n, R_n^m = ρ (R_{n−1}^{|m−1|} + R_{n−1}^{m+1}) − R_{n−2}^m.
// Ergebnis an Position n·(nmax+1) + m; Einträge mit ungeradem n − m bleiben 0.
class ZernikeRadialTable {
public:
    ZernikeRadialTable(int nmax, double rho) : nmax_(nmax), r_((nmax + 1) * (nmax + 1), 0.0) {
        for (int n = 0; n <= nmax; ++n) {
            at(n, n) = (n == 0) ? 1.0 : rho * at(n - 1, n - 1);
            for (int m = n - 2; m >= 0; m -= 2) {
                at(n, m) = rho * (at(n - 1, std::abs(m - 1)) + at(n - 1, m + 1)) - at(n - 2, m);
            }
        }
    }
    double operator()(int n, int m) const { return r_[n * (nmax_ + 1) + m]; }

private:
    double& at(int n, int m) { return r_[n * (nmax_ + 1) + m]; }
    int nmax_;
    std::vector<double> r_;
};

// Legendre-Polynom P_k(x) und Ableitung über die Dreitermrekursion.
std::pair<double, double> legendre(int k, double x) {
    double p0 = 1.0;
    double p1 = x;
    if (k == 0) {
        return {1.0, 0.0};
    }
    for (int j = 1; j < k; ++j) {
        const double p2 = ((2 * j + 1) * x * p1 - j * p0) / (j + 1);
        p0 = p1;
        p1 = p2;
    }
    return {p1, k * (x * p1 - p0) / (x * x - 1.0)};
}

// Doppelt-double-Arithmetik (Dekker, Knuth; ca. 106 Bit Mantisse) für eine unabhängige,
// genauere Referenz der Knoten und Gewichte. In FP64 allein wären die Gewichte an den
// Rändern nur auf ~1e-14 genau, weil der gerundete Knoten x in 1 + x eingeht.
struct DD {
    double hi, lo;
};

DD quick_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

DD operator+(DD a, DD b) {
    const double s = a.hi + b.hi;
    const double bb = s - a.hi;
    const double e = (a.hi - (s - bb)) + (b.hi - bb) + a.lo + b.lo;
    return quick_two_sum(s, e);
}

DD operator-(DD a) {
    return {-a.hi, -a.lo};
}
DD operator-(DD a, DD b) {
    return a + (-b);
}

DD operator*(DD a, DD b) {
    const double p = a.hi * b.hi;
    const double e = std::fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
    return quick_two_sum(p, e);
}

DD operator/(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    const DD r = a - b * DD{q1, 0.0};
    const double q2 = r.hi / b.hi;
    const DD r2 = r - b * DD{q2, 0.0};
    const double q3 = r2.hi / b.hi;
    return quick_two_sum(q1, q2) + DD{q3, 0.0};
}

DD dd(double x) {
    return {x, 0.0};
}

// P_n(x) und P_n'(x) in doppelt-double.
std::pair<DD, DD> legendre_dd(int n, DD x) {
    DD p0 = dd(1.0);
    DD p1 = x;
    for (int j = 1; j < n; ++j) {
        const DD p2 = (dd(2 * j + 1) * x * p1 - dd(j) * p0) / dd(j + 1);
        p0 = p1;
        p1 = p2;
    }
    return {p1, dd(n) * (x * p1 - p0) / (x * x - dd(1.0))};
}

// Gauß-Legendre auf [0, 1] per Newton-Iteration in doppelt-double, auf FP64 gerundet
// (unabhängig von der Tabelle und vom Generatorskript).
void gauss_legendre_reference(int n, std::vector<double>& u, std::vector<double>& w) {
    u.assign(n, 0.0);
    w.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        DD x = dd(std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5))); // absteigend
        for (int it = 0; it < 100; ++it) {
            const auto [p, dp] = legendre_dd(n, x);
            const DD dx = p / dp;
            x = x - dx;
            if (std::abs(dx.hi) < 1e-31) {
                break;
            }
        }
        const DD dp = legendre_dd(n, x).second;
        const DD one = dd(1.0);
        // hi eines normierten doppelt-double ist der auf FP64 gerundete Wert.
        u[n - 1 - i] = ((one + x) * dd(0.5)).hi;
        w[n - 1 - i] = (one / ((one - x) * (one + x) * dp * dp)).hi;
    }
}

// Mittelwert einer Funktion f(ρ, φ) über die Quadratur.
template <class F> double quad_mean(const PupilQuadrature<double>& q, F f) {
    double sum = 0.0;
    for (int k = 0; k < mko::sample_count(q); ++k) {
        const PupilSample<double> s = mko::sample(q, k);
        sum += s.weight * f(s.rho, s.phi);
    }
    return sum;
}

// Basisfunktion R_n^m(ρ)·cos(mφ) (sine = false) bzw. R_n^m(ρ)·sin(mφ) (sine = true).
struct ZernikeTerm {
    int n;
    int m;
    bool sine;
};

double expected_norm(const ZernikeTerm& t) {
    return (t.m == 0 ? 2.0 : 1.0) / (2.0 * (t.n + 1));
}

// Maximaler Fehler und Zahl der geprüften Paare.
struct OrthogonalityResult {
    double max_err = 0.0;
    long pairs = 0;
    void merge(const OrthogonalityResult& o) {
        max_err = std::max(max_err, o.max_err);
        pairs += o.pairs;
    }
};

// Prüft ⟨Z_a Z_b⟩ = δ_ab·Norm für alle Paare mit m + m' ≤ max_m_sum, für die der radiale
// Teil exakt ist (m + m' ungerade oder n + n' ≤ 4N−2).
OrthogonalityResult check_orthogonality(const PupilQuadrature<double>& q,
                                        const std::vector<ZernikeTerm>& terms, int max_m_sum) {
    const int nmax = 4 * q.rings - 2;
    const int count = mko::sample_count(q);
    const auto nt = terms.size();
    // Werte aller Basisfunktionen an allen Stützstellen.
    std::vector<double> z(nt * static_cast<std::size_t>(count));
    std::vector<double> weight(count);
    for (int k = 0; k < count; ++k) {
        const PupilSample<double> s = mko::sample(q, k);
        weight[k] = s.weight;
        const ZernikeRadialTable r(nmax, s.rho);
        for (std::size_t a = 0; a < nt; ++a) {
            const ZernikeTerm& t = terms[a];
            const double ang = terms[a].sine ? std::sin(t.m * s.phi) : std::cos(t.m * s.phi);
            z[a * count + k] = r(t.n, t.m) * ang;
        }
    }
    OrthogonalityResult res;
    for (std::size_t a = 0; a < nt; ++a) {
        for (std::size_t b = a; b < nt; ++b) {
            const ZernikeTerm& ta = terms[a];
            const ZernikeTerm& tb = terms[b];
            const int msum = ta.m + tb.m;
            if (msum > max_m_sum) {
                continue;
            }
            if (msum % 2 == 0 && ta.n + tb.n > nmax) {
                continue; // radialer Teil nicht garantiert exakt
            }
            double g = 0.0;
            for (int k = 0; k < count; ++k) {
                g += weight[k] * z[a * count + k] * z[b * count + k];
            }
            const double expected = (a == b) ? expected_norm(ta) : 0.0;
            const double err = std::abs(g - expected);
            res.max_err = std::max(res.max_err, err);
            ++res.pairs;
            EXPECT_LT(err, kQuadrature)
                << "N=" << q.rings << " M=" << q.spokes << " (n,m,sin)=(" << ta.n << "," << ta.m
                << "," << ta.sine << ") x (" << tb.n << "," << tb.m << "," << tb.sine << ")";
        }
    }
    EXPECT_GT(res.pairs, 0);
    return res;
}

} // namespace

// ---------------------------------------------------------------------------
// Orakel selbst
// ---------------------------------------------------------------------------

TEST(PupilQuadratureOracle, PrataRuschMatchesExplicitSum) {
    constexpr int kNmax = 12; // Summenformel darüber zu ungenau (Auslöschung)
    for (const double rho : {0.0, 0.1, 0.37, 0.5, 0.81, 0.99, 1.0}) {
        const ZernikeRadialTable r(kNmax, rho);
        for (int n = 0; n <= kNmax; ++n) {
            for (int m = n % 2; m <= n; m += 2) {
                EXPECT_NEAR(r(n, m), zernike_radial_explicit(n, m, rho), 1e-12)
                    << "n=" << n << " m=" << m << " rho=" << rho;
            }
        }
    }
    // Geschlossene Formen niedriger Ordnung.
    const double rho = 0.6;
    const ZernikeRadialTable r(4, rho);
    EXPECT_NEAR(r(2, 0), 2 * rho * rho - 1, 1e-15);
    EXPECT_NEAR(r(3, 1), 3 * std::pow(rho, 3) - 2 * rho, 1e-15);
    EXPECT_NEAR(r(4, 0), 6 * std::pow(rho, 4) - 6 * rho * rho + 1, 1e-15);
    EXPECT_NEAR(r(4, 2), 4 * std::pow(rho, 4) - 3 * rho * rho, 1e-15);
}

TEST(PupilQuadratureOracle, RotationalZernikeEqualsShiftedLegendre) {
    // R_2k^0(ρ) = P_k(2ρ² − 1), zweite unabhängige Darstellung. Das gerundete Argument
    // 2ρ² − 1 verschiebt P_k um bis zu |P_k'|·2ε; das geht in die Toleranz ein.
    constexpr int kNmax = 62;
    constexpr double kEps = std::numeric_limits<double>::epsilon();
    for (const double rho : {0.0, 0.05, 0.3, 0.5, 0.7071, 0.9, 0.999, 1.0}) {
        const ZernikeRadialTable r(kNmax, rho);
        for (int k = 0; 2 * k <= kNmax; ++k) {
            const double xi = 2 * rho * rho - 1;
            const auto [p, dp] = legendre(k, xi);
            // An ξ = ±1 liefert die Ableitungsformel 0/0; dort gilt |P_k'| = k(k+1)/2.
            const double dp_abs = std::isfinite(dp) ? std::abs(dp) : 0.5 * k * (k + 1);
            const double tol = 1e-14 + 2.0 * kEps * (dp_abs + 1.0);
            EXPECT_NEAR(r(2 * k, 0), p, tol) << "k=" << k << " rho=" << rho;
        }
    }
}

// ---------------------------------------------------------------------------
// Knoten und Gewichte
// ---------------------------------------------------------------------------

TEST(PupilQuadrature, NodesAscendingSymmetricPositive) {
    for (int n = 1; n <= mko::kMaxPupilRings; ++n) {
        const auto q = mko::make_pupil_quadrature(n, 1, PupilMode::Half);
        for (int i = 0; i < n; ++i) {
            EXPECT_GT(q.u[i], 0.0);
            EXPECT_LT(q.u[i], 1.0);
            EXPECT_GT(q.w[i], 0.0);
            if (i > 0) {
                EXPECT_LT(q.u[i - 1], q.u[i]) << "N=" << n << " i=" << i;
            }
            // Symmetrie u ↔ 1 − u: Gewichte bitgleich, Knoten bis auf die Rundung.
            EXPECT_EQ(q.w[i], q.w[n - 1 - i]) << "N=" << n << " i=" << i;
            EXPECT_NEAR(q.u[i] + q.u[n - 1 - i], 1.0, kQuadratureNode) << "N=" << n;
        }
    }
}

TEST(PupilQuadrature, NodesMatchClosedForms) {
    // Gauß-Legendre auf [−1, 1] in geschlossener Form, abgebildet auf u = (1+x)/2, W = w/2.
    struct Node {
        double x, w;
    };
    const double s30 = std::sqrt(30.0);
    const double s70 = std::sqrt(70.0);
    const std::vector<std::vector<Node>> rules = {
        {{0.0, 2.0}},
        {{-1.0 / std::sqrt(3.0), 1.0}, {1.0 / std::sqrt(3.0), 1.0}},
        {{-std::sqrt(0.6), 5.0 / 9.0}, {0.0, 8.0 / 9.0}, {std::sqrt(0.6), 5.0 / 9.0}},
        {{-std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(1.2)), (18.0 - s30) / 36.0},
         {-std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(1.2)), (18.0 + s30) / 36.0},
         {std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(1.2)), (18.0 + s30) / 36.0},
         {std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(1.2)), (18.0 - s30) / 36.0}},
        {{-std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0, (322.0 - 13.0 * s70) / 900.0},
         {-std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0, (322.0 + 13.0 * s70) / 900.0},
         {0.0, 128.0 / 225.0},
         {std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0, (322.0 + 13.0 * s70) / 900.0},
         {std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0, (322.0 - 13.0 * s70) / 900.0}},
    };
    for (std::size_t r = 0; r < rules.size(); ++r) {
        const int n = static_cast<int>(r) + 1;
        const auto q = mko::make_pupil_quadrature(n, 1, PupilMode::Axis);
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(q.u[i], 0.5 * (1.0 + rules[r][i].x), kQuadratureNode)
                << "N=" << n << " i=" << i;
            EXPECT_NEAR(q.w[i], 0.5 * rules[r][i].w, kQuadratureNode) << "N=" << n << " i=" << i;
        }
    }
    // Exakt darstellbare Werte.
    EXPECT_EQ(mko::make_pupil_quadrature(1, 1, PupilMode::Half).u[0], 0.5);
    EXPECT_EQ(mko::make_pupil_quadrature(1, 1, PupilMode::Half).w[0], 1.0);
    EXPECT_EQ(mko::make_pupil_quadrature(3, 1, PupilMode::Half).u[1], 0.5);
}

TEST(PupilQuadrature, NodesMatchIndependentNewton) {
    // Referenz in doppelt-double; die Tabelle soll korrekt gerundet sein (Abweichung 0,
    // höchstens 1 ulp bei Doppelrundung). Geprüft wird absolut (u) bzw. relativ (W).
    double max_du = 0.0;
    double max_dw = 0.0;
    int differing = 0;
    for (int n = 1; n <= mko::kMaxPupilRings; ++n) {
        std::vector<double> u, w;
        gauss_legendre_reference(n, u, w);
        const auto q = mko::make_pupil_quadrature(n, 1, PupilMode::Half);
        for (int i = 0; i < n; ++i) {
            const double du = std::abs(q.u[i] - u[i]);
            const double dw = std::abs(q.w[i] - w[i]) / q.w[i];
            max_du = std::max(max_du, du);
            max_dw = std::max(max_dw, dw);
            differing += (q.u[i] != u[i] ? 1 : 0) + (q.w[i] != w[i] ? 1 : 0);
            EXPECT_LT(du, kQuadratureNode) << "N=" << n << " i=" << i;
            EXPECT_LT(dw, kQuadratureNode) << "N=" << n << " i=" << i;
        }
    }
    std::cout << "[ info     ] max |du| = " << max_du << ", max |dW|/W = " << max_dw
              << ", nicht bitgleich: " << differing << " von 272\n";
    // Die Tabelle ist korrekt gerundet: bitgleich zur auf FP64 gerundeten Referenz.
    // (Bei Doppelrundung wäre 1 ulp zulässig; für die vorliegenden Werte tritt sie nicht auf.)
    EXPECT_EQ(differing, 0);
}

TEST(PupilQuadrature, WeightsSumToOne) {
    double max_err = 0.0;
    for (int n = 1; n <= mko::kMaxPupilRings; ++n) {
        for (int m = 1; m <= 16; ++m) {
            for (const PupilMode mode : kModes) {
                const auto q = mko::make_pupil_quadrature(n, m, mode);
                double sum = 0.0;
                double sum_sq_residual = 0.0;
                for (int k = 0; k < mko::sample_count(q); ++k) {
                    const auto s = mko::sample(q, k);
                    EXPECT_GT(s.weight, 0.0);
                    sum += s.weight;
                    const double rw = mko::residual_weight(s);
                    sum_sq_residual += rw * rw;
                }
                max_err = std::max(max_err, std::abs(sum - 1.0));
                EXPECT_NEAR(sum, 1.0, kQuadrature) << "N=" << n << " M=" << m;
                EXPECT_NEAR(sum_sq_residual, 1.0, kQuadrature) << "N=" << n << " M=" << m;
            }
        }
    }
    std::cout << "[ info     ] max |sum W - 1| = " << max_err << "\n";
}

TEST(PupilQuadrature, MonomialMomentsExactUpToDegree2NMinus1) {
    for (int n = 1; n <= mko::kMaxPupilRings; ++n) {
        const auto q = mko::make_pupil_quadrature(n, 1, PupilMode::Axis);
        for (int k = 0; k <= 2 * n - 1; ++k) {
            const double qm =
                quad_mean(q, [k](double rho, double) { return std::pow(rho, 2 * k); });
            EXPECT_NEAR(qm, 1.0 / (k + 1), kQuadrature) << "N=" << n << " k=" << k;
        }
    }
}

TEST(PupilQuadrature, DegreeTwoNIsNotExact) {
    // Schärfe des Bereichs: Für u^(2N) ist der Fehler der Gauß-Regel exakt
    // ∫ π_N(u)² du = (N!)⁴ / ((2N)!² (2N+1)) mit dem normierten verschobenen Legendre-Polynom.
    for (int n = 1; n <= 5; ++n) {
        const auto q = mko::make_pupil_quadrature(n, 1, PupilMode::Axis);
        const double qm = quad_mean(q, [n](double rho, double) { return std::pow(rho, 4 * n); });
        const double expected_err =
            std::pow(factorial(n), 4) / (factorial(2 * n) * factorial(2 * n) * (2 * n + 1));
        EXPECT_NEAR((1.0 / (2 * n + 1) - qm) / expected_err, 1.0, 1e-8) << "N=" << n;
    }
}

// ---------------------------------------------------------------------------
// Abnahme: Zernike-Mittelwerte und Orthogonalität
// ---------------------------------------------------------------------------

TEST(PupilQuadrature, RotationalZernikeMeansExact) {
    double max_err = 0.0;
    for (int n = 1; n <= mko::kMaxPupilRings; ++n) {
        const int nmax = 4 * n - 2; // R_nmax^0 hat Grad 2N−1 in u
        for (int m = 1; m <= 16; ++m) {
            for (const PupilMode mode : kModes) {
                const auto q = mko::make_pupil_quadrature(n, m, mode);
                std::vector<double> mean(nmax + 1, 0.0);
                for (int k = 0; k < mko::sample_count(q); ++k) {
                    const auto s = mko::sample(q, k);
                    const ZernikeRadialTable r(nmax, s.rho);
                    for (int zn = 0; zn <= nmax; zn += 2) {
                        mean[zn] += s.weight * r(zn, 0);
                    }
                }
                for (int zn = 0; zn <= nmax; zn += 2) {
                    const double err = std::abs(mean[zn] - (zn == 0 ? 1.0 : 0.0));
                    max_err = std::max(max_err, err);
                    EXPECT_LT(err, kQuadrature) << "N=" << n << " M=" << m << " n=" << zn;
                }
            }
        }
    }
    std::cout << "[ info     ] max |<R_n^0> - delta_n0| = " << max_err << "\n";
}

TEST(PupilQuadrature, ZernikeOrthogonalityHalfPupil) {
    std::vector<std::pair<int, int>> configs;
    for (int n = 1; n <= mko::kMaxPupilRings; ++n) {
        for (const int m : {1, 3, 8}) {
            configs.emplace_back(n, m);
        }
    }
    configs.emplace_back(16, 16);
    OrthogonalityResult total;
    for (const auto& [rings, spokes] : configs) {
        const auto q = mko::make_pupil_quadrature(rings, spokes, PupilMode::Half);
        const int nmax = 4 * rings - 2;
        const int mmax = 2 * spokes - 1;
        std::vector<ZernikeTerm> terms;
        for (int m = 0; m <= std::min(mmax, nmax); ++m) {
            for (int n = m; n <= nmax; n += 2) {
                terms.push_back({n, m, false});
            }
        }
        total.merge(check_orthogonality(q, terms, mmax));
    }
    std::cout << "[ info     ] halbe Pupille: " << total.pairs
              << " Paare, max Orthogonalitaetsfehler = " << total.max_err << "\n";
}

TEST(PupilQuadrature, ZernikeOrthogonalityFullPupil) {
    OrthogonalityResult total;
    for (const auto& [rings, spokes] : {std::pair{1, 1}, std::pair{3, 4}, std::pair{6, 7},
                                        std::pair{10, 16}, std::pair{16, 16}}) {
        const auto q = mko::make_pupil_quadrature(rings, spokes, PupilMode::Full);
        const int nmax = 4 * rings - 2;
        const int mmax = spokes - 1;
        std::vector<ZernikeTerm> terms;
        for (int m = 0; m <= std::min(mmax, nmax); ++m) {
            for (int n = m; n <= nmax; n += 2) {
                terms.push_back({n, m, false});
                if (m > 0) {
                    terms.push_back({n, m, true});
                }
            }
        }
        total.merge(check_orthogonality(q, terms, mmax));
    }
    std::cout << "[ info     ] volle Pupille: " << total.pairs
              << " Paare, max Orthogonalitaetsfehler = " << total.max_err << "\n";
}

TEST(PupilQuadrature, AxisCaseIsMeridionalFanAndExactForRotationalTerms) {
    OrthogonalityResult total;
    for (int n = 1; n <= mko::kMaxPupilRings; ++n) {
        const auto q = mko::make_pupil_quadrature(n, 8, PupilMode::Axis);
        EXPECT_EQ(q.spokes, 1);
        ASSERT_EQ(mko::sample_count(q), n);
        for (int k = 0; k < n; ++k) {
            const auto s = mko::sample(q, k);
            EXPECT_EQ(s.phi, 0.0); // φ = 0: p₀ = (0, aρ, 0) exakt meridional
            EXPECT_EQ(s.weight, q.w[k]);
            EXPECT_EQ(s.rho, std::sqrt(q.u[k]));
        }
        std::vector<ZernikeTerm> terms;
        for (int zn = 0; zn <= 4 * n - 2; zn += 2) {
            terms.push_back({zn, 0, false});
        }
        total.merge(check_orthogonality(q, terms, 0));
    }
    std::cout << "[ info     ] Achsfall: " << total.pairs
              << " Paare, max Orthogonalitaetsfehler = " << total.max_err << "\n";
}

// ---------------------------------------------------------------------------
// Index-Abbildung und Konfiguration
// ---------------------------------------------------------------------------

TEST(PupilQuadrature, IndexMappingRingMajor) {
    const int rings = 10;
    const int spokes = 8;
    const auto half = mko::make_pupil_quadrature(rings, spokes, PupilMode::Half);
    const auto full = mko::make_pupil_quadrature(rings, 2 * spokes, PupilMode::Full);
    ASSERT_EQ(mko::sample_count(half), rings * spokes);
    ASSERT_EQ(mko::sample_count(full), 2 * rings * spokes);
    const double pi = std::numbers::pi;
    for (int i = 0; i < rings; ++i) {
        for (int j = 0; j < spokes; ++j) {
            const auto s = mko::sample(half, i * spokes + j);
            EXPECT_EQ(s.rho, std::sqrt(half.u[i]));
            EXPECT_NEAR(s.phi, (j + 0.5) * pi / spokes, 1e-15);
            EXPECT_GT(s.phi, 0.0);
            EXPECT_LT(s.phi, pi);
            EXPECT_EQ(s.weight, half.w[i] / spokes);
        }
        for (int j = 0; j < 2 * spokes; ++j) {
            const auto s = mko::sample(full, i * 2 * spokes + j);
            EXPECT_NEAR(s.phi, (j + 0.5) * 2.0 * pi / (2 * spokes), 1e-15);
            if (j < spokes) {
                // Erste Hälfte der vollen Regel mit 2M Speichen = halbe Regel mit M Speichen.
                const auto h = mko::sample(half, i * spokes + j);
                EXPECT_EQ(s.phi, h.phi);
                EXPECT_EQ(s.rho, h.rho);
            } else {
                // Zweite Hälfte: Spiegelbild x → −x, d. h. φ → 2π − φ.
                const auto h = mko::sample(half, i * spokes + (2 * spokes - 1 - j));
                EXPECT_NEAR(s.phi, 2.0 * pi - h.phi, 1e-14);
            }
        }
    }
    // Halbe Pupille mit M = 1: Fächer bei φ = π/2 (entlang +x), im Unterschied zum Achsfall.
    const auto one = mko::make_pupil_quadrature(4, 1, PupilMode::Half);
    EXPECT_NEAR(mko::sample(one, 2).phi, 0.5 * pi, 1e-16);
}

TEST(PupilQuadrature, InvalidConfigurationThrows) {
    EXPECT_THROW(mko::make_pupil_quadrature(0, 8, PupilMode::Half), std::invalid_argument);
    EXPECT_THROW(mko::make_pupil_quadrature(17, 8, PupilMode::Half), std::invalid_argument);
    EXPECT_THROW(mko::make_pupil_quadrature(10, 0, PupilMode::Full), std::invalid_argument);
    EXPECT_THROW(mko::make_pupil_quadrature(10, 0, PupilMode::Axis), std::invalid_argument);
    EXPECT_THROW(mko::make_pupil_quadrature(10, 8, static_cast<PupilMode>(7)),
                 std::invalid_argument);
    EXPECT_NO_THROW(mko::make_pupil_quadrature(16, 1, PupilMode::Half));
}
