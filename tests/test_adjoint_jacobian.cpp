// Rückwärtsdurchlauf und adjungierte Jacobi-Matrix (M5-T2), CPU: lokale Ableitungen jeder
// Flächenoperation gegen Richardson-Extrapolation, Gradient der OPL einzelner Strahlen, Jacobi
// gegen Richardson-Referenz und FD-Jacobi für alle Variablentypen (Abnahme: relative Abweichung
// ≤ 1·10⁻⁶), Zusatzzeilen, mehrere Läufe, hartes Versagen, Fehlerfälle.

#include "adjoint_test_systems.hpp"
#include "fd_test_systems.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/merit/ray_jacobian.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/trace/adjoint_backward.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/trace.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using mko::AdjointJacobian;
using mko::AdjointJacobianOptions;
using mko::AdjointSeed;
using mko::RayAdjoint;
using mko::ResidualLayout;
using mko::SurfaceAdjoint;
using mko::SurfaceKind;
using mko::TraceStatus;
using mko::Vec3;

namespace {

// ---- Richardson-Gradient einer skalaren Funktion -------------------------------------------

// ∂F/∂x_i aus zentralen Differenzen mit h_i = factor·max(|x_i|, typ_i), h/2, h/4 und
// Richardson-Extrapolation R₂ = (64 D(h/4) − 20 D(h/2) + D(h))/45 (Fehler O(h⁶)), wie
// richardson_reference. Adaptiv über `levels` Stufen factor, factor/10, …: jeweils auch
// R₁ = (4 D(h/4) − D(h/2))/3 (O(h⁴)); pro Komponente gilt die Stufe mit dem kleinsten |R₁ − R₂|
// (Konvergenz der Extrapolation). Nahe einer Singularität (streifender Strahl, Profil weit
// außerhalb von r_g) ist der Abbruchfehler bei großem h groß, bei kleinem h die Rundung; bei
// fast linearen Abhängigkeiten mit kleinem x_typ (Geisterkoeffizienten, ν₂) ist großes h besser.
std::vector<double> richardson_gradient(const std::function<double(const std::vector<double>&)>& f,
                                        const std::vector<double>& x,
                                        const std::vector<double>& typical, double factor,
                                        int levels = 4) {
    std::vector<double> g(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        double best = std::numeric_limits<double>::infinity();
        double fac = factor;
        for (int level = 0; level < levels; ++level, fac /= 10.0) {
            const double h = fac * std::max(std::abs(x[i]), typical[i]);
            double d[3];
            for (int j = 0; j < 3; ++j) {
                const double hj = h / static_cast<double>(1 << j);
                std::vector<double> xp = x;
                std::vector<double> xm = x;
                xp[i] += hj;
                xm[i] -= hj;
                d[j] = (f(xp) - f(xm)) / (xp[i] - xm[i]);
            }
            const double r1 = (4.0 * d[2] - d[1]) / 3.0;
            const double r2 = (64.0 * d[2] - 20.0 * d[1] + d[0]) / 45.0;
            if (std::abs(r1 - r2) < best) {
                best = std::abs(r1 - r2);
                g[i] = r2;
            }
        }
    }
    return g;
}

// Relative Abweichung zweier Gradienten in der 2-Norm.
double relative_error(const std::vector<double>& a, const std::vector<double>& b) {
    double diff = 0.0;
    double ref = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff += (a[i] - b[i]) * (a[i] - b[i]);
        ref += b[i] * b[i];
    }
    return std::sqrt(diff / ref);
}

Vec3<double> unit(Vec3<double> v) {
    const double n = std::sqrt(mko::dot(v, v));
    return Vec3<double>{v.x / n, v.y / n, v.z / n};
}

// Zufällige Gewichte der Spur und Adjungierte des Ausgangszustands.
struct LocalSeeds {
    AdjointSeed<double> seed;
    RayAdjoint<double> out;
};

LocalSeeds random_seeds(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    LocalSeeds s;
    s.seed = AdjointSeed<double>{u(rng), u(rng)};
    s.out = RayAdjoint<double>{{u(rng), u(rng), u(rng)}, {u(rng), u(rng), u(rng)}};
    return s;
}

double phi(const LocalSeeds& s, const Vec3<double>& p, const Vec3<double>& d, double opl,
           double penalty) {
    return mko::dot(s.out.p, p) + mko::dot(s.out.d, d) + s.seed.opl * opl +
           s.seed.penalty * penalty;
}

} // namespace

// ---- Lokale Ableitungen ----------------------------------------------------------------------

// Schritt an Stop/Plane/Sphere (Transfer, Spencer-Murty-Schnitt, Brechung) in allen Zweigen
// (regulär, Fläche verfehlt, Totalreflexion, beides): Adjungierte von Eingangszustand, t, c, n_vor,
// n_nach gegen Richardson-Differenzen von Φ = p̄·p′ + d̄·d′ + ō·ΔOPL + π̄·Δπ über
// propagate_surface (mit Akkumulator des weichen Versagens).
TEST(AdjointLocal, RefractingStepMatchesRichardson) {
    std::mt19937_64 rng(20261019);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::array<int, 4> cases{};
    double worst[4] = {0.0, 0.0, 0.0, 0.0};
    int tries = 0;
    while ((cases[0] < 200 || cases[1] < 200 || cases[2] < 200 || cases[3] < 50) &&
           tries < 400000) {
        ++tries;
        // Eingang: x = (p_x, p_y, p_z, d_x, d_y, d_z, t, c, n_vor, n_nach)
        const double height = 8.0 * u(rng);
        const Vec3<double> p{0.5 * u(rng), height, 0.3 * u(rng)};
        const Vec3<double> d = unit({0.4 * u(rng), 0.6 * u(rng), 1.0});
        const double t = 3.0 + 2.0 * u(rng);
        const bool plane = tries % 7 == 0;
        const double c = plane ? 0.0 : 0.2 * u(rng);
        const double n_b = 1.0 + 0.8 * std::abs(u(rng));
        const double n_a = 1.0 + 0.8 * std::abs(u(rng));
        const SurfaceKind kind = plane ? SurfaceKind::Plane : SurfaceKind::Sphere;
        const std::vector<double> x0{p.x, p.y, p.z, d.x, d.y, d.z, t, c, n_b, n_a};
        // Zweig und Abstand zur Grenze bei x₀
        unsigned char branch = 0;
        mko::SoftFailure<double> soft{0.0, TraceStatus::Ok};
        mko::Ray<double> ray{p, d, 0.0};
        if (mko::propagate_surface(ray, t, kind, c, n_b, n_a, nullptr, &soft, &branch) !=
            TraceStatus::Ok) {
            continue;
        }
        // Grenzabstand (D bzw. k) groß genug, damit die Differenzen im Zweig bleiben
        const auto tl = [&](const std::vector<double>& x, unsigned char& b) {
            mko::SoftFailure<double> sf{0.0, TraceStatus::Ok};
            mko::Ray<double> r{{x[0], x[1], x[2]}, {x[3], x[4], x[5]}, 0.0};
            const TraceStatus st =
                mko::propagate_surface(r, x[6], kind, x[7], x[8], x[9], nullptr, &sf, &b);
            return std::make_pair(st, std::make_pair(r, sf.penalty));
        };
        const int bucket = branch;
        if (cases[static_cast<std::size_t>(bucket)] >= (bucket == 3 ? 50 : 200)) {
            continue;
        }
        bool stable = true;
        for (std::size_t i = 0; i < x0.size() && stable; ++i) {
            for (const double sgn : {-1.0, 1.0}) {
                std::vector<double> xs = x0;
                xs[i] += sgn * 1e-3 * std::max(std::abs(x0[i]), 1.0);
                unsigned char b = 0;
                const auto r = tl(xs, b);
                if (r.first != TraceStatus::Ok || b != branch || !(r.second.first.d.z > 1e-3)) {
                    stable = false;
                }
            }
        }
        if (!stable || !(ray.d.z > 1e-3)) {
            continue;
        }
        const LocalSeeds s = random_seeds(rng);
        const auto f = [&](const std::vector<double>& x) {
            unsigned char b = 0;
            const auto r = tl(x, b);
            return phi(s, r.second.first.p, r.second.first.d, r.second.first.opl, r.second.second);
        };
        RayAdjoint<double> adj = s.out;
        SurfaceAdjoint<double> par{0.0, 0.0, 0.0, 0.0};
        mko::refracting_step_backward(p, d, t, c, n_b, n_a, branch, s.seed, adj, par);
        if (plane) {
            par.curvature = 0.0; // Ebene: c wird ignoriert (backward_ray meldet sie nicht)
        }
        const std::vector<double> g{adj.p.x,      adj.p.y,    adj.p.z,       adj.d.x,
                                    adj.d.y,      adj.d.z,    par.thickness, par.curvature,
                                    par.n_before, par.n_after};
        const std::vector<double> typical{1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 0.01, 1.0, 1.0};
        const std::vector<double> ref = richardson_gradient(f, x0, typical, 1e-2);
        const double e = relative_error(g, ref);
        worst[bucket] = std::max(worst[bucket], e);
        ++cases[static_cast<std::size_t>(bucket)];
        EXPECT_LT(e, mko::test::kAdjointLocalRichardson)
            << "Zweig " << bucket << ", Fall " << tries;
    }
    std::printf("Brechender Schritt: regulär %d Fälle (max %.2e), verfehlt %d (%.2e), "
                "Totalreflexion %d (%.2e), beides %d (%.2e)\n",
                cases[0], worst[0], cases[1], worst[1], cases[2], worst[2], cases[3], worst[3]);
    EXPECT_GE(cases[0], 200);
    EXPECT_GE(cases[1], 200);
    EXPECT_GE(cases[2], 200);
    EXPECT_GE(cases[3], 50);
}

// Schritt an einer Geisterplatte (Transfer, Legendre-Profil, Phase, Ablenkung) im regulären Zweig
// und im Zweig Radikand < 0, K = 1 … 8, auch außerhalb des Normierungsradius und mit n ≠ n′
// (brechende Ebene mit Phase): Adjungierte von Eingangszustand, t, n_vor, n_nach und allen
// Koeffizienten gegen Richardson.
TEST(AdjointLocal, GhostStepMatchesRichardson) {
    std::mt19937_64 rng(46);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::array<int, 2> cases{};
    double worst[2] = {0.0, 0.0};
    int tries = 0;
    while ((cases[0] < 300 || cases[1] < 100) && tries < 200000) {
        ++tries;
        const int K = 1 + tries % 8;
        const double r_g = 2.0 + 3.0 * std::abs(u(rng));
        const Vec3<double> p{1.5 * r_g * u(rng), 1.5 * r_g * u(rng), 0.2 * u(rng)};
        const Vec3<double> d = unit({0.3 * u(rng), 0.5 * u(rng), 1.0});
        const double t = 2.0 + u(rng);
        const bool same = tries % 3 != 0;
        const double n_b = same ? 1.0 : 1.0 + 0.5 * std::abs(u(rng));
        const double n_a = same ? 1.0 : 1.0 + 0.5 * std::abs(u(rng));
        const double n_g = 1.4 + 0.4 * std::abs(u(rng));
        const double amp = (tries % 4 == 0) ? 0.5 : 0.02;
        std::vector<double> x0{p.x, p.y, p.z, d.x, d.y, d.z, t, n_b, n_a};
        for (int k = 0; k < K; ++k) {
            x0.push_back(amp * u(rng) / static_cast<double>(k + 1));
        }
        const auto run = [&](const std::vector<double>& x, unsigned char& b, mko::Ray<double>& r,
                             double& pen) {
            const mko::GhostPlateParams<double> g{x.data() + 9, K, r_g, n_g};
            mko::SoftFailure<double> sf{0.0, TraceStatus::Ok};
            r = mko::Ray<double>{{x[0], x[1], x[2]}, {x[3], x[4], x[5]}, 0.0};
            const TraceStatus st = mko::propagate_surface(r, x[6], SurfaceKind::GhostPlate, 0.0,
                                                          x[7], x[8], &g, &sf, &b);
            pen = sf.penalty;
            return st;
        };
        unsigned char branch = 0;
        mko::Ray<double> ray{};
        double pen = 0.0;
        if (run(x0, branch, ray, pen) != TraceStatus::Ok) {
            continue;
        }
        const int bucket = branch == mko::kBranchSoftTIR ? 1 : 0;
        if (cases[static_cast<std::size_t>(bucket)] >= (bucket == 0 ? 300 : 100)) {
            continue;
        }
        bool stable = !(bucket == 0 && !(ray.d.z > 1e-2));
        for (std::size_t i = 0; i < x0.size() && stable; ++i) {
            for (const double sgn : {-1.0, 1.0}) {
                std::vector<double> xs = x0;
                xs[i] += sgn * 1e-3 * std::max(std::abs(x0[i]), 1.0);
                unsigned char b = 0;
                mko::Ray<double> r{};
                double pp = 0.0;
                if (run(xs, b, r, pp) != TraceStatus::Ok || b != branch) {
                    stable = false;
                }
            }
        }
        if (!stable) {
            continue;
        }
        const LocalSeeds s = random_seeds(rng);
        const auto f = [&](const std::vector<double>& x) {
            unsigned char b = 0;
            mko::Ray<double> r{};
            double pp = 0.0;
            run(x, b, r, pp);
            return phi(s, r.p, r.d, r.opl, pp);
        };
        RayAdjoint<double> adj = s.out;
        SurfaceAdjoint<double> par{0.0, 0.0, 0.0, 0.0};
        std::vector<double> coeff_adj(static_cast<std::size_t>(K), 0.0);
        const mko::GhostPlateParams<double> plate{x0.data() + 9, K, r_g, n_g};
        mko::ghost_step_backward(
            p, d, t, plate, n_b, n_a, branch, s.seed, adj, par,
            [&](int k, double v) { coeff_adj[static_cast<std::size_t>(k)] = v; });
        std::vector<double> g{adj.p.x, adj.p.y,       adj.p.z,      adj.d.x,    adj.d.y,
                              adj.d.z, par.thickness, par.n_before, par.n_after};
        g.insert(g.end(), coeff_adj.begin(), coeff_adj.end());
        std::vector<double> typical{1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0};
        typical.resize(x0.size(), 1e-3);
        const std::vector<double> ref = richardson_gradient(f, x0, typical, 1e-2);
        const double e = relative_error(g, ref);
        worst[bucket] = std::max(worst[bucket], e);
        ++cases[static_cast<std::size_t>(bucket)];
        EXPECT_LT(e, mko::test::kAdjointLocalRichardson) << "Zweig " << bucket << ", K = " << K;
    }
    std::printf("Geisterplatte: regulär %d Fälle (max %.2e), Radikand < 0 %d (%.2e)\n", cases[0],
                worst[0], cases[1], worst[1]);
    EXPECT_GE(cases[0], 300);
    EXPECT_GE(cases[1], 100);
}

// Schritt auf die Referenzsphäre (σ = ±1: reeller und virtueller Bildpunkt; Zweig verfehlt mit
// Strafe −(b² − g)/R_ref²): Adjungierte von Eingangszustand, Mittelpunkt M und n_B gegen
// Richardson-Differenzen von ō·ΔOPL + π̄·Δπ über reference_sphere_step.
TEST(AdjointLocal, ReferenceStepMatchesRichardson) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::array<int, 3> cases{}; // reell, virtuell, verfehlt
    double worst[3] = {0.0, 0.0, 0.0};
    int tries = 0;
    while ((cases[0] < 200 || cases[1] < 200 || cases[2] < 200) && tries < 200000) {
        ++tries;
        const bool virt = tries % 2 == 0;
        const Vec3<double> center{0.0, 2.0 * u(rng),
                                  (virt ? -1.0 : 1.0) * (20.0 + 30.0 * std::abs(u(rng)))};
        const Vec3<double> p{3.0 * u(rng), 3.0 * u(rng), 0.5 * u(rng)};
        const double spread = tries % 5 == 0 ? 3.0 : 0.15;
        const Vec3<double> d = unit({spread * u(rng), spread * u(rng), 1.0});
        const double n_img = 1.0 + 0.5 * std::abs(u(rng));
        const std::vector<double> x0{p.x, p.y,      p.z,      d.x,      d.y,
                                     d.z, center.x, center.y, center.z, n_img};
        const auto run = [&](const std::vector<double>& x, TraceStatus& st, double& pen) {
            mko::Ray<double> r{{x[0], x[1], x[2]}, {x[3], x[4], x[5]}, 0.0};
            mko::SoftFailure<double> sf{0.0, TraceStatus::Ok};
            st = mko::reference_sphere_step(r, Vec3<double>{x[6], x[7], x[8]}, x[9], nullptr, &sf);
            pen = sf.penalty;
            return r.opl;
        };
        TraceStatus st0;
        double pen0 = 0.0;
        run(x0, st0, pen0);
        if (mko::is_hard_failure(st0)) {
            continue;
        }
        const int bucket = st0 == TraceStatus::SoftMissed ? 2 : (virt ? 1 : 0);
        if (cases[static_cast<std::size_t>(bucket)] >= 200) {
            continue;
        }
        bool stable = true;
        for (std::size_t i = 0; i < x0.size() && stable; ++i) {
            for (const double sgn : {-1.0, 1.0}) {
                std::vector<double> xs = x0;
                xs[i] += sgn * 1e-3 * std::max(std::abs(x0[i]), 1.0);
                TraceStatus st;
                double pp = 0.0;
                run(xs, st, pp);
                stable = stable && st == st0;
            }
        }
        if (!stable) {
            continue;
        }
        const LocalSeeds s = random_seeds(rng);
        const auto f = [&](const std::vector<double>& x) {
            TraceStatus st;
            double pp = 0.0;
            const double opl = run(x, st, pp);
            return s.seed.opl * opl + s.seed.penalty * pp;
        };
        RayAdjoint<double> adj{};
        Vec3<double> center_adj{0.0, 0.0, 0.0};
        double n_adj = 0.0;
        mko::reference_step_backward(p, d, center, n_img,
                                     st0 == TraceStatus::SoftMissed ? mko::kBranchSoftMissed
                                                                    : mko::kBranchRegular,
                                     s.seed, adj, center_adj, n_adj);
        const std::vector<double> g{adj.p.x, adj.p.y,      adj.p.z,      adj.d.x,      adj.d.y,
                                    adj.d.z, center_adj.x, center_adj.y, center_adj.z, n_adj};
        const std::vector<double> typical{1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0};
        const std::vector<double> ref = richardson_gradient(f, x0, typical, 1e-2);
        const double e = relative_error(g, ref);
        worst[bucket] = std::max(worst[bucket], e);
        ++cases[static_cast<std::size_t>(bucket)];
        EXPECT_LT(e, mko::test::kAdjointLocalRichardson) << "Fall " << bucket;
    }
    std::printf("Referenzsphäre: reell %d Fälle (max %.2e), virtuell %d (%.2e), verfehlt %d "
                "(%.2e)\n",
                cases[0], worst[0], cases[1], worst[1], cases[2], worst[2]);
    EXPECT_GE(cases[0], 200);
    EXPECT_GE(cases[1], 200);
    EXPECT_GE(cases[2], 200);
}

// Startstrahl der Punktquelle: ∂(d̄·d₀)/∂n₀ über sin_max = NA/n₀ gegen Richardson.
TEST(AdjointLocal, PointSourceStartMatchesRichardson) {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double worst = 0.0;
    for (int i = 0; i < 500; ++i) {
        const double na = 0.05 + 0.8 * std::abs(u(rng));
        const double n0 = na + 0.1 + std::abs(u(rng));
        const double rho = std::abs(u(rng));
        const double phi_s = 3.14159 * u(rng);
        const RayAdjoint<double> adj{{u(rng), u(rng), u(rng)}, {u(rng), u(rng), u(rng)}};
        const auto f = [&](const std::vector<double>& x) {
            const mko::Ray<double> r =
                mko::point_source_ray(Vec3<double>{0.0, 0.0, 0.0}, na / x[0], rho, phi_s);
            return mko::dot(adj.d, r.d);
        };
        const mko::Ray<double> r0 =
            mko::point_source_ray(Vec3<double>{0.0, 0.0, 0.0}, na / n0, rho, phi_s);
        const double g = mko::point_source_backward(r0.d, n0, adj);
        const std::vector<double> ref = richardson_gradient(f, {n0}, {1.0}, 1e-2);
        const double e = std::abs(g - ref[0]) / std::max(std::abs(ref[0]), 1e-12);
        worst = std::max(worst, e);
        EXPECT_LT(e, mko::test::kAdjointLocalRichardson);
    }
    std::printf("Startstrahl der Punktquelle: max %.2e\n", worst);
}

// ---- Gradient der OPL einzelner Strahlen --------------------------------------------------

namespace {

// OPL-Gradient eines Strahls (v = 0, Feld f, Wellenlänge l, Stützstelle k) über backward_ray
// nach allen Variablen (eine Zeile, rows = 1).
std::vector<double> ray_opl_gradient(const mko::test::FdProblem& p, const ResidualLayout& layout,
                                     std::span<const double> x, int f, int l, int k,
                                     TraceStatus* status) {
    mko::SystemBatchHost b = mko::make_adjoint_batch(p.set, p.base, x);
    const mko::AdjointParameters params(p.set, b);
    const auto view = b.view();
    const auto table = view.surface_table(0);
    const Vec3<double> center = mko::batch_image_point(view, table, 0, f);
    const auto rec = std::make_unique<mko::AdjointRecord<double>>();
    mko::SoftFailure<double> soft{0.0, TraceStatus::Ok};
    mko::NoSurfaceObserver* const none = nullptr;
    double opl = 0.0;
    const auto s = mko::sample(layout.quadrature(f), k);
    *status = mko::batch_ray_opl(view, table, l, 0, f, l, s, center, &opl, none, &soft, rec.get());
    std::vector<double> g(static_cast<std::size_t>(p.set.size()), 0.0);
    const mko::JacobianRowWriter<double> w{params.view(), l, g.data(), 1, 0, 1.0};
    mko::backward_ray(table, l, *rec, true, center, view.source == mko::SourceKind::Point,
                      AdjointSeed<double>{1.0, 0.0}, w);
    return g;
}

double ray_opl(const mko::test::FdProblem& p, const ResidualLayout& layout,
               std::span<const double> x, int f, int l, int k) {
    mko::SystemBatchHost b = mko::make_adjoint_batch(p.set, p.base, x);
    const auto view = b.view();
    const auto table = view.surface_table(0);
    const Vec3<double> center = mko::batch_image_point(view, table, 0, f);
    mko::SoftFailure<double> soft{0.0, TraceStatus::Ok};
    mko::NoSurfaceObserver* const none = nullptr;
    mko::NoTraceRecorder* const no_record = nullptr;
    double opl = 0.0;
    mko::batch_ray_opl(view, table, l, 0, f, l, mko::sample(layout.quadrature(f), k), center, &opl,
                       none, &soft, no_record);
    return opl;
}

// Typische Größen für die Schrittweite der Richardson-Referenz einzelner Strahlen: x_typ der
// Variablen, für Geisterkoeffizienten 10⁻² mm statt 10⁻⁴ mm (OPL und Strafe hängen fast linear
// von a ab; mit x_typ = 10⁻⁴ mm wäre die Differenz rundungsbegrenzt, ε·OPL/h ≈ 10⁻⁸ relativ).
std::vector<double> reference_typical(const mko::VariableSet& set) {
    std::vector<double> typical(set.typical().begin(), set.typical().end());
    for (int k = 0; k < set.size(); ++k) {
        if (set[k].type == mko::VariableType::GhostCoefficient) {
            typical[static_cast<std::size_t>(k)] = 1e-2;
        }
    }
    return typical;
}

} // namespace

// ∂OPL/∂x eines Einzelstrahls (ohne Kolbenabzug) gegen Richardson über die ganze Kette
// (Startstrahl, alle Flächen, Referenzsphäre) für alle Variablentypen: Randstrahl und Strahl in
// der Pupillenmitte jedes Feldes.
TEST(AdjointRay, OplGradientMatchesRichardson) {
    const auto cases = mko::test::adjoint_cases();
    double worst = 0.0;
    int rays = 0;
    for (const auto& c : cases) {
        const mko::test::FdProblem& p = c->p;
        const ResidualLayout layout(p.base.topology(), c->options.quadrature);
        const std::vector<double> typical = reference_typical(p.set);
        for (int f = 0; f < layout.fields(); ++f) {
            const int n = layout.samples(f);
            for (const int k : {0, n / 2, n - 1}) {
                const int l = (f + k) % layout.wavelengths();
                TraceStatus st;
                const std::vector<double> g = ray_opl_gradient(p, layout, p.x0, f, l, k, &st);
                if (st != TraceStatus::Ok) {
                    continue; // weich versagte Strahlen: eigener Test unten
                }
                const std::vector<double> ref = richardson_gradient(
                    [&](const std::vector<double>& x) { return ray_opl(p, layout, x, f, l, k); },
                    p.x0, typical, 1e-1, 5);
                const double e = relative_error(g, ref);
                worst = std::max(worst, e);
                ++rays;
                EXPECT_LT(e, mko::test::kAdjointRayRichardson)
                    << c->name << ", Feld " << f << ", Stützstelle " << k;
            }
        }
    }
    std::printf("OPL-Gradient einzelner Strahlen: %d Strahlen, max. relativ %.2e\n", rays, worst);
    EXPECT_GT(rays, 20);
}

// ---- Jacobi-Matrix -----------------------------------------------------------------------

// Abnahme M5-T2: adjungierte Jacobi gegen die Richardson-Referenz (adaptiv: H/x = 10⁻¹ … 10⁻⁵,
// je H, H/2, H/4, Stufe mit der besten Konvergenz pro Spalte, nur mit stabilem Strahlstatus)
// für alle Variablentypen auf allen Testsystemen. Geprüft werden die Zeilen des Traces (Strahl-
// und Strafresiduen, adjungiert): relative Abweichung pro Spalte und spaltenskaliert in der
// Frobeniusnorm ≤ 1·10⁻⁶. Die Zusatzzeilen sind finite Differenzen wie in der FD-Jacobi
// (hybrider Ansatz, Vergleich mit ihr in MatchesFdJacobian); ihre Abweichung von der Referenz wird
// nur ausgegeben.
TEST(AdjointJacobian, AcceptanceAgainstRichardson) {
    const auto cases = mko::test::adjoint_cases();
    double worst_type[5] = {-1.0, -1.0, -1.0, -1.0, -1.0};
    for (const auto& c : cases) {
        if (!c->richardson) {
            continue;
        }
        const mko::test::FdProblem& p = c->p;
        const int P = p.set.size();
        const ResidualLayout layout(p.base.topology(), c->options.quadrature);
        const std::int64_t rv = layout.trace_rows();
        const AdjointJacobian j = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0, c->options);
        const mko::test::AdaptiveReference ref =
            mko::test::adaptive_reference(p, layout, c->options.extra, rv);
        ASSERT_FALSE(ref.r2.empty()) << c->name << ": Status bei keiner Schrittweite stabil";
        EXPECT_LT(ref.max_convergence, mko::test::kFdRichardsonReference) << c->name;
        const std::span<const double> jr = mko::test::run_columns(j, 0);
        const auto d = mko::test::adjoint_deviation(mko::test::row_block(jr, P, 0, rv),
                                                    mko::test::row_block(ref.r2, P, 0, rv), p);
        mko::test::print_deviation("Richardson, Trace-Zeilen", c->name, p, d);
        const auto full = mko::test::adjoint_deviation(jr, ref.r2, p);
        std::printf("    Referenz |R1 - R2| %.3e (Stufen H/x %.0e … %.0e), hart versagte Strahlen "
                    "bei x0: %zu; mit %d Zusatzzeilen (FD): skaliert %.3e, größte Spalte %.3e\n",
                    ref.max_convergence, ref.min_factor, ref.max_factor, j.hard_failed_count(),
                    j.shape.extra_rows, full.scaled.frobenius, full.plain.max_column);
        EXPECT_LT(d.scaled.frobenius, mko::test::kAdjointJacobian) << c->name;
        EXPECT_LT(d.plain.frobenius, mko::test::kAdjointJacobian) << c->name;
        for (int k = 0; k < P; ++k) {
            EXPECT_LT(d.plain.column[static_cast<std::size_t>(k)], mko::test::kAdjointJacobian)
                << c->name << ", " << p.set[k].name;
        }
        for (const auto& t : mko::test::deviation_by_type(p, d.plain)) {
            auto& w = worst_type[static_cast<int>(t.type)];
            w = std::max(w, t.max_column);
        }
    }
    std::printf("Größte Spaltenabweichung pro Variablentyp:");
    for (int t = 0; t < 5; ++t) {
        std::printf(" %s %.2e;", mko::test::variable_type_name(static_cast<mko::VariableType>(t)),
                    worst_type[t]);
        EXPECT_GE(worst_type[t], 0.0) << "Variablentyp ohne Test";
    }
    std::printf("\n");
}

namespace {

template <class T>
bool same_bytes(std::span<const T> a, std::span<const T> b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

} // namespace

// Gegen die FD-Jacobi (M3-T2, gleiche Ablage): r₀ bitgleich, Zusatzzeilen (hybrider Ansatz,
// dieselben finiten Differenzen) bitgleich, die ganze Matrix spaltenskaliert in der
// Frobeniusnorm ≤ 1·10⁻⁶ (die FD-Jacobi selbst hat Fehler ~10⁻⁸, einzelne rundungsbegrenzte
// Spalten bis einige 10⁻⁶, docs/optimierung.md, Abschnitt „Finite-Differenzen-Jacobi“). Auch der
// Fall mit trace-abhängigen Zusatzresiduen (Mechanik).
TEST(AdjointJacobian, MatchesFdJacobian) {
    const auto cases = mko::test::adjoint_cases();
    for (const auto& c : cases) {
        const mko::test::FdProblem& p = c->p;
        const int P = p.set.size();
        const AdjointJacobian j = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0, c->options);
        const mko::FdJacobian fd =
            mko::fd_jacobian_cpu(p.set, p.base, p.x0, mko::fd_options_of(c->options));
        ASSERT_EQ(j.shape, fd.shape) << c->name;
        EXPECT_TRUE(same_bytes<double>(j.residuals, fd.residuals)) << c->name;
        const std::int64_t rv = j.shape.ray_rows;
        const std::int64_t R = j.shape.rows();
        const std::span<const double> ja = mko::test::run_columns(j, 0);
        const std::span<const double> jf = mko::test::run_jacobian(fd, 0);
        const std::vector<double> ea = mko::test::row_block(ja, P, rv, R);
        const std::vector<double> ef = mko::test::row_block(jf, P, rv, R);
        EXPECT_TRUE(same_bytes<double>(ea, ef)) << c->name << ": Zusatzzeilen";
        std::int64_t changes = 0;
        for (int k = 0; k < P; ++k) {
            changes += fd.status_change(0, k);
        }
        EXPECT_EQ(changes, 0) << c->name;
        // Trace-Zeilen: adjungiert gegen FD; Schranke 10⁻⁶ plus der Fehler der FD-Jacobi selbst
        // gegen die Richardson-Referenz (Dreiecksungleichung), wo diese anwendbar ist
        const std::vector<double> ta = mko::test::row_block(ja, P, 0, rv);
        const std::vector<double> tf = mko::test::row_block(jf, P, 0, rv);
        const auto d = mko::test::adjoint_deviation(ta, tf, p);
        mko::test::print_deviation("FD-Jacobi, Trace-Zeilen", c->name, p, d);
        double fd_error = 0.0;
        if (c->richardson) {
            const ResidualLayout layout(p.base.topology(), c->options.quadrature);
            const auto ref = mko::test::adaptive_reference(p, layout, c->options.extra, rv);
            ASSERT_FALSE(ref.r2.empty()) << c->name;
            fd_error = mko::test::adjoint_deviation(tf, mko::test::row_block(ref.r2, P, 0, rv), p)
                           .scaled.frobenius;
        }
        const auto full = mko::test::adjoint_deviation(ja, jf, p);
        std::printf("    FD-Jacobi gegen Richardson (skaliert): %.3e; mit %d Zusatzzeilen "
                    "(bitgleich): skaliert %.3e\n",
                    fd_error, j.shape.extra_rows, full.scaled.frobenius);
        EXPECT_LT(d.scaled.frobenius, mko::test::kAdjointJacobian + fd_error) << c->name;
        EXPECT_LT(full.scaled.frobenius, mko::test::kAdjointJacobian + fd_error) << c->name;
    }
}

// Kolbenabzug als Projektion: Pro Tupel und Spalte ist Σ_k √(W_k)·J(k, p) = s_{f,l} Σ_k W_k
// (∂OPL_k − mittel) = 0 über die Strahlen im Kolben (bis auf Rundung gegen Σ_k √W_k |J(k, p)|);
// Tupel mit Gewicht 0 (Feld 1, Wellenlängen 0 und 3 des Mehrlinsers) haben Zeilen exakt 0.
TEST(AdjointJacobian, PistonProjectionAndZeroWeights) {
    mko::test::FdProblem p = mko::test::mehrlinser_problem();
    mko::SystemDescription d = p.loaded.system;
    d.source.field_weights = {2.0, 0.0, 1.0};
    d.spectrum.weights = {0.0, 1.0, 1.0, 0.0};
    p = mko::test::make_fd_problem(
        mko::prepare_system(d, std::filesystem::path(mko::test::kResidualSystems)));
    const ResidualLayout layout(p.base.topology());
    const AdjointJacobian j = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0);
    const int P = p.set.size();
    double worst = 0.0;
    for (int f = 0; f < layout.fields(); ++f) {
        const auto& q = layout.quadrature(f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const bool off = f == 1 || l == 0 || l == 3;
            for (int k = 0; k < P; ++k) {
                double sum = 0.0;
                double mag = 0.0;
                for (int i = 0; i < layout.samples(f); ++i) {
                    const double v = j.at(0, layout.offset(f, l) + i, k);
                    if (off) {
                        EXPECT_EQ(v, 0.0);
                    }
                    const double rw = mko::residual_weight(mko::sample(q, i));
                    sum += rw * v;
                    mag += rw * std::abs(v);
                }
                if (!off && mag > 0.0) {
                    worst = std::max(worst, std::abs(sum) / mag);
                }
            }
        }
    }
    std::printf("Kolbenprojektion: max |Σ √W J| / Σ √W |J| = %.2e\n", worst);
    EXPECT_LT(worst, mko::test::kAdjointPistonProjection);
}

// Weiches Versagen (weiches_versagen_start.json, 27 von 90 Strahlen weich versagt): Strafzeilen
// gegen die Richardson-Referenz (Block der Strafresiduen allein), Strafzeilen der Strahlen ohne
// Ereignis exakt 0, Wellenfrontzeilen der weich versagten Strahlen ≠ 0 (sie bleiben im Kolben).
TEST(AdjointJacobian, SoftFailurePenaltyRows) {
    const mko::test::FdProblem p =
        mko::test::make_fd_problem(mko::test::load_residual_system("weiches_versagen_start.json"));
    const ResidualLayout layout(p.base.topology());
    const AdjointJacobian j = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0);
    const int P = p.set.size();
    const std::int64_t rays = layout.rays_per_variant();
    int soft = 0;
    for (std::int64_t i = 0; i < rays; ++i) {
        const auto st = static_cast<TraceStatus>(j.status[static_cast<std::size_t>(i)]);
        double penalty_norm = 0.0;
        double wave_norm = 0.0;
        for (int k = 0; k < P; ++k) {
            penalty_norm += std::abs(j.at(0, rays + i, k));
            wave_norm += std::abs(j.at(0, i, k));
        }
        if (st == TraceStatus::Ok) {
            EXPECT_EQ(penalty_norm, 0.0) << "Strahl " << i;
        } else {
            ++soft;
            EXPECT_TRUE(mko::is_soft_failure(st));
            EXPECT_GT(penalty_norm, 0.0) << "Strahl " << i;
            EXPECT_GT(wave_norm, 0.0) << "Strahl " << i;
        }
    }
    EXPECT_EQ(soft, 27);
    const auto ref = mko::test::adaptive_reference(p, layout, {}, -1);
    ASSERT_FALSE(ref.r2.empty());
    const std::span<const double> jr = mko::test::run_columns(j, 0);
    const auto dp =
        mko::test::adjoint_deviation(mko::test::row_block(jr, P, rays, 2 * rays),
                                     mko::test::row_block(ref.r2, P, rays, 2 * rays), p);
    const auto dw = mko::test::adjoint_deviation(mko::test::row_block(jr, P, 0, rays),
                                                 mko::test::row_block(ref.r2, P, 0, rays), p);
    std::printf("Weiches Versagen: %d weich versagte Strahlen; Strafzeilen skaliert %.3e (größte "
                "Spalte %.3e), Wellenfrontzeilen skaliert %.3e (größte Spalte %.3e)\n",
                soft, dp.scaled.frobenius, dp.plain.max_column, dw.scaled.frobenius,
                dw.plain.max_column);
    EXPECT_LT(dp.plain.max_column, mko::test::kAdjointJacobian);
    EXPECT_LT(dw.plain.max_column, mko::test::kAdjointJacobian);
}

// Strafgradient einzelner Strahlen über die ganze Kette, auch für Strahlen, die nach einem
// weichen Ereignis hart versagen (Festanteil ohne Gradient, nur die weichen Ereignisse davor):
// weiches_versagen_start.json und Plankonvexlinse mit Totalreflexion an der Rückfläche
// (tir_problem mit stärkerer Krümmung); Referenz Richardson der Strafgröße π aus
// trace_to_reference_sphere.
TEST(AdjointRay, PenaltyGradientMatchesRichardson) {
    std::vector<mko::test::FdProblem> problems;
    problems.push_back(
        mko::test::make_fd_problem(mko::test::load_residual_system("weiches_versagen_start.json")));
    {
        const mko::test::FdProblem tir = mko::test::tir_problem();
        mko::SystemDescription d = tir.loaded.system;
        d.surfaces[2].curvature *= 1.25;
        problems.push_back(mko::test::make_fd_problem(
            mko::prepare_system(d, std::filesystem::path(mko::test::kResidualSystems))));
    }
    {
        // Geisterplatte mit starkem Profil a₁ = −8 mm (r_g = 10 mm): Für r ≳ 7 mm ist der Radikand
        // < 0, der Strahl läuft tangential weiter (d_z = 0) und versagt an der nächsten Fläche
        // hart; die Strafe −Radikand hängt von den Koeffizienten ab.
        const mko::test::FdProblem g = mko::test::ghost_gap_problem();
        mko::SystemDescription d = g.loaded.system;
        d.ghost_plates[0].coefficients[0] = -8.0;
        problems.push_back(mko::test::make_fd_problem(
            mko::prepare_system(d, std::filesystem::path(mko::test::kResidualSystems))));
    }
    double worst = 0.0;
    int soft_rays = 0;
    int hard_after_soft = 0;
    for (const mko::test::FdProblem& p : problems) {
        const ResidualLayout layout(p.base.topology());
        const std::vector<double> typical = reference_typical(p.set);
        // Trace eines Strahls am Punkt x (mit Aufzeichnung, wenn rec gesetzt)
        const auto trace = [&](std::span<const double> x, int f, int l, int k,
                               mko::AdjointRecord<double>* rec, TraceStatus* st,
                               Vec3<double>* center_out) {
            const mko::SystemBatchHost b = mko::make_adjoint_batch(p.set, p.base, x);
            const auto view = b.view();
            const auto table = view.surface_table(0);
            const Vec3<double> center = mko::batch_image_point(view, table, 0, f);
            mko::SoftFailure<double> soft{0.0, TraceStatus::Ok};
            mko::NoSurfaceObserver* const none = nullptr;
            double opl = 0.0;
            const auto s = mko::sample(layout.quadrature(f), k);
            if (rec != nullptr) {
                *st =
                    mko::batch_ray_opl(view, table, l, 0, f, l, s, center, &opl, none, &soft, rec);
            } else {
                mko::NoTraceRecorder* const no_record = nullptr;
                *st = mko::batch_ray_opl(view, table, l, 0, f, l, s, center, &opl, none, &soft,
                                         no_record);
            }
            if (center_out != nullptr) {
                *center_out = center;
            }
            return soft;
        };
        const mko::SystemBatchHost b0 = mko::make_adjoint_batch(p.set, p.base, p.x0);
        const mko::AdjointParameters params(p.set, b0);
        for (int f = 0; f < layout.fields(); ++f) {
            for (int l = 0; l < layout.wavelengths(); ++l) {
                for (int k = 0; k < layout.samples(f); ++k) {
                    const auto rec = std::make_unique<mko::AdjointRecord<double>>();
                    TraceStatus st;
                    Vec3<double> center{};
                    const mko::SoftFailure<double> s0 =
                        trace(p.x0, f, l, k, rec.get(), &st, &center);
                    if (s0.first == TraceStatus::Ok) {
                        continue;
                    }
                    ++soft_rays;
                    if (mko::is_hard_failure(st)) {
                        ++hard_after_soft;
                    }
                    std::vector<double> g(static_cast<std::size_t>(p.set.size()), 0.0);
                    const mko::JacobianRowWriter<double> w{params.view(), l, g.data(), 1, 0, 1.0};
                    mko::backward_ray(b0.view().surface_table(0), l, *rec, mko::in_piston(st),
                                      center, false, AdjointSeed<double>{0.0, 1.0}, w);
                    const auto f_pen = [&](const std::vector<double>& x) {
                        TraceStatus s2;
                        return trace(x, f, l, k, nullptr, &s2, nullptr).penalty;
                    };
                    const std::vector<double> ref = richardson_gradient(f_pen, p.x0, typical, 1e-2);
                    const double e = relative_error(g, ref);
                    worst = std::max(worst, e);
                    EXPECT_LT(e, mko::test::kAdjointRayRichardson)
                        << "Feld " << f << ", Stützstelle " << k << ", Status "
                        << mko::trace_status_name(st);
                }
            }
        }
    }
    std::printf("Strafgradient: %d Strahlen mit weichem Ereignis (%d danach hart versagt), max. "
                "relativ %.2e\n",
                soft_rays, hard_after_soft, worst);
    EXPECT_GT(soft_rays, 20);
    EXPECT_GT(hard_after_soft, 0);
}

// Hartes Versagen (FailureMode::Hard): keine Strafzeilen, hart versagte Strahlen haben Zeilen 0,
// die übrigen gegen die Richardson-Referenz (weiches_versagen_start.json im harten Modus,
// 27 versagte Strahlen aus dem Kolben ausgeschlossen, Gewichte renormiert).
TEST(AdjointJacobian, HardFailureMode) {
    const mko::test::FdProblem p =
        mko::test::make_fd_problem(mko::test::load_residual_system("weiches_versagen_start.json"));
    AdjointJacobianOptions o;
    o.quadrature.failure = mko::FailureMode::Hard;
    const ResidualLayout layout(p.base.topology(), o.quadrature);
    const AdjointJacobian j = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0, o);
    EXPECT_EQ(j.shape.ray_rows, layout.rays_per_variant());
    EXPECT_EQ(j.hard_failed_count(), 27u);
    for (std::int64_t i = 0; i < layout.rays_per_variant(); ++i) {
        if (mko::is_hard_failure(static_cast<TraceStatus>(j.status[static_cast<std::size_t>(i)]))) {
            for (int k = 0; k < p.set.size(); ++k) {
                EXPECT_EQ(j.at(0, i, k), 0.0);
            }
        }
    }
    const auto ref = mko::test::adaptive_reference(p, layout, {}, -1);
    ASSERT_FALSE(ref.r2.empty());
    const auto d = mko::test::adjoint_deviation(mko::test::run_columns(j, 0), ref.r2, p);
    mko::test::print_deviation("Richardson, hart", "Weiches Versagen im harten Modus", p, d);
    EXPECT_LT(d.plain.max_column, mko::test::kAdjointJacobian);
    EXPECT_LT(d.scaled.frobenius, mko::test::kAdjointJacobian);
}

// Mehrere Läufe in einem Aufruf (gebündelter LM): Lauf i ist bitgleich zum Einzelaufruf mit
// x₀_i; Vorlage mit einer Variante pro Lauf ebenso.
TEST(AdjointJacobian, MultipleRunsAreIndependent) {
    const mko::test::FdProblem p = mko::test::ghost_gap_problem();
    const int P = p.set.size();
    std::vector<double> x0;
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < P; ++k) {
            x0.push_back(p.x0[static_cast<std::size_t>(k)] *
                         (1.0 + 0.01 * (i + 1) * ((k % 3) - 1)));
        }
    }
    AdjointJacobianOptions o;
    o.extra = mko::test::paraxial_extra(p);
    const AdjointJacobian all = mko::adjoint_jacobian_cpu(p.set, p.base, x0, o);
    ASSERT_EQ(all.shape.runs, 3);
    mko::SystemBatchHost per_run = p.base;
    per_run.resize(3);
    for (int i = 0; i < 3; ++i) {
        per_run.copy_variant(0, i);
    }
    const AdjointJacobian all2 = mko::adjoint_jacobian_cpu(p.set, per_run, x0, o);
    EXPECT_TRUE(same_bytes<double>(all.jacobian, all2.jacobian));
    EXPECT_TRUE(same_bytes<double>(all.residuals, all2.residuals));
    for (int i = 0; i < 3; ++i) {
        const auto xi = std::span<const double>(x0).subspan(static_cast<std::size_t>(i * P),
                                                            static_cast<std::size_t>(P));
        const AdjointJacobian one = mko::adjoint_jacobian_cpu(p.set, p.base, xi, o);
        EXPECT_TRUE(
            same_bytes<double>(mko::test::run_columns(all, i), mko::test::run_columns(one, 0)))
            << "Lauf " << i;
        EXPECT_TRUE(same_bytes<double>(all.residual(i), one.residual(0))) << "Lauf " << i;
    }
}

// Fehlerfälle: x0 mit falscher Länge, Vorlage mit falscher Variantenzahl, Zusatzresiduen ohne
// Funktion, Spaltenzuordnung für einen Batch anderer Topologie, Zusatzzeilen falscher Größe.
TEST(AdjointJacobian, ErrorCases) {
    const mko::test::FdProblem p = mko::test::einzellinse_problem();
    EXPECT_THROW(mko::adjoint_jacobian_cpu(p.set, p.base, std::vector<double>{1.0, 2.0, 3.0}),
                 std::invalid_argument);
    EXPECT_THROW(mko::adjoint_jacobian_cpu(p.set, p.base, std::vector<double>{}),
                 std::invalid_argument);
    mko::SystemBatchHost two = p.base;
    two.resize(2);
    std::vector<double> x3 = p.x0;
    x3.insert(x3.end(), p.x0.begin(), p.x0.end());
    x3.insert(x3.end(), p.x0.begin(), p.x0.end());
    EXPECT_THROW(mko::adjoint_jacobian_cpu(p.set, two, x3), std::invalid_argument);
    AdjointJacobianOptions o;
    o.extra.count = 2;
    EXPECT_THROW(mko::adjoint_jacobian_cpu(p.set, p.base, p.x0, o), std::invalid_argument);
    const mko::test::FdProblem other = mko::test::mehrlinser_problem();
    EXPECT_THROW(mko::AdjointParameters(p.set, other.base), std::invalid_argument);
    AdjointJacobian j = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0);
    mko::ExtraRowsJacobian e;
    e.runs = 1;
    e.variables = p.set.size();
    e.rows = 1;
    EXPECT_THROW(mko::insert_extra_rows(e, j), std::invalid_argument);
}
