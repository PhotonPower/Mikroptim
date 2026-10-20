// Dezentrierung und Verkippung einzelner Flächen (M8-T3, ADR-0011), CPU: Transformation,
// analytische Orakel (verkippte Planplatte, dezentrierte und verkippte Kugel, Verkippung um den
// Scheitel gegen verschobenen Krümmungsmittelpunkt, Elementverkippung um einen Punkt gegen die
// direkte Rechnung, Translation einer dezentrierten Linse, Bildversatz Δ(1 − m), Koma linear in
// Δ, Spiegelsymmetrie der OPD-Karte), Transformation 0 bitgleich zum koaxialen Pfad in allen
// CPU-Referenzen, adjungierte Jacobi-Matrix gegen Richardson, Datei, Batch und CLI. GPU-Teil:
// test_cuda_decenter.cpp; Fingerabdrücke gegen main: test_coaxial_fingerprint.cpp.

#include "adjoint_test_systems.hpp"
#include "cli/cli_util.hpp"
#include "cli/eval_command.hpp"
#include "cli/trace_command.hpp"
#include "decenter_test_systems.hpp"
#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/surface_transform.hpp"
#include "mikroptim/trace/trace.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using mko::HostSurfaceTable;
using mko::Ray;
using mko::SurfaceKind;
using mko::SurfaceTransform;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::kDecenterDeg;

namespace {

Vec3<double> cross(Vec3<double> a, Vec3<double> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double dist(Vec3<double> a, Vec3<double> b) {
    return mko::norm(a - b);
}

// Explizite Drehmatrix R = R_y(β) R_x(α) angewandt auf v (unabhängig von surface_transform.hpp).
Vec3<double> rotate_explicit(double alpha, double beta, Vec3<double> v) {
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);
    const double cb = std::cos(beta);
    const double sb = std::sin(beta);
    const double m[3][3] = {{cb, sb * sa, sb * ca}, {0.0, ca, -sa}, {-sb, cb * sa, cb * ca}};
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
}

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Kollimierter Strahl parallel zur Achse in der Blendenebene.
Ray<double> axial_ray(double x, double y) {
    return mko::collimated_ray(1.0, 0.0, std::hypot(x, y), std::atan2(x, y));
}

// Trace bis zur Bildebene (Abstand thickness[S − 1] hinter der letzten Fläche, nominell).
TraceStatus trace_to_image(const HostSurfaceTable& t, Ray<double>& ray, int l = 0) {
    const TraceStatus st = mko::trace_ray(t.view(), l, ray, nullptr);
    if (st != TraceStatus::Ok) {
        return st;
    }
    return mko::transfer_to_image_plane(t.view(), l, ray);
}

// Gewichtete kleinste Quadrate (Normalgleichungen, Gauß mit Pivotsuche): Koeffizienten zu den
// Basisfunktionen basis(x, y, out).
template <int N, class Basis>
std::array<double, N> least_squares(const std::vector<std::array<double, 3>>& pts, Basis basis) {
    double a[N][N + 1] = {};
    for (const auto& p : pts) {
        double b[N];
        basis(p[0], p[1], b);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                a[i][j] += b[i] * b[j];
            }
            a[i][N] += b[i] * p[2];
        }
    }
    for (int c = 0; c < N; ++c) {
        int piv = c;
        for (int r = c + 1; r < N; ++r) {
            if (std::abs(a[r][c]) > std::abs(a[piv][c])) {
                piv = r;
            }
        }
        for (int k = 0; k <= N; ++k) {
            std::swap(a[c][k], a[piv][k]);
        }
        for (int r = 0; r < N; ++r) {
            if (r == c) {
                continue;
            }
            const double f = a[r][c] / a[c][c];
            for (int k = c; k <= N; ++k) {
                a[r][k] -= f * a[c][k];
            }
        }
    }
    std::array<double, N> x{};
    for (int i = 0; i < N; ++i) {
        x[static_cast<std::size_t>(i)] = a[i][N] / a[i][i];
    }
    return x;
}

} // namespace

// ---- Transformation -------------------------------------------------------------------------

// R = R_y(β) R_x(α) aktiv und rechtshändig (ADR-0011): gegen die explizite Matrix, Rᵀ R = 1,
// Hin- und Rücktransformation, Vorzeichen (α > 0 dreht +y nach +z, β > 0 dreht +z nach +x).
TEST(SurfaceTransform, RotationMatchesConvention) {
    std::mt19937_64 rng(83001);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double worst_matrix = 0.0;
    double worst_round_trip = 0.0;
    for (int i = 0; i < 200; ++i) {
        const double alpha = 1.4 * u(rng);
        const double beta = 1.4 * u(rng);
        const SurfaceTransform<double> t = mko::make_surface_transform(u(rng), u(rng), alpha, beta);
        const Vec3<double> v{u(rng), u(rng), u(rng)};
        worst_matrix = std::max(
            worst_matrix, dist(mko::rotate_from_surface(t, v), rotate_explicit(alpha, beta, v)));
        worst_round_trip = std::max(
            worst_round_trip, dist(mko::rotate_to_surface(t, mko::rotate_from_surface(t, v)), v));
        worst_round_trip = std::max(
            worst_round_trip, dist(mko::point_from_surface(t, mko::point_to_surface(t, v)), v));
        EXPECT_NEAR(mko::norm(mko::rotate_to_surface(t, v)), mko::norm(v), mko::test::kUnitFp64);
    }
    EXPECT_LT(worst_matrix, mko::test::kUnitFp64);
    EXPECT_LT(worst_round_trip, mko::test::kUnitFp64);
    const double a = 0.3;
    const SurfaceTransform<double> tx = mko::make_surface_transform(0.0, 0.0, a, 0.0);
    const Vec3<double> ey = mko::rotate_from_surface(tx, Vec3<double>{0.0, 1.0, 0.0});
    EXPECT_NEAR(ey.y, std::cos(a), 1e-16);
    EXPECT_NEAR(ey.z, std::sin(a), 1e-16);
    const SurfaceTransform<double> ty = mko::make_surface_transform(0.0, 0.0, 0.0, a);
    const Vec3<double> ez = mko::rotate_from_surface(ty, Vec3<double>{0.0, 0.0, 1.0});
    EXPECT_NEAR(ez.x, std::sin(a), 1e-16);
    EXPECT_NEAR(ez.z, std::cos(a), 1e-16);
    // Verschiebung: der Scheitel (Ursprung des gekippten Systems) liegt bei (Δx, Δy, 0)
    const SurfaceTransform<double> td = mko::make_surface_transform(0.2, -0.1, 0.4, -0.3);
    const Vec3<double> vertex = mko::point_from_surface(td, Vec3<double>{0.0, 0.0, 0.0});
    EXPECT_EQ(vertex.x, 0.2);
    EXPECT_EQ(vertex.y, -0.1);
    EXPECT_EQ(vertex.z, 0.0);
    std::printf("[ Info     ] R gegen explizite Matrix %.1e, Hin und zurück %.1e\n", worst_matrix,
                worst_round_trip);
}

// Transformation 0 im Tracer: Tabelle mit Transformationsdaten, alle Werte 0, ist bitgleich zur
// koaxialen Tabelle (Aufzeichnung aller Zwischenzustände).
TEST(SurfaceTransform, ZeroTransformIsBitIdenticalInTracer) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    HostSurfaceTable moved = ls.table;
    mko::set_surface_transform(moved, 1, 0.0, 0.0, 0.0, 0.0);
    ASSERT_FALSE(moved.coaxial());
    const mko::PupilQuadrature<double> q = mko::make_pupil_quadrature(8, 12, mko::PupilMode::Full);
    int compared = 0;
    for (const double theta : {0.0, 0.02, 0.05}) {
        for (int k = 0; k < mko::sample_count(q); ++k) {
            const mko::PupilSample<double> s = mko::sample(q, k);
            for (int l = 0; l < ls.table.wavelengths; ++l) {
                Ray<double> a = mko::collimated_ray(4.0, theta, s.rho, s.phi);
                Ray<double> b = a;
                mko::TraceRecord<double> ra{};
                mko::TraceRecord<double> rb{};
                const TraceStatus sa = mko::trace_ray(ls.table.view(), l, a, &ra);
                const TraceStatus sb = mko::trace_ray(moved.view(), l, b, &rb);
                ASSERT_EQ(sa, sb);
                ASSERT_EQ(std::memcmp(&a, &b, sizeof(a)), 0);
                ASSERT_EQ(ra.count, rb.count);
                ASSERT_EQ(std::memcmp(ra.p, rb.p, sizeof(Vec3<double>) * ra.count), 0);
                ASSERT_EQ(std::memcmp(ra.d, rb.d, sizeof(Vec3<double>) * ra.count), 0);
                ASSERT_EQ(std::memcmp(ra.opl, rb.opl, sizeof(double) * ra.count), 0);
                ++compared;
            }
        }
    }
    EXPECT_GT(compared, 1000);
}

// ---- Analytische Orakel ---------------------------------------------------------------------

// Planparallele Platte (Dicke T senkrecht zu den Flächen, n = 1,5) um α verkippt: Strahlversatz
// Δ = T sin(α − α′)/cos α′ quer zum Strahl und OPL-Zuwachs T (n cos α′ − cos α) gegen Luft. Zwei
// Beschreibungen: Elementdrehung um den ersten Scheitel nach ADR-0011 (zweite Fläche um
// (0, −T sin α) dezentriert, Abstand T cos α) und beide Flächen um ihren Scheitel verkippt mit
// Abstand T/cos α (dieselbe Platte); um x (Versatz in −y) und um y (Versatz in +x).
TEST(DecenterOracle, TiltedPlaneParallelPlate) {
    const double T = 3.0;
    const double n = 1.5;
    const double d1 = 2.0;
    const double w = 5.0;
    double worst_pos = 0.0;
    double worst_opl = 0.0;
    double worst_dir = 0.0;
    for (const double deg : {0.5, 5.0, 20.0, 40.0}) {
        const double a = deg * kDecenterDeg;
        const double ap = std::asin(std::sin(a) / n);
        const double shift = T * std::sin(a - ap) / std::cos(ap);
        const double extra = T * (n * std::cos(ap) - std::cos(a));
        for (const bool about_y : {false, true}) {
            for (const bool element : {true, false}) {
                const double alpha = about_y ? 0.0 : a;
                const double beta = about_y ? a : 0.0;
                const double gap = element ? T * std::cos(a) : T / std::cos(a);
                HostSurfaceTable t = mko::test::plate_table(n, d1, gap, w);
                mko::set_surface_transform(t, 1, 0.0, 0.0, alpha, beta);
                if (element) {
                    // P + R(V − P) − V für V = (0, 0, T) und P = erster Scheitel
                    const Vec3<double> v2 = rotate_explicit(alpha, beta, Vec3<double>{0, 0, T});
                    mko::set_surface_transform(t, 2, v2.x, v2.y, alpha, beta);
                } else {
                    mko::set_surface_transform(t, 2, 0.0, 0.0, alpha, beta);
                }
                const double z_image = d1 + gap + w;
                for (const auto& xy : {std::array<double, 2>{0.0, 0.0}, {0.7, -1.1}, {-1.3, 0.4}}) {
                    Ray<double> r = axial_ray(xy[0], xy[1]);
                    ASSERT_EQ(trace_to_image(t, r), TraceStatus::Ok);
                    const double ex = about_y ? xy[0] + shift : xy[0];
                    const double ey = about_y ? xy[1] : xy[1] - shift;
                    worst_pos = std::max(worst_pos, std::hypot(r.p.x - ex, r.p.y - ey));
                    worst_opl = std::max(worst_opl, std::abs(r.opl - (z_image + extra)));
                    worst_dir = std::max(worst_dir, dist(r.d, Vec3<double>{0.0, 0.0, 1.0}));
                }
            }
        }
    }
    std::printf("[ Info     ] Planplatte: Versatz %.1e mm, OPL %.1e mm, Richtung %.1e\n", worst_pos,
                worst_opl, worst_dir);
    EXPECT_LT(worst_pos, mko::test::kLengthFp64);
    EXPECT_LT(worst_opl, mko::test::kLengthFp64);
    EXPECT_LT(worst_dir, mko::test::kUnitFp64);
}

// Dezentrierte und verkippte Kugelfläche: Der Schnittpunkt liegt auf der Kugel um
// C = Δ + R·(R_y(β) R_x(α) e_z)·(1/c) (unabhängige quadratische Gleichung, Schnittpunkt nahe dem
// Scheitel), die gebrochene Richtung erfüllt das Brechungsgesetz mit der Normalen
// N = −c (p − C): |d′| = 1, n |d × N| = n′ |d′ × N|, d′ − (n/n′) d ∥ N, und die OPL wächst um
// n · |p − p₀|.
TEST(DecenterOracle, DecenteredSphereIntersectionAndSnell) {
    std::mt19937_64 rng(83003);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double worst_point = 0.0;
    double worst_snell = 0.0;
    double worst_opl = 0.0;
    int traced = 0;
    for (int i = 0; i < 400; ++i) {
        const double c = (i % 2 == 0 ? 1.0 : -1.0) * (0.02 + 0.08 * (0.5 + 0.5 * u(rng)));
        const double n0 = 1.0 + 0.3 * (0.5 + 0.5 * u(rng));
        const double n1 = 1.4 + 0.4 * (0.5 + 0.5 * u(rng));
        const double alpha = 0.2 * u(rng);
        const double beta = 0.2 * u(rng);
        const SurfaceTransform<double> xf =
            mko::make_surface_transform(0.5 * u(rng), 0.5 * u(rng), alpha, beta);
        const double t = 3.0 + u(rng);
        // Start in der Ebene der vorigen Fläche (z = 0), Abstand t zum Scheitel
        Ray<double> r;
        r.p = Vec3<double>{2.0 * u(rng), 2.0 * u(rng), 0.0};
        const double dx = 0.1 * u(rng);
        const double dy = 0.1 * u(rng);
        r.d = Vec3<double>{dx, dy, std::sqrt(1.0 - dx * dx - dy * dy)};
        r.opl = 0.0;
        const Ray<double> r0 = r;
        const TraceStatus st = mko::propagate_surface(r, t, SurfaceKind::Sphere, c, n0, n1, nullptr,
                                                      nullptr, nullptr, &xf);
        if (st != TraceStatus::Ok) {
            continue;
        }
        ++traced;
        // Unabhängig: Kugel im nominellen System der Fläche (Scheitel bei z = 0)
        const Vec3<double> axis = rotate_explicit(alpha, beta, Vec3<double>{0.0, 0.0, 1.0});
        const Vec3<double> C = Vec3<double>{xf.dx, xf.dy, 0.0} + (1.0 / c) * axis;
        const Vec3<double> p0{r0.p.x, r0.p.y, r0.p.z - t};
        const Vec3<double> q = p0 - C;
        const double b = mko::dot(q, r0.d);
        const double g = mko::dot(q, q) - 1.0 / (c * c);
        const double root = std::sqrt(b * b - g);
        // Schnittpunkt auf der dem Scheitel zugewandten Kappe: (p − C)·axis hat das Vorzeichen
        // von −1/c
        const double s1 = -b - root;
        const double s2 = -b + root;
        const Vec3<double> p1 = p0 + s1 * r0.d;
        const Vec3<double> p2 = p0 + s2 * r0.d;
        const Vec3<double> pe = mko::dot(p1 - C, axis) * c < 0.0 ? p1 : p2;
        worst_point = std::max(worst_point, dist(r.p, pe));
        worst_opl = std::max(worst_opl, std::abs(r.opl - n0 * dist(pe, p0)));
        const Vec3<double> N = (-c) * (r.p - C);
        const double cos_i = mko::dot(r0.d, N);
        worst_snell = std::max(worst_snell, std::abs(mko::norm(r.d) - 1.0));
        worst_snell = std::max(
            worst_snell, std::abs(n0 * mko::norm(cross(r0.d, N)) - n1 * mko::norm(cross(r.d, N))));
        worst_snell = std::max(worst_snell, mko::norm(cross(r.d - (n0 / n1) * r0.d, N)));
        EXPECT_GT(cos_i * mko::dot(r.d, N), 0.0); // Brechung, keine Reflexion
    }
    std::printf("[ Info     ] Kugel: %d Strahlen, Schnittpunkt %.1e mm, OPL %.1e mm, Snellius "
                "%.1e\n",
                traced, worst_point, worst_opl, worst_snell);
    EXPECT_GT(traced, 300);
    EXPECT_LT(worst_point, mko::test::kLengthFp64);
    EXPECT_LT(worst_opl, mko::test::kLengthFp64);
    EXPECT_LT(worst_snell, mko::test::kUnitFp64);
}

// Verkippung einer Kugel um ihren Scheitel = Verschiebung des Krümmungsmittelpunkts: Mit
// R·(R_y(β) R_x(α) e_z) wandert der Mittelpunkt lateral um R (sin β cos α, −sin α) und axial um
// R (cos α cos β − 1) (für β = 0: R sin α und R (1 − cos α)). Beschreibung B dezentriert die
// unverkippte Kugel entsprechend und verschiebt ihren Scheitel axial (Dicken davor und danach).
// Beide Beschreibungen ergeben bis auf Rundung denselben Trace (Einzellinse, alle Felder und
// Wellenlängen, volle Pupille, beide Flächen einzeln).
TEST(DecenterOracle, TiltAboutVertexEqualsShiftedCenter) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    double worst_pos = 0.0;
    double worst_dir = 0.0;
    double worst_opl = 0.0;
    int rays = 0;
    for (const int s : {1, 2}) {
        for (const auto& ab :
             {std::array<double, 2>{0.5, 0.0}, {2.0, 0.0}, {0.0, -1.5}, {1.0, 0.7}, {-3.0, 2.0}}) {
            const double alpha = ab[0] * kDecenterDeg;
            const double beta = ab[1] * kDecenterDeg;
            const double R = 1.0 / ls.table.curvature[static_cast<std::size_t>(s)];
            HostSurfaceTable a = ls.table;
            mko::set_surface_transform(a, s, 0.0, 0.0, alpha, beta);
            const Vec3<double> center = R * rotate_explicit(alpha, beta, Vec3<double>{0, 0, 1});
            const Vec3<double> vertex = center - Vec3<double>{0.0, 0.0, R};
            HostSurfaceTable b = ls.table;
            mko::set_surface_transform(b, s, vertex.x, vertex.y, 0.0, 0.0);
            b.thickness[static_cast<std::size_t>(s - 1)] += vertex.z;
            b.thickness[static_cast<std::size_t>(s)] -= vertex.z;
            const mko::PupilQuadrature<double> q =
                mko::make_pupil_quadrature(6, 8, mko::PupilMode::Full);
            for (const double theta : {0.0, 2.0 * kDecenterDeg}) {
                for (int k = 0; k < mko::sample_count(q); ++k) {
                    const mko::PupilSample<double> smp = mko::sample(q, k);
                    for (int l = 0; l < a.wavelengths; ++l) {
                        Ray<double> ra = mko::collimated_ray(10.0, theta, smp.rho, smp.phi);
                        Ray<double> rb = ra;
                        const TraceStatus sa = trace_to_image(a, ra, l);
                        const TraceStatus sb = trace_to_image(b, rb, l);
                        ASSERT_EQ(sa, sb);
                        if (sa != TraceStatus::Ok) {
                            continue;
                        }
                        worst_pos = std::max(worst_pos, dist(ra.p, rb.p));
                        worst_dir = std::max(worst_dir, dist(ra.d, rb.d));
                        worst_opl = std::max(worst_opl, std::abs(ra.opl - rb.opl));
                        ++rays;
                    }
                }
            }
        }
    }
    std::printf(
        "[ Info     ] Kippung um den Scheitel gegen Mittelpunktsverschiebung (%d Strahlen): "
        "Ort %.1e mm, Richtung %.1e, OPL %.1e mm\n",
        rays, worst_pos, worst_dir, worst_opl);
    EXPECT_GT(rays, 1000);
    EXPECT_LT(worst_pos, mko::test::kLengthFp64);
    EXPECT_LT(worst_dir, mko::test::kUnitFp64);
    EXPECT_LT(worst_opl, mko::test::kLengthFp64);
}

// Elementverkippung einer Linse um einen Punkt P nach ADR-0011 (Verkippung R an beiden Flächen,
// Dezentrierung = Lateralanteil von P + R(V − P) − V, Axialanteil in den Dicken) gegen die direkte
// Rechnung: Strahlen mit Rᵀ um P ins System der unverkippten Linse drehen, koaxial durch die
// Linse, mit R zurück und bis zur festen Bildebene.
TEST(DecenterOracle, ElementTiltAboutPointMatchesDirectRotation) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const HostSurfaceTable& base = ls.table;
    const double z1 = base.thickness[0];
    const double z2 = z1 + base.thickness[1];
    const double z_image = z2 + base.thickness[2];
    double worst_pos = 0.0;
    double worst_dir = 0.0;
    double worst_opl = 0.0;
    int rays = 0;
    for (const auto& c : {std::array<double, 4>{1.0, -0.7, 0.3, 3.0},
                          {-2.0, 0.5, -0.2, -10.0},
                          {0.3, 1.2, 0.0, 50.0}}) {
        const double alpha = c[0] * kDecenterDeg;
        const double beta = c[1] * kDecenterDeg;
        const Vec3<double> P{0.0, c[2], z1 + c[3]};
        const SurfaceTransform<double> rot = mko::make_surface_transform(0.0, 0.0, alpha, beta);
        HostSurfaceTable t = base;
        double prev = 0.0;
        for (const int s : {1, 2}) {
            const Vec3<double> V{0.0, 0.0, s == 1 ? z1 : z2};
            const Vec3<double> moved = P + mko::rotate_from_surface(rot, V - P) - V;
            mko::set_surface_transform(t, s, moved.x, moved.y, alpha, beta);
            t.thickness[static_cast<std::size_t>(s - 1)] += moved.z - prev;
            prev = moved.z;
        }
        t.thickness[2] -= prev;
        for (const double theta : {0.0, 1.5 * kDecenterDeg}) {
            for (const auto& xy :
                 {std::array<double, 2>{0.0, 0.0}, {3.0, -2.0}, {-6.0, 5.0}, {7.5, 1.0}}) {
                for (int l = 0; l < base.wavelengths; ++l) {
                    const double r = std::hypot(xy[0], xy[1]);
                    const Ray<double> start =
                        mko::collimated_ray(1.0, theta, r, std::atan2(xy[0], xy[1]));
                    Ray<double> a = start;
                    ASSERT_EQ(trace_to_image(t, a, l), TraceStatus::Ok);
                    // Direkt: ins System der Linse (Drehung um P), koaxial, zurück
                    Ray<double> b = start;
                    b.p = P + mko::rotate_to_surface(rot, b.p - P);
                    b.d = mko::rotate_to_surface(rot, b.d);
                    const double n_air = base.index[static_cast<std::size_t>(l)];
                    const auto L = static_cast<std::size_t>(base.wavelengths);
                    const double n_glass = base.index[L + static_cast<std::size_t>(l)];
                    ASSERT_EQ(mko::propagate_surface(b, z1, SurfaceKind::Sphere, base.curvature[1],
                                                     n_air, n_glass),
                              TraceStatus::Ok);
                    ASSERT_EQ(mko::propagate_surface(b, z2 - z1, SurfaceKind::Sphere,
                                                     base.curvature[2], n_glass, n_air),
                              TraceStatus::Ok);
                    b.p.z += z2;
                    b.p = P + mko::rotate_from_surface(rot, b.p - P);
                    b.d = mko::rotate_from_surface(rot, b.d);
                    const double s = (z_image - b.p.z) / b.d.z;
                    b.p = b.p + s * b.d;
                    b.opl += n_air * s;
                    worst_pos = std::max(worst_pos, std::hypot(a.p.x - b.p.x, a.p.y - b.p.y));
                    worst_dir = std::max(worst_dir, dist(a.d, b.d));
                    worst_opl = std::max(worst_opl, std::abs(a.opl - b.opl));
                    ++rays;
                }
            }
        }
    }
    std::printf("[ Info     ] Elementverkippung um P gegen direkte Drehung (%d Strahlen): Ort %.1e "
                "mm, Richtung %.1e, OPL %.1e mm\n",
                rays, worst_pos, worst_dir, worst_opl);
    EXPECT_LT(worst_pos, mko::test::kLengthFp64);
    EXPECT_LT(worst_dir, mko::test::kUnitFp64);
    EXPECT_LT(worst_opl, mko::test::kLengthFp64);
}

// Dezentrierte Einzellinse (beide Flächen um Δ): exakt gleich der koaxialen Linse mit um −Δ
// verschobenen Startpunkten (Translation), Bildpunkte um Δ versetzt, OPL um Δ·d₀ (Start-OPL der
// ebenen Welle). Der "gleichwertige Fall" eines Feldes ist damit die koaxiale Linse mit
// dezentrierter Pupille.
TEST(DecenterOracle, DecenteredLensIsTranslatedLens) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const double dx = 0.37;
    const double dy = -0.52;
    HostSurfaceTable t = ls.table;
    mko::set_surface_transform(t, 1, dx, dy, 0.0, 0.0);
    mko::set_surface_transform(t, 2, dx, dy, 0.0, 0.0);
    double worst_pos = 0.0;
    double worst_opl = 0.0;
    double worst_dir = 0.0;
    const mko::PupilQuadrature<double> q = mko::make_pupil_quadrature(6, 8, mko::PupilMode::Full);
    for (const double theta : {0.0, 1.0 * kDecenterDeg, 2.0 * kDecenterDeg}) {
        for (int k = 0; k < mko::sample_count(q); ++k) {
            const mko::PupilSample<double> s = mko::sample(q, k);
            for (int l = 0; l < t.wavelengths; ++l) {
                Ray<double> a = mko::collimated_ray(10.0, theta, s.rho, s.phi);
                Ray<double> b = a;
                b.p.x -= dx;
                b.p.y -= dy;
                b.opl = mko::dot(b.p, b.d); // Start-OPL der verschobenen Stützstelle
                const double opl_shift = dx * a.d.x + dy * a.d.y;
                ASSERT_EQ(trace_to_image(t, a, l), TraceStatus::Ok);
                ASSERT_EQ(trace_to_image(ls.table, b, l), TraceStatus::Ok);
                worst_pos =
                    std::max(worst_pos, std::hypot(a.p.x - (b.p.x + dx), a.p.y - (b.p.y + dy)));
                worst_opl = std::max(worst_opl, std::abs(a.opl - (b.opl + opl_shift)));
                worst_dir = std::max(worst_dir, dist(a.d, b.d));
            }
        }
    }
    std::printf("[ Info     ] Translation: Ort %.1e mm, OPL %.1e mm, Richtung %.1e\n", worst_pos,
                worst_opl, worst_dir);
    EXPECT_LT(worst_pos, mko::test::kLengthFp64);
    EXPECT_LT(worst_opl, mko::test::kLengthFp64);
    EXPECT_LT(worst_dir, mko::test::kUnitFp64);
}

// Bildversatz einer dezentrierten Einzellinse ≈ Δ (1 − m): kollimiert (m = 0) über den
// Hauptstrahl in der Bildebene (paraxialer Fokus), endliche Konjugation (Punktquelle auf der
// Achse) über den Schnitt zweier benachbarter Strahlen durch die Linsenmitte, m aus demselben
// Verfahren für die koaxiale Linse und eine Quelle neben der Achse. Die Abweichung ist von
// höherer Ordnung: relativ ∝ Δ² (Verhältnis bei Δ und Δ/2 ≈ 4).
TEST(DecenterOracle, ImageShiftOfDecenteredLens) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    // Bildebene im paraxialen Fokus bei 587,56 nm (Achsschnittpunkt eines achsnahen Strahls)
    HostSurfaceTable focus = ls.table;
    {
        Ray<double> r = axial_ray(0.0, 1e-4);
        ASSERT_EQ(mko::trace_ray(focus.view(), 1, r, nullptr), TraceStatus::Ok);
        focus.thickness[2] = mko::meridional_axis_crossing(r);
    }
    const auto shift_collimated = [&](double delta) {
        HostSurfaceTable t = focus;
        mko::set_surface_transform(t, 1, 0.0, delta, 0.0, 0.0);
        mko::set_surface_transform(t, 2, 0.0, delta, 0.0, 0.0);
        Ray<double> r = axial_ray(0.0, 0.0);
        EXPECT_EQ(trace_to_image(t, r, 1), TraceStatus::Ok);
        return r.p.y;
    };
    // Punktquelle im Abstand z0 vor der Blende; Bildpunkt als Schnitt zweier Strahlen um die
    // Richtung zum Punkt `aim` in der Ebene der ersten Fläche (Meridionalschnitt x = 0).
    const double z0 = 300.0;
    const auto image_point = [&](const HostSurfaceTable& t, double source_y, double aim_y) {
        HostSurfaceTable u = t;
        u.thickness[2] = 400.0; // Bildebene weit hinter der Linse; der Schnitt bestimmt den Punkt
        const double z1 = u.thickness[0];
        std::array<Vec3<double>, 2> p{};
        std::array<Vec3<double>, 2> d{};
        for (int i = 0; i < 2; ++i) {
            const double target = aim_y + (i == 0 ? -1e-3 : 1e-3);
            Ray<double> r;
            r.p = Vec3<double>{0.0, source_y, -z0};
            const Vec3<double> dir{0.0, target - source_y, z0 + z1};
            r.d = (1.0 / mko::norm(dir)) * dir;
            r.opl = 0.0;
            EXPECT_EQ(mko::trace_ray(u.view(), 1, r, nullptr), TraceStatus::Ok);
            p[static_cast<std::size_t>(i)] = r.p;
            d[static_cast<std::size_t>(i)] = r.d;
        }
        // y + s m = y', z + s = z' für beide Strahlen (Steigung m = d_y/d_z)
        const double m0 = d[0].y / d[0].z;
        const double m1 = d[1].y / d[1].z;
        const double z = (p[1].y - p[0].y - m1 * p[1].z + m0 * p[0].z) / (m0 - m1);
        return p[0].y + m0 * (z - p[0].z);
    };
    const double h = 1e-3;
    const double m = image_point(ls.table, h, 0.0) / h;
    double prev_c = 0.0;
    double prev_f = 0.0;
    std::printf("[ Info     ] Bildversatz, m = %.6f:\n", m);
    for (const double delta : {0.2, 0.1, 0.05}) {
        const double yc = shift_collimated(delta);
        HostSurfaceTable t = ls.table;
        mko::set_surface_transform(t, 1, 0.0, delta, 0.0, 0.0);
        mko::set_surface_transform(t, 2, 0.0, delta, 0.0, 0.0);
        const double yf = image_point(t, 0.0, delta);
        const double ec = yc / delta - 1.0;
        const double ef = yf / (delta * (1.0 - m)) - 1.0;
        std::printf("    Δ = %.3f mm: kollimiert y/Δ − 1 = %.3e, endlich y/(Δ(1 − m)) − 1 = %.3e\n",
                    delta, ec, ef);
        EXPECT_LT(std::abs(ec), 1e-3);
        EXPECT_LT(std::abs(ef), 1e-3);
        if (prev_c != 0.0) {
            EXPECT_NEAR(prev_c / ec, 4.0, 0.2);
            EXPECT_NEAR(prev_f / ef, 4.0, 0.2);
        }
        prev_c = ec;
        prev_f = ef;
    }
}

// Koma linear in Δ: Die OPD-Karte (EVAL, Achsfeld, volle Pupille) der um Δ dezentrierten Linse
// enthält den Term y·ρ² mit dem Koeffizienten −4 c₂ (Δ/a) + O(Δ³), wobei c₂ der ρ⁴-Koeffizient
// der koaxialen Linse ist (Verschiebung der Pupille um −Δ/a in W = c₂ ρ⁴; normierte Koordinaten
// der Blende mit Radius a).
TEST(DecenterOracle, ComaIsLinearInDecenter) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost base = mko::make_system_batch(ls, 1);
    const double a = base.pupil_radius(0);
    mko::EvalSettings es;
    es.grid = 64;
    es.reference = mko::EvalReference::Vertex;
    const auto points = [&](const mko::EvalResult& r) {
        std::vector<std::array<double, 3>> pts;
        for (std::int64_t k = 0; k < r.samples(); ++k) {
            const double opd = r.opd[r.ray_index(0, 0, 1, k)];
            if (std::isfinite(opd)) {
                const mko::EvalPoint& p = r.grid.points[static_cast<std::size_t>(k)];
                pts.push_back({p.x, p.y, opd});
            }
        }
        return pts;
    };
    // Koaxial: c₂ aus 1, ρ², ρ⁴, ρ⁶
    const auto c0 =
        least_squares<4>(points(mko::eval_cpu(base, es)), [](double x, double y, double* b) {
            const double r2 = x * x + y * y;
            b[0] = 1.0;
            b[1] = r2;
            b[2] = r2 * r2;
            b[3] = r2 * r2 * r2;
        });
    const double c2 = c0[2];
    double prev = 0.0;
    std::printf("[ Info     ] Koma: c₂ = %.4e mm\n", c2);
    for (const double delta : {0.4, 0.2, 0.1}) {
        SystemBatchHost b = mko::make_non_coaxial(base);
        b.set_surface_transform(0, 1, 0.0, delta, 0.0, 0.0);
        b.set_surface_transform(0, 2, 0.0, delta, 0.0, 0.0);
        // Referenzsphäre um den verschobenen Bildpunkt Δ(1 − m) = Δ, sonst brächte die
        // seitliche Lage der Sphäre einen Term y·ρ² der Ordnung Δ·NA³ mit
        b.field_target(0, 0) = delta;
        const auto c =
            least_squares<10>(points(mko::eval_cpu(b, es)), [](double x, double y, double* v) {
                const double r2 = x * x + y * y;
                v[0] = 1.0;
                v[1] = x;
                v[2] = y;
                v[3] = r2;
                v[4] = y * y;
                v[5] = y * r2;
                v[6] = r2 * r2;
                v[7] = y * r2 * r2;
                v[8] = y * y * r2;
                v[9] = r2 * r2 * r2;
            });
        const double expected = -4.0 * c2 * delta / a;
        const double ratio = c[5] / expected;
        std::printf("    Δ = %.2f mm: Koeffizient y·ρ² %.5e mm, erwartet %.5e mm, Verhältnis "
                    "%.6f, x-Kippung %.1e\n",
                    delta, c[5], expected, ratio, c[1]);
        EXPECT_NEAR(ratio, 1.0, 2e-3);
        EXPECT_LT(std::abs(c[1]), 1e-12); // Symmetrie x → −x bleibt (Δ in y)
        if (prev != 0.0) {
            EXPECT_NEAR(prev / c[5], 2.0, 4e-3); // linear in Δ
        }
        prev = c[5];
    }
}

// Spiegelsymmetrie (EVAL): Dezentrierung in −x und Verkippung −β (um die y-Achse) ergeben die an
// x = 0 gespiegelte OPD-Karte und gespiegelte Spots.
TEST(DecenterOracle, MirroredDecenterGivesMirroredOpdMap) {
    const mko::LoadedSystem ls =
        mko::test::load_residual_system("testsystem_mehrlinser_dezentriert.json");
    const SystemBatchHost a = mko::make_system_batch(ls, 1);
    ASSERT_FALSE(a.coaxial());
    SystemBatchHost b = a;
    for (int s = 0; s < a.surfaces(); ++s) {
        b.decenter_tilt(0, s, 0) = -a.decenter_tilt(0, s, 0);
        b.decenter_tilt(0, s, 3) = -a.decenter_tilt(0, s, 3);
    }
    mko::EvalSettings es;
    es.grid = 33;
    const mko::EvalResult ra = mko::eval_cpu(a, es);
    const mko::EvalResult rb = mko::eval_cpu(b, es);
    const int N = es.grid;
    std::vector<std::int64_t> at(static_cast<std::size_t>(N * N), -1);
    for (std::int64_t k = 0; k < ra.samples(); ++k) {
        const auto ku = static_cast<std::size_t>(k);
        at[static_cast<std::size_t>(ra.grid.row[ku] * N + ra.grid.column[ku])] = k;
    }
    double worst_opd = 0.0;
    double worst_spot = 0.0;
    double asym = 0.0;
    for (int f = 0; f < a.fields(); ++f) {
        for (int l = 0; l < a.wavelengths(); ++l) {
            for (std::int64_t k = 0; k < ra.samples(); ++k) {
                const auto ku = static_cast<std::size_t>(k);
                const std::int64_t km = at[static_cast<std::size_t>(ra.grid.row[ku] * N +
                                                                    (N - 1 - ra.grid.column[ku]))];
                ASSERT_GE(km, 0);
                const std::size_t i = ra.ray_index(0, f, l, k);
                const std::size_t j = rb.ray_index(0, f, l, km);
                worst_opd = std::max(worst_opd, std::abs(ra.opd[i] - rb.opd[j]));
                worst_spot = std::max(worst_spot, std::abs(ra.spot_x[i] + rb.spot_x[j]));
                worst_spot = std::max(worst_spot, std::abs(ra.spot_y[i] - rb.spot_y[j]));
                asym = std::max(asym, std::abs(ra.opd[i] - ra.opd[ra.ray_index(0, f, l, km)]));
            }
        }
    }
    std::printf("[ Info     ] Spiegelung: OPD %.1e mm, Spot %.1e mm (Asymmetrie der Karte selbst "
                "%.1e mm)\n",
                worst_opd, worst_spot, asym);
    EXPECT_GT(asym, 1e-6); // die Karte ist tatsächlich nicht symmetrisch
    EXPECT_LT(worst_opd, mko::test::kDecenterMirror);
    EXPECT_LT(worst_spot, mko::test::kDecenterMirror);
}

// ---- Transformation 0 in allen CPU-Referenzen ------------------------------------------------

// make_non_coaxial (Transformation 0, nicht koaxiale Topologie) ist bitgleich zum koaxialen Batch
// mit voller Pupille: RESIDUALS (weich, hart), MERIT mit Max-Reduktionen, EVAL, FP32-Vorfilterung
// und die adjungierte Jacobi-Matrix; gestörte Varianten mit versagten Strahlen.
TEST(DecenterZero, CpuReferencesAreBitIdentical) {
    for (const char* file : {"testsystem_mehrlinser.json", "einzellinse_geisterplatte.json",
                             "weiches_versagen_start.json"}) {
        SCOPED_TRACE(file);
        const mko::LoadedSystem ls = mko::test::load_residual_system(file);
        const SystemBatchHost coax =
            mko::test::with_full_pupil(mko::test::perturbed_batch(ls, 16, 83010));
        const SystemBatchHost moved = mko::make_non_coaxial(coax);
        ASSERT_FALSE(moved.coaxial());
        for (const mko::ResidualSettings& s :
             {mko::ResidualSettings{6, 8, mko::FailureMode::Soft},
              mko::ResidualSettings{5, 6, mko::FailureMode::Hard}}) {
            const mko::ResidualLayout la(coax.topology(), s);
            const mko::ResidualLayout lb(moved.topology(), s);
            const mko::ResidualResult ra = mko::residuals_cpu(coax, la);
            const mko::ResidualResult rb = mko::residuals_cpu(moved, lb);
            EXPECT_GT(ra.failed_count(), 0u);
            EXPECT_TRUE(same_bytes(ra.residuals, rb.residuals));
            EXPECT_TRUE(same_bytes(ra.opd, rb.opd));
            EXPECT_TRUE(same_bytes(ra.status, rb.status));
            const mko::MeritResult ma = mko::merit_cpu(coax, la, true);
            const mko::MeritResult mb = mko::merit_cpu(moved, lb, true);
            EXPECT_TRUE(same_bytes(ma.merit, mb.merit));
            EXPECT_TRUE(same_bytes(ma.extrema.values, mb.extrema.values));
            const mko::PrefilterStats pa = mko::prefilter_stats_cpu(coax, la, 0.15);
            const mko::PrefilterStats pb = mko::prefilter_stats_cpu(moved, lb, 0.15);
            EXPECT_TRUE(same_bytes(pa.merit, pb.merit));
            EXPECT_TRUE(same_bytes(pa.optical_length, pb.optical_length));
            EXPECT_TRUE(same_bytes(pa.uncertain_ok, pb.uncertain_ok));
        }
        mko::EvalSettings es;
        es.grid = 17;
        const mko::EvalResult ea = mko::eval_cpu(coax, es);
        const mko::EvalResult eb = mko::eval_cpu(moved, es);
        EXPECT_TRUE(same_bytes(ea.opd, eb.opd));
        EXPECT_TRUE(same_bytes(ea.spot_x, eb.spot_x));
        EXPECT_TRUE(same_bytes(ea.field_rays, eb.field_rays));
        EXPECT_TRUE(same_bytes(ea.status, eb.status));
    }
    // Adjungierte Jacobi-Matrix (Variablen der Fälle aus M5-T2)
    for (const auto& c : mko::test::adjoint_cases()) {
        SCOPED_TRACE(c->name);
        const mko::test::FdProblem& p = c->p;
        const SystemBatchHost coax = mko::test::with_full_pupil(p.base);
        const SystemBatchHost moved = mko::make_non_coaxial(coax);
        const mko::VariableSet sa(p.loaded.system, coax.topology());
        const mko::VariableSet sb(p.loaded.system, moved.topology());
        mko::AdjointJacobianOptions o = c->options;
        o.trace_extra = {};
        o.extra = {};
        const mko::AdjointJacobian ja = mko::adjoint_jacobian_cpu(sa, coax, p.x0, o);
        const mko::AdjointJacobian jb = mko::adjoint_jacobian_cpu(sb, moved, p.x0, o);
        EXPECT_TRUE(same_bytes(ja.jacobian, jb.jacobian));
        EXPECT_TRUE(same_bytes(ja.residuals, jb.residuals));
    }
}

// ---- Adjungierte Ableitungen ---------------------------------------------------------------

// Der Rückwärtsdurchlauf bleibt für dezentrierte und verkippte Flächen korrekt (Kettenregel durch
// die Drehung; Ableitungen nach den Transformationen selbst gibt es nicht): Jacobi-Matrix gegen
// die Richardson-Referenz (Krümmungen, Dicken, Arbeitsabstand, Glas, Geisterkoeffizienten) für
// dezentrierte Varianten der Testfälle aus M5-T2.
TEST(DecenterAdjoint, JacobianMatchesRichardson) {
    for (const auto& c : mko::test::decentered_adjoint_cases()) {
        if (!c->richardson) {
            continue;
        }
        const mko::test::FdProblem& p = c->p;
        ASSERT_FALSE(p.base.coaxial()) << c->name;
        const int P = p.set.size();
        const mko::ResidualLayout layout(p.base.topology(), c->options.quadrature);
        const std::int64_t rv = layout.trace_rows();
        const mko::AdjointJacobian j = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0, c->options);
        const mko::test::AdaptiveReference ref =
            mko::test::adaptive_reference(p, layout, c->options.extra, rv);
        ASSERT_FALSE(ref.r2.empty()) << c->name;
        const std::span<const double> jr = mko::test::run_columns(j, 0);
        const auto d = mko::test::adjoint_deviation(mko::test::row_block(jr, P, 0, rv),
                                                    mko::test::row_block(ref.r2, P, 0, rv), p);
        mko::test::print_deviation("Dezentriert, Richardson", c->name, p, d);
        EXPECT_LT(d.scaled.frobenius, mko::test::kAdjointJacobian) << c->name;
        for (int k = 0; k < P; ++k) {
            EXPECT_LT(d.plain.column[static_cast<std::size_t>(k)], mko::test::kAdjointJacobian)
                << c->name << ", " << p.set[k].name;
        }
    }
}

// Neuberechnung einer transformierten Fläche (recompute_surface_step) bitgleich zum
// aufgezeichneten Vorwärtsdurchlauf; die lokalen Größen im gekippten System erfüllen dieselben
// Beziehungen wie im koaxialen Fall (Schnittpunkt auf der Fläche, Normale normiert).
TEST(DecenterAdjoint, RecomputedStepMatchesRecord) {
    const mko::LoadedSystem ls =
        mko::test::load_residual_system("testsystem_mehrlinser_dezentriert.json");
    const mko::SurfaceTableView<double> table = ls.table.view();
    const mko::PupilQuadrature<double> q = mko::make_pupil_quadrature(4, 6, mko::PupilMode::Full);
    int transformed = 0;
    for (int k = 0; k < mko::sample_count(q); ++k) {
        const mko::PupilSample<double> s = mko::sample(q, k);
        Ray<double> r = mko::collimated_ray(4.0, 0.03, s.rho, s.phi);
        mko::TraceRecord<double> rec{};
        ASSERT_EQ(mko::trace_ray(table, 0, r, &rec), TraceStatus::Ok);
        for (int j = 1; j < table.surfaces; ++j) {
            const mko::SurfaceStepState<double> st = mko::recompute_surface_step(
                table, 0, j, mko::ghost_index_before(table, j), rec.p[j - 1], rec.d[j - 1], false);
            ASSERT_EQ(st.status, TraceStatus::Ok);
            EXPECT_EQ(std::memcmp(&st.p_out, &rec.p[j], sizeof(Vec3<double>)), 0);
            EXPECT_EQ(std::memcmp(&st.d_out, &rec.d[j], sizeof(Vec3<double>)), 0);
            if (st.transformed) {
                ++transformed;
                const Vec3<double> qn{st.p_plane.x + st.s * st.d_local.x,
                                      st.p_plane.y + st.s * st.d_local.y, st.s * st.d_local.z};
                const double c = st.c;
                // Punkt auf der Sphäre im gekippten System: c|q|² − 2 q_z = 0
                EXPECT_NEAR(c * mko::dot(qn, qn) - 2.0 * qn.z, 0.0, 1e-13);
                EXPECT_NEAR(mko::norm(st.normal), 1.0, mko::test::kUnitFp64);
            }
        }
    }
    EXPECT_GT(transformed, 0);
}

// ---- Datei ----------------------------------------------------------------------------------

TEST(DecenterFile, ReadWriteRoundTripAndDefaults) {
    const mko::SystemDescription d = mko::read_system_file(
        std::string(MKO_SOURCE_DIR) + "/data/systems/testsystem_mehrlinser_dezentriert.json");
    EXPECT_FALSE(mko::is_coaxial(d));
    EXPECT_EQ(d.surfaces[4].decenter[0], 0.02);
    EXPECT_EQ(d.surfaces[4].decenter[1], -0.03);
    EXPECT_EQ(d.surfaces[6].tilt_deg[1], 0.08);
    EXPECT_EQ(d.surfaces[2].decenter[0], 0.0); // Standard 0
    EXPECT_FALSE(mko::is_transformed(d.surfaces[2]));
    const std::string text = mko::write_system_json(d);
    const mko::SystemDescription back = mko::read_system_json(text);
    EXPECT_EQ(back, d);
    EXPECT_EQ(mko::write_system_json(back), text);
    // Koaxiale Flächen schreiben keine Felder (Dateien von vor M8-T3 bleiben unverändert)
    const mko::SystemDescription coax = mko::read_system_file(
        std::string(MKO_SOURCE_DIR) + "/data/systems/testsystem_mehrlinser.json");
    EXPECT_TRUE(mko::is_coaxial(coax));
    const std::string coax_text = mko::write_system_json(coax);
    EXPECT_EQ(coax_text.find("decenter"), std::string::npos);
    EXPECT_EQ(coax_text.find("tilt_deg"), std::string::npos);
    // Nur Verkippung: nur tilt_deg
    mko::SystemDescription only_tilt = coax;
    only_tilt.surfaces[1].tilt_deg = {0.0, -0.25};
    const std::string t2 = mko::write_system_json(only_tilt);
    EXPECT_EQ(t2.find("decenter"), std::string::npos);
    EXPECT_NE(t2.find("\"tilt_deg\": [0.0, -0.25]"), std::string::npos) << t2;
    EXPECT_EQ(mko::read_system_json(t2), only_tilt);
}

TEST(DecenterFile, ValidationRejectsInvalidTransforms) {
    const std::string base = mko::write_system_json(
        mko::read_system_file(std::string(MKO_SOURCE_DIR) + "/data/systems/einzellinse_bk7.json"));
    const auto expect_path = [&](const std::string& from, const std::string& to,
                                 const std::string& path) {
        std::string text = base;
        const auto pos = text.find(from);
        ASSERT_NE(pos, std::string::npos) << from;
        text.replace(pos, from.size(), to);
        try {
            (void)mko::read_system_json(text);
            ADD_FAILURE() << "kein Fehler für " << to;
        } catch (const mko::FormatError& e) {
            EXPECT_EQ(e.json_path(), path) << e.what();
        }
    };
    const std::string s1 = "\"material\": \"N-BK7\"";
    expect_path(s1, s1 + ", \"decenter\": [0.1]", "surfaces[1].decenter");
    expect_path(s1, s1 + ", \"decenter\": [0.1, 0.2, 0.3]", "surfaces[1].decenter");
    expect_path(s1, s1 + ", \"decenter\": \"0.1\"", "surfaces[1].decenter");
    expect_path(s1, s1 + ", \"tilt_deg\": [0.0, 90.0]", "surfaces[1].tilt_deg[1]");
    expect_path(s1, s1 + ", \"tilt_deg\": [-95.0, 0.0]", "surfaces[1].tilt_deg[0]");
    expect_path("\"material\": \"AIR\"", "\"material\": \"AIR\", \"decenter\": [0.0, 0.1]",
                "surfaces[0]");
    // Modell: nicht endliche Werte
    mko::SystemDescription d = mko::read_system_json(base);
    d.surfaces[2].decenter[1] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(mko::validate_system(d), mko::FormatError);
}

// ---- Tabelle, Batch und Pupille ---------------------------------------------------------------

TEST(DecenterBatch, TableTopologyAndBatchFromFile) {
    const mko::LoadedSystem ls =
        mko::test::load_residual_system("testsystem_mehrlinser_dezentriert.json");
    ASSERT_FALSE(ls.table.coaxial());
    const std::size_t o = 4 * mko::kSurfaceTransformValues;
    EXPECT_EQ(ls.table.decenter_tilt[o], 0.02);
    EXPECT_EQ(ls.table.decenter_tilt[o + 1], -0.03);
    EXPECT_EQ(ls.table.decenter_tilt[6 * 4 + 3], 0.08 * (std::numbers::pi / 180.0));
    EXPECT_EQ(ls.table.tilt_trig[6 * 4 + 3], std::sin(0.08 * (std::numbers::pi / 180.0)));
    const SystemBatchHost b = mko::make_system_batch(ls, 3);
    EXPECT_TRUE(b.topology().non_coaxial);
    for (const mko::PupilMode m : b.topology().pupil_mode) {
        EXPECT_EQ(m, mko::PupilMode::Full);
    }
    for (int v = 0; v < 3; ++v) {
        for (int s = 0; s < b.surfaces(); ++s) {
            for (int k = 0; k < 4; ++k) {
                EXPECT_EQ(b.decenter_tilt(v, s, k),
                          ls.table.decenter_tilt[static_cast<std::size_t>(s * 4 + k)]);
            }
        }
    }
    // Wellenfront (CLI trace): volle Pupille für alle Felder, auch auf der Achse
    for (int f = 0; f < 3; ++f) {
        EXPECT_EQ(mko::wavefront_pupil_mode(ls.system, f), mko::PupilMode::Full);
    }
    // CPU-Pfad über den Batch (Sinus und Kosinus im Tracer) = Host-Tabelle (vorab berechnet)
    const mko::WavefrontResult w = mko::evaluate_wavefront(ls);
    const mko::ResidualLayout layout(b.topology(), {10, 8, mko::FailureMode::Hard});
    const mko::ResidualResult r = mko::residuals_cpu(b, 0, layout);
    double worst = 0.0;
    for (int f = 0; f < 3; ++f) {
        for (int l = 0; l < b.wavelengths(); ++l) {
            const mko::WavefrontMap& m = w.at(f, l);
            for (int k = 0; k < m.samples(); ++k) {
                worst = std::max(worst, std::abs(m.opd[static_cast<std::size_t>(k)] -
                                                 r.opd_at(0, layout.offset(f, l) + k)));
            }
        }
    }
    EXPECT_EQ(worst, 0.0) << "Batch-Pfad und Host-Tabelle sollen bitgleich sein";
    // Speicher: nur nicht koaxiale Batches belegen Transformationsdaten
    const mko::SystemBatchLayout lb = mko::make_batch_layout(b.topology(), 10);
    EXPECT_EQ(lb.decenter_tilt.bytes, 10u * 8u * 4u * 8u);
    {
        mko::BatchTopology t = b.topology();
        t.non_coaxial = false;
        t.pupil_mode.assign(3, mko::PupilMode::Half);
        EXPECT_EQ(mko::make_batch_layout(t, 10).decenter_tilt.bytes, 0u);
        t.non_coaxial = true;
        EXPECT_THROW(mko::validate_topology(t), std::invalid_argument);
    }
}

TEST(DecenterBatch, AccessCopyAndValidation) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost coax = mko::make_system_batch(ls, 2);
    EXPECT_TRUE(coax.coaxial());
    EXPECT_EQ(std::as_const(coax).decenter_tilt(1, 1, 2), 0.0);
    EXPECT_THROW(coax.decenter_tilt(1, 1, 2) = 1.0, std::invalid_argument);
    EXPECT_EQ(coax.view().decenter_tilt, nullptr);
    EXPECT_EQ(coax.surface_table(1).decenter_tilt, nullptr);
    SystemBatchHost b = mko::make_non_coaxial(coax);
    EXPECT_FALSE(b.coaxial());
    EXPECT_EQ(b.decenter_tilt_data().size(), 2u * 3u * 4u);
    b.set_surface_transform(1, 2, 0.1, -0.2, 0.01, -0.02);
    EXPECT_EQ(b.decenter_tilt(1, 2, 1), -0.2);
    EXPECT_THROW(b.decenter_tilt(1, 2, 4), std::out_of_range);
    const mko::SurfaceTableView<double> t1 = b.surface_table(1);
    ASSERT_NE(t1.decenter_tilt, nullptr);
    EXPECT_EQ(t1.decenter_tilt[2 * 4 + 3], -0.02);
    EXPECT_EQ(t1.tilt_trig, nullptr);
    const int added = b.add_variant_copy(1);
    EXPECT_EQ(b.decenter_tilt(added, 2, 0), 0.1);
    SystemBatchHost part;
    part.assign_variants(b, 1, 2);
    EXPECT_EQ(part.decenter_tilt(0, 2, 2), 0.01);
    EXPECT_FALSE(part == b);
    SystemBatchHost same = b;
    EXPECT_TRUE(same == b);
    same.decenter_tilt(0, 1, 0) = 1e-9;
    EXPECT_FALSE(same == b);
    EXPECT_NO_THROW(b.validate_values());
    b.decenter_tilt(0, 1, 2) = 2.0; // |α| >= π/2
    EXPECT_THROW(b.validate_values(), std::invalid_argument);
    b.decenter_tilt(0, 1, 2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(b.validate_values(), std::invalid_argument);
    // Halbe Pupille in EVAL nur koaxial
    mko::EvalSettings es;
    es.half_pupil = true;
    EXPECT_THROW(mko::check_eval_settings(es, mko::make_non_coaxial(coax)), std::invalid_argument);
    EXPECT_NO_THROW(mko::check_eval_settings(es, coax));
}

// ---- CLI ------------------------------------------------------------------------------------

namespace {

struct CliRun {
    int code = -1;
    std::string out;
    std::string err;
};

template <class Fn>
CliRun run_cli(Fn fn, std::initializer_list<std::string_view> args) {
    const std::vector<std::string_view> v(args);
    std::ostringstream out;
    std::ostringstream err;
    CliRun r;
    r.code = fn(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

const std::string kDecenteredFile =
    std::string(MKO_SOURCE_DIR) + "/data/systems/testsystem_mehrlinser_dezentriert.json";

} // namespace

// `mikroptim trace` und `mikroptim eval` mit einem dezentrierten System: volle Pupille für alle
// Felder, Ergebnisse wie die Bibliothek; `eval --half` wird abgelehnt (keine Symmetrie).
TEST(DecenterCli, TraceAndEvalWorkWithDecenteredSystem) {
    const CliRun t = run_cli(mko::cli::run_trace_command, {kDecenteredFile, "--json"});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    const nlohmann::json j = nlohmann::json::parse(t.out);
    const mko::WavefrontResult w = mko::evaluate_wavefront(mko::load_system(kDecenteredFile));
    ASSERT_EQ(j["wavefront"].size(), w.maps.size());
    for (std::size_t i = 0; i < w.maps.size(); ++i) {
        const nlohmann::json& e = j["wavefront"][i];
        EXPECT_EQ(e["pupil_mode"], "full");
        EXPECT_EQ(e["rays"].get<int>(), 80);
        EXPECT_EQ(e["rms_mm"].get<double>(), w.maps[i].rms);
    }
    const CliRun ev =
        run_cli(mko::cli::run_eval_command, {kDecenteredFile, "--json", "--grid", "16"});
    ASSERT_EQ(ev.code, mko::cli::kExitOk) << ev.err;
    const nlohmann::json je = nlohmann::json::parse(ev.out);
    EXPECT_FALSE(je["half_pupil"].get<bool>());
    const CliRun half = run_cli(mko::cli::run_eval_command, {kDecenteredFile, "--half"});
    EXPECT_NE(half.code, mko::cli::kExitOk);
    EXPECT_NE(half.err.find("koaxial"), std::string::npos) << half.err;
}
