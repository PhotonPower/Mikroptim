// Analytische Orakel für die Flächenoperationen (M0).

#include "mikroptim/trace/surface_ops.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using mko::Ray;
using mko::TraceStatus;
using mko::test::kLengthFp64;
using mko::test::kUnitFp64;

namespace {

constexpr double kDeg = std::numbers::pi / 180.0;

Ray<double> collimated_at_height(double h) {
    return Ray<double>{{0.0, h, 0.0}, {0.0, 0.0, 1.0}, 0.0};
}

} // namespace

TEST(SurfaceOps, PlaneSurfaceObeysSnell) {
    const double n1 = 1.0;
    const double n2 = 1.5;
    for (const double deg : {0.0, 10.0, 30.0, 60.0, 80.0}) {
        const double th = deg * kDeg;
        Ray<double> r{{0.0, 1.0, 0.0}, {0.0, std::sin(th), std::cos(th)}, 0.0};
        double s = -1.0;
        ASSERT_EQ(mko::intersect_sphere(r, 0.0, s), TraceStatus::Ok);
        EXPECT_NEAR(s, 0.0, kLengthFp64);
        ASSERT_EQ(mko::refract_sphere(r, 0.0, n1 / n2), TraceStatus::Ok);
        EXPECT_NEAR(r.d.y, n1 / n2 * std::sin(th), kUnitFp64);
        EXPECT_NEAR(mko::norm(r.d), 1.0, kUnitFp64);
    }
}

TEST(SurfaceOps, IntersectionLiesOnSphere) {
    const double R = 10.0;
    for (const double h : {0.0, 1.0, 5.0, 9.0}) {
        auto r = collimated_at_height(h);
        double s = 0.0;
        ASSERT_EQ(mko::intersect_sphere(r, 1.0 / R, s), TraceStatus::Ok);
        const double dz = r.p.z - R; // Mittelpunkt bei (0, 0, R)
        EXPECT_NEAR(std::sqrt(r.p.x * r.p.x + r.p.y * r.p.y + dz * dz), R, kLengthFp64);
        EXPECT_NEAR(s, R - std::sqrt(R * R - h * h), kLengthFp64);
    }
}

// Exakter trigonometrischer Strahl an einer Fläche (Kingslake):
// sin I = h/R, sin I' = (n/n') sin I, U' = I - I', L' = R + R sin I' / sin U'.
TEST(SurfaceOps, SingleSphereMatchesClosedForm) {
    const double R = 10.0;
    const double n1 = 1.0;
    const double n2 = 1.5;
    for (const double h : {0.5, 2.0, 4.0, 6.0, 8.0}) {
        auto r = collimated_at_height(h);
        double s = 0.0;
        ASSERT_EQ(mko::intersect_sphere(r, 1.0 / R, s), TraceStatus::Ok);
        ASSERT_EQ(mko::refract_sphere(r, 1.0 / R, n1 / n2), TraceStatus::Ok);
        const double sinI = h / R;
        const double sinIp = n1 / n2 * sinI;
        const double U = std::asin(sinI) - std::asin(sinIp);
        const double Lp = R + R * sinIp / std::sin(U);
        EXPECT_NEAR(mko::meridional_axis_crossing(r), Lp, kLengthFp64) << "h = " << h;
        EXPECT_NEAR(mko::norm(r.d), 1.0, kUnitFp64);
    }
}

TEST(SurfaceOps, DetectsTotalInternalReflection) {
    const double th = 50.0 * kDeg; // Grenzwinkel Glas n=1.5 gegen Luft: 41,8°
    Ray<double> r{{0.0, 0.0, 0.0}, {0.0, std::sin(th), std::cos(th)}, 0.0};
    double s = 0.0;
    ASSERT_EQ(mko::intersect_sphere(r, 0.0, s), TraceStatus::Ok);
    EXPECT_EQ(mko::refract_sphere(r, 0.0, 1.5 / 1.0), TraceStatus::TIR);
}

TEST(SurfaceOps, DetectsMissedSphere) {
    auto r = collimated_at_height(6.0); // Radius 5 mm: Höhe 6 mm verfehlt die Kugel
    double s = 0.0;
    EXPECT_EQ(mko::intersect_sphere(r, 1.0 / 5.0, s), TraceStatus::Missed);
}

TEST(SurfaceOps, TransferToTangentPlane) {
    const double th = 20.0 * kDeg;
    Ray<double> r{{0.0, 1.0, 0.25}, {0.0, std::sin(th), std::cos(th)}, 0.0};
    const double t = 10.0;
    const double s = mko::transfer_to_tangent_plane(r, t);
    EXPECT_NEAR(r.p.z, 0.0, kLengthFp64);
    EXPECT_NEAR(s, (t - 0.25) / std::cos(th), kLengthFp64);
    EXPECT_NEAR(r.p.y, 1.0 + (t - 0.25) * std::tan(th), kLengthFp64);
}
