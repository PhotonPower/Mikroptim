// Dispersionsformeln (Sellmeier, Buchdahl) gegen analytisch berechnete Werte.

#include "mikroptim/glass/dispersion.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>

using mko::test::kUnitFp64;

TEST(Dispersion, NmToUm) {
    EXPECT_DOUBLE_EQ(mko::nm_to_um(587.56), 0.58756);
    EXPECT_EQ(mko::nm_to_um(1000.0), 1.0);
}

// Ein Term, Pol bei C = 0: n^2 = 1 + B, unabhängig von lambda.
TEST(Dispersion, SellmeierSingleTermWithoutPole) {
    const double B[3] = {1.25, 0.0, 0.0};
    const double C[3] = {0.0, 0.0, 0.0};
    for (const double l : {0.4, 0.55, 1.0, 2.0}) {
        EXPECT_NEAR(mko::sellmeier_index(B, C, l), 1.5, kUnitFp64) << "lambda = " << l;
    }
}

// Drei Terme, von Hand: lambda = 0,5 µm, l2 = 0,25.
// n^2 = 1 + 0,5*0,25/0,24 + 0,2*0,25/0,2 + 0,8*0,25/(0,25-100)
TEST(Dispersion, SellmeierThreeTermsByHand) {
    const double B[3] = {0.5, 0.2, 0.8};
    const double C[3] = {0.01, 0.05, 100.0};
    const double n2 = 1.0 + 0.5 * 0.25 / 0.24 + 0.2 * 0.25 / 0.2 + 0.8 * 0.25 / (0.25 - 100.0);
    EXPECT_NEAR(mko::sellmeier_index(B, C, 0.5), std::sqrt(n2), kUnitFp64);
}

// Normale Dispersion: n fällt im sichtbaren Bereich mit lambda (N-BK7-Koeffizienten).
TEST(Dispersion, SellmeierNormalDispersion) {
    const double B[3] = {1.03961212, 0.231792344, 1.01046945};
    const double C[3] = {0.00600069867, 0.0200179144, 103.560653};
    double prev = mko::sellmeier_index(B, C, 0.40);
    for (double l = 0.41; l <= 0.70; l += 0.01) {
        const double n = mko::sellmeier_index(B, C, l);
        EXPECT_LT(n, prev) << "lambda = " << l;
        prev = n;
    }
}

TEST(Dispersion, BuchdahlOmegaVanishesAtLambda0) {
    EXPECT_EQ(mko::buchdahl_omega(mko::kBuchdahlLambda0Um), 0.0);
    EXPECT_EQ(mko::buchdahl_index(1.6, -0.1, 0.03, -0.004, mko::kBuchdahlLambda0Um), 1.6);
}

// lambda = lambda0 + 0,1 µm: omega = 0,1 / 1,25 = 0,08.
// n = 1,5 - 0,1*0,08 + 0,02*0,0064 - 0,003*0,000512 = 1,492126464
TEST(Dispersion, BuchdahlAnalyticValue) {
    const double l = mko::kBuchdahlLambda0Um + 0.1;
    EXPECT_NEAR(mko::buchdahl_omega(l), 0.08, kUnitFp64);
    EXPECT_NEAR(mko::buchdahl_index(1.5, -0.1, 0.02, -0.003, l), 1.492126464, kUnitFp64);
}

// lambda = lambda0 - 0,1 µm: omega = -0,1 / 0,75 = -2/15.
TEST(Dispersion, BuchdahlBelowLambda0) {
    const double l = mko::kBuchdahlLambda0Um - 0.1;
    const double w = -2.0 / 15.0;
    EXPECT_NEAR(mko::buchdahl_omega(l), w, kUnitFp64);
    EXPECT_NEAR(mko::buchdahl_index(1.5, -0.1, 0.02, -0.003, l),
                1.5 - 0.1 * w + 0.02 * w * w - 0.003 * w * w * w, kUnitFp64);
    EXPECT_GT(mko::buchdahl_denominator(0.1461), 0.0);
    EXPECT_LT(mko::buchdahl_denominator(0.1460), 0.0);
}
