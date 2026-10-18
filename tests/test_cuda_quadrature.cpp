// CPU/GPU-Abgleich der Pupillenquadratur (nur mit MIKROPTIM_ENABLE_CUDA).
//
// sample() rechnet nur √u, eine Multiplikation und Divisionen; alle drei sind in IEEE-FP64
// korrekt gerundet (keine fast-math-Optionen) und bieten keine Gelegenheit zur
// FMA-Kontraktion. Daher wird Bitgleichheit erwartet.

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/quadrature_check.hpp"
#include "mikroptim/pupil/quadrature.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(CudaPupilQuadrature, GpuSamplesBitIdenticalToCpu) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const mko::PupilMode modes[] = {mko::PupilMode::Half, mko::PupilMode::Full,
                                    mko::PupilMode::Axis};
    int configs = 0;
    for (int rings = 1; rings <= mko::kMaxPupilRings; ++rings) {
        for (const int spokes : {1, 3, 8, 16, 64}) {
            for (const mko::PupilMode mode : modes) {
                const auto q = mko::make_pupil_quadrature(rings, spokes, mode);
                std::vector<double> rho, phi, weight;
                ASSERT_TRUE(mko::cuda::pupil_samples_gpu(q, rho, phi, weight));
                ASSERT_EQ(static_cast<int>(rho.size()), mko::sample_count(q));
                for (int k = 0; k < mko::sample_count(q); ++k) {
                    const auto s = mko::sample(q, k);
                    EXPECT_EQ(s.rho, rho[k]) << "N=" << rings << " M=" << spokes << " k=" << k;
                    EXPECT_EQ(s.phi, phi[k]) << "N=" << rings << " M=" << spokes << " k=" << k;
                    EXPECT_EQ(s.weight, weight[k])
                        << "N=" << rings << " M=" << spokes << " k=" << k;
                }
                ++configs;
            }
        }
    }
    EXPECT_EQ(configs, 16 * 5 * 3);
}
