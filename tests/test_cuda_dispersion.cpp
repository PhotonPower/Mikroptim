// CPU/GPU-Abgleich der Dispersionsformeln (nur mit MIKROPTIM_ENABLE_CUDA).

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/dispersion_eval.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/dispersion.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

TEST(CudaDispersion, GpuMatchesCpu) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const mko::GlassCatalog cat =
        mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json");
    const double n0 = 1.62;
    const std::array<double, 3> nu{-0.13, 0.031, -0.0042};
    double max_dev = 0.0;

    for (const mko::Glass& g : cat.glasses()) {
        // Die Basisgläser sind 3-Term-Sellmeier (Prüfung des Kerns sellmeier_index).
        ASSERT_EQ(g.B.size(), 3u) << g.name;
        const std::array<double, 3> B{g.B[0], g.B[1], g.B[2]};
        const std::array<double, 3> C{g.C[0], g.C[1], g.C[2]};
        std::vector<double> lambda_um;
        for (int i = 0; i <= 1000; ++i) {
            const double nm = g.range_nm[0] + (g.range_nm[1] - g.range_nm[0]) * i / 1000.0;
            lambda_um.push_back(mko::nm_to_um(nm));
        }
        std::vector<double> ns_gpu;
        std::vector<double> nb_gpu;
        ASSERT_TRUE(mko::cuda::evaluate_dispersion(B, C, n0, nu, lambda_um, ns_gpu, nb_gpu));
        ASSERT_EQ(ns_gpu.size(), lambda_um.size());
        for (std::size_t i = 0; i < lambda_um.size(); ++i) {
            const double l = lambda_um[i];
            const double ns_cpu = mko::sellmeier_index(g.B.data(), g.C.data(), l);
            const double nb_cpu = mko::buchdahl_index(n0, nu[0], nu[1], nu[2], l);
            max_dev =
                std::max({max_dev, std::abs(ns_cpu - ns_gpu[i]), std::abs(nb_cpu - nb_gpu[i])});
            EXPECT_NEAR(ns_cpu, ns_gpu[i], mko::test::kCpuGpuFp64)
                << g.name << ", lambda = " << l << " µm";
            EXPECT_NEAR(nb_cpu, nb_gpu[i], mko::test::kCpuGpuFp64)
                << g.name << ", lambda = " << l << " µm";
        }
    }
    std::printf("[ CPU/GPU ] max |n_CPU - n_GPU| = %.3e\n", max_dev);
}
