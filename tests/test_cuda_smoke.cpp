// CPU/GPU-Abgleich des Mathe-Kerns (nur mit MIKROPTIM_ENABLE_CUDA).

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/smoke_trace.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(CudaSmoke, GpuMatchesCpu) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const double c = 1.0 / 10.0;
    const double mu = 1.0 / 1.5;
    std::vector<double> heights;
    for (int i = 0; i <= 1000; ++i) {
        heights.push_back(-11.0 + 22.0 * i / 1000.0); // enthält verfehlte Strahlen
    }
    std::vector<double> z_gpu;
    std::vector<unsigned> st_gpu;
    ASSERT_TRUE(mko::cuda::smoke_trace(heights, c, mu, z_gpu, st_gpu));

    for (std::size_t i = 0; i < heights.size(); ++i) {
        mko::Ray<double> r{{0.0, heights[i], 0.0}, {0.0, 0.0, 1.0}, 0.0};
        double s = 0.0;
        mko::TraceStatus st = mko::intersect_sphere(r, c, s);
        if (st == mko::TraceStatus::Ok) {
            st = mko::refract_sphere(r, c, mu);
        }
        ASSERT_EQ(static_cast<unsigned>(st), st_gpu[i]) << "h = " << heights[i];
        if (st == mko::TraceStatus::Ok && heights[i] != 0.0) {
            EXPECT_NEAR(mko::meridional_axis_crossing(r), z_gpu[i], mko::test::kCpuGpuFp64)
                << "h = " << heights[i];
        }
    }
}
