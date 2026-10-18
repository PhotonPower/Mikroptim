// CPU/GPU-Abgleich von trace_ray (M1-T5, nur mit MIKROPTIM_ENABLE_CUDA): dieselbe Funktion
// läuft im Kernel über eine Flächentabelle im Shared Memory.

#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/trace_check.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/trace.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

TEST(CudaTrace, GpuMatchesCpu) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    // Einzellinse mit vergrößerter Blende und einer stark gekrümmten Fläche, damit auch
    // verfehlte und totalreflektierte Strahlen vorkommen.
    mko::SystemDescription sys =
        mko::read_system_file(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    sys.surfaces[0].semi_diameter = 14.0;
    sys.surfaces[2].curvature = -0.08;
    const mko::LoadedSystem ls = mko::prepare_system(sys, MKO_SOURCE_DIR);
    const auto view = ls.table.view();
    const auto q = mko::make_pupil_quadrature(16, 16, mko::PupilMode::Full);

    std::size_t compared = 0;
    std::size_t failed = 0;
    double max_err = 0.0;
    for (int f = 0; f < mko::source_field_count(ls.system); ++f) {
        for (int l = 0; l < ls.table.wavelengths; ++l) {
            std::vector<mko::Ray<double>> in;
            for (int k = 0; k < mko::sample_count(q); ++k) {
                in.push_back(mko::source_ray(ls.system, ls.table, f, l, mko::sample(q, k)));
            }
            std::vector<mko::Ray<double>> out;
            std::vector<unsigned> status;
            std::vector<int> surface;
            ASSERT_TRUE(mko::cuda::trace_rays_gpu(view, l, in, out, status, surface));
            for (std::size_t i = 0; i < in.size(); ++i) {
                mko::Ray<double> ray = in[i];
                int fs = 0;
                const mko::TraceStatus st = mko::trace_ray(view, l, ray, nullptr, &fs);
                ASSERT_EQ(static_cast<unsigned>(st), status[i]);
                ASSERT_EQ(fs, surface[i]);
                if (st != mko::TraceStatus::Ok) {
                    ++failed;
                    continue;
                }
                const double err =
                    std::max({std::abs(ray.p.x - out[i].p.x), std::abs(ray.p.y - out[i].p.y),
                              std::abs(ray.p.z - out[i].p.z), std::abs(ray.d.x - out[i].d.x),
                              std::abs(ray.d.y - out[i].d.y), std::abs(ray.d.z - out[i].d.z),
                              std::abs(ray.opl - out[i].opl)});
                EXPECT_LE(err, mko::test::kCpuGpuFp64) << "f=" << f << " l=" << l << " i=" << i;
                max_err = std::max(max_err, err);
                ++compared;
            }
        }
    }
    EXPECT_GT(compared, 0u);
    EXPECT_GT(failed, 0u);
    std::printf("[ Info     ] CPU/GPU: max Abweichung = %.3e (%zu Strahlen verglichen, %zu mit "
                "gleichem Versagensstatus)\n",
                max_err, compared, failed);
}
