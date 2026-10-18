// Der Header der TraceEngine (M2-T4) ist reines C++: ohne nvcc und ohne CUDA-Bibliothek
// einbindbar (AGENTS.md, Abschnitt 5). Dieser Test läuft auch im CPU-Build; er braucht nur
// die Typen, nicht die Implementierung (die gibt es nur mit MIKROPTIM_ENABLE_CUDA).

#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/cuda/trace_engine.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

using mko::cuda::DeviceFdJacobian;
using mko::cuda::DeviceTranche;
using mko::cuda::EngineOptions;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;

static_assert(!std::is_copy_constructible_v<TraceEngine>);
static_assert(std::is_nothrow_move_constructible_v<TraceEngine>);
static_assert(!std::is_default_constructible_v<TraceEngine>); // Gerät ist Pflicht
// FD-Jacobi auf dem Gerät (M3-T2): ebenfalls reines C++, nur verschiebbar
static_assert(!std::is_copy_constructible_v<DeviceFdJacobian>);
static_assert(std::is_nothrow_move_constructible_v<DeviceFdJacobian>);

TEST(TraceEngineHeader, DefaultsWithoutCuda) {
    const TraceSettings s;
    EXPECT_EQ(s.quadrature.rings, 10);
    EXPECT_EQ(s.quadrature.spokes, 8);
    EXPECT_FALSE(s.opd);
    EXPECT_TRUE(s.status);
    EXPECT_EQ(s.extra_residuals, 0);
    EXPECT_FALSE(s.extrema);
    EXPECT_TRUE(s.extra_merit.empty());
    const EngineOptions o;
    EXPECT_FALSE(o.memory_limit_bytes.has_value());
    EXPECT_EQ(o.reserve.min_bytes, std::size_t{256} << 20);
    EXPECT_EQ(o.reserve.fraction, 0.05);
    const TraceOutput out;
    EXPECT_EQ(out.mode, TraceMode::Residuals);
    EXPECT_EQ(out.info.tranches, 0);
    EXPECT_EQ(out.residuals.variants, 0);
    EXPECT_EQ(out.merit.variants, 0);
    EXPECT_TRUE(out.merit.extrema.empty());
}

TEST(TraceEngineHeader, DeviceTrancheDefaults) {
    const DeviceTranche t;
    EXPECT_EQ(t.first, 0);
    EXPECT_EQ(t.count, 0);
    EXPECT_EQ(t.residuals, nullptr);
    EXPECT_EQ(t.status, nullptr);
    EXPECT_EQ(t.opd, nullptr);
    EXPECT_EQ(t.extrema, nullptr);
}
