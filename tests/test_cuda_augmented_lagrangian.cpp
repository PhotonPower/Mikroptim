// Abnahme des Augmented Lagrangian (M11-T2) auf der GPU (nur mit MIKROPTIM_ENABLE_CUDA):
// dieselbe äußere Schleife wie test_augmented_lagrangian.cpp mit optimize_lm_gpu (die
// Fortschreibung der Verschiebungen rechnet der Host), Grenzen eingehalten, nicht mehr LM-Zyklen
// als die Penalty-Fortsetzung auf der GPU.

#include "augmented_lagrangian_test_systems.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace mko::test {
namespace {

TEST(CudaAugmentedLagrangian, PetzvalWithFixedWeight) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const FdProblem p = petzval_problem();
    const double kappa_max =
        p.loaded.system.constraints->items[petzval_item(p.loaded.system)].max_curvature.value();
    cuda::TraceEngine engine(0);
    const auto gpu = [&](const LmOptions& o, std::span<const double> x0) {
        return cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o);
    };
    const SystemDescription d = petzval_al_system(p);
    const double weight = d.constraints->items[petzval_item(d)].weight_or_default();
    const AlResult al = al_loop(p, d, petzval_al_options, gpu,
                                al_tolerance(weight, kPetzvalPenaltyRelative * kappa_max));
    EXPECT_LT(al.rounds.size(), static_cast<std::size_t>(kAlRounds));
    for (const AlRound& r : al.rounds) {
        const ShiftEntry& e = r.update.entries.at(0);
        std::printf("[GPU] AL-Lauf: F %.10e, %3d Zyklen, %-16s g/κ_max = %+.3e, √w|Δθ| = %.3e\n",
                    r.run.merit, r.run.iterations, to_string(r.run.status), e.violation / kappa_max,
                    r.update.max_change);
        EXPECT_TRUE(lm_converged(r.run.status)) << to_string(r.run.status);
    }
    const double kappa = std::abs(paraxial_at(p, al.x).petzval_curvature);
    EXPECT_LE(kappa, kappa_max * (1.0 + kPetzvalPenaltyRelative));
    EXPECT_GE(kappa, kappa_max * (1.0 - kPetzvalPenaltyRelative));
    const int continuation = stage_iterations(petzval_continuation(p, gpu));
    std::printf("[GPU] AL: %zu Läufe, %d Zyklen, (|κ_P| − κ_max)/κ_max = %+.3e; "
                "Penalty-Fortsetzung %d Zyklen\n",
                al.rounds.size(), al.iterations, (kappa - kappa_max) / kappa_max, continuation);
    EXPECT_LE(al.iterations, continuation);
}

TEST(CudaAugmentedLagrangian, MechanicalWithFixedWeight) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const FdProblem p = mechanical_problem();
    cuda::TraceEngine engine(0);
    const auto gpu = [&](const LmOptions& o, std::span<const double> x0) {
        return cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o);
    };
    const SystemDescription d = mechanical_al_system(p);
    double weight = 0.0;
    for (const ConstraintSpec& c : d.constraints->items) {
        weight = is_mechanical(c.type) ? c.weight_or_default() : weight;
    }
    const AlResult al =
        al_loop(p, d, mechanical_al_options, gpu,
                al_tolerance(weight, std::min(kMechanicalPenaltyLength, kMechanicalPenaltyCos)));
    EXPECT_LT(al.rounds.size(), static_cast<std::size_t>(kAlRounds));
    for (const AlRound& r : al.rounds) {
        EXPECT_TRUE(lm_converged(r.run.status)) << to_string(r.run.status);
        EXPECT_EQ(r.run.failed_rays, 0);
    }
    const MechanicalState s = mechanical_state(p, al.x);
    double worst = -1.0;
    for (std::size_t k = 0; k < s.violation.size(); ++k) {
        const MechanicalConstraint& c = s.constraints.items[k];
        worst = std::max(worst, s.violation[k]);
        EXPECT_LE(s.violation[k],
                  mechanical_tolerance(c.type, kMechanicalPenaltyLength, kMechanicalPenaltyCos))
            << to_string(c.type) << " Fläche " << c.file_surface;
    }
    const int continuation = stage_iterations(mechanical_continuation(p, gpu));
    std::printf("[GPU] AL: %zu Läufe, %d Zyklen, größte Verletzung %+.3e; Penalty-Fortsetzung %d "
                "Zyklen\n",
                al.rounds.size(), al.iterations, worst, continuation);
    EXPECT_LE(al.iterations, continuation);
}

} // namespace
} // namespace mko::test
