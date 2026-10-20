// Fingerabdrücke der CPU-Referenzen für koaxiale Systeme (M8-T3, Abnahme (1)): Wellenfront
// (M1-T7), Trace der Varianten, RESIDUALS weich und hart, MERIT mit Max-Reduktionen, EVAL (volle
// und halbe Pupille, beide Referenzsphären), FP32-Vorfilterung und adjungierte Jacobi-Matrix.
// Nur APIs von vor M8-T3 (coaxial_fingerprint.hpp): Dieselbe Datei auf main übersetzt muss
// dieselben Zeilen ausgeben.

#include "adjoint_test_systems.hpp"
#include "coaxial_fingerprint.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/batch_trace_cpu.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "residual_test_systems.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace {

using mko::test::Fingerprint;
using mko::test::print_fingerprint;

constexpr int kFingerprintVariants = 12;

void add_residuals(Fingerprint& f, const mko::ResidualResult& r) {
    f.add(r.residuals);
    f.add(r.opd);
    f.add(r.status);
    f.add(r.extrema.values);
}

void add_eval(Fingerprint& f, const mko::EvalResult& r) {
    f.add(r.opd);
    f.add(r.spot_x);
    f.add(r.spot_y);
    f.add(r.status);
    f.add(r.field_rays);
    f.add(r.field_status);
    f.add(r.reference_curvature);
    f.add(r.exit_pupil_position);
}

} // namespace

TEST(CoaxialFingerprint, CpuReferencesOfAllTestSystems) {
    std::uint64_t seed = 8300;
    int systems = 0;
    for (const std::string& file : mko::test::fingerprint_system_files()) {
        SCOPED_TRACE(file);
        mko::LoadedSystem ls;
        try {
            ls = mko::test::load_residual_system(file);
        } catch (const std::exception& e) {
            std::printf("[ Info     ] %s übersprungen: %s\n", file.c_str(), e.what());
            continue;
        }
        ++systems;
        {
            Fingerprint f;
            try {
                const mko::WavefrontResult w = mko::evaluate_wavefront(ls);
                for (const mko::WavefrontMap& m : w.maps) {
                    f.add(m.opl);
                    f.add(m.opd);
                    f.add(m.status);
                    f.add(m.failed_surface);
                }
            } catch (const std::exception& e) {
                f.add_value(std::uint64_t{0xdead}); // z. B. afokales System ohne Sollbrennweite
                std::printf("[ Info     ] %s wavefront: %s\n", file.c_str(), e.what());
            }
            print_fingerprint("cpu", file, "wavefront", f);
        }
        const mko::SystemBatchHost batch =
            mko::test::perturbed_batch(ls, kFingerprintVariants, ++seed);
        {
            Fingerprint f;
            const mko::PupilQuadrature<double> q =
                mko::make_pupil_quadrature(6, 6, mko::PupilMode::Full);
            for (int v = 0; v < batch.variants(); ++v) {
                const mko::CpuTraceResult t = mko::trace_variant_cpu(batch, v, q);
                f.add(t.rays);
                f.add(t.status);
                f.add(t.failed_surface);
            }
            print_fingerprint("cpu", file, "trace", f);
        }
        for (const mko::ResidualSettings& s :
             {mko::ResidualSettings{10, 8, mko::FailureMode::Soft},
              mko::ResidualSettings{6, 5, mko::FailureMode::Hard}}) {
            const mko::ResidualLayout layout(batch.topology(), s);
            const std::string tag = s.failure == mko::FailureMode::Soft ? "soft" : "hard";
            {
                Fingerprint f;
                add_residuals(f, mko::residuals_cpu(batch, layout));
                print_fingerprint("cpu", file, "residuals-" + tag, f);
            }
            {
                Fingerprint f;
                const mko::MeritResult m = mko::merit_cpu(batch, layout, true);
                f.add(m.merit);
                f.add(m.failed);
                f.add(m.hard_failed);
                f.add(m.extrema.values);
                print_fingerprint("cpu", file, "merit-" + tag, f);
            }
            {
                Fingerprint f;
                const mko::PrefilterStats p = mko::prefilter_stats_cpu(batch, layout, 0.15);
                f.add(p.merit);
                f.add(p.optical_length);
                f.add(p.failed);
                f.add(p.uncertain_ok);
                f.add(p.uncertain_failed);
                print_fingerprint("cpu", file, "prefilter-" + tag, f);
            }
        }
        for (const bool half : {false, true}) {
            for (const mko::EvalReference ref :
                 {mko::EvalReference::ExitPupil, mko::EvalReference::Vertex}) {
                mko::EvalSettings es;
                es.grid = 15;
                es.half_pupil = half;
                es.reference = ref;
                Fingerprint f;
                try {
                    add_eval(f, mko::eval_cpu(batch, es));
                } catch (const std::exception& e) {
                    f.add_value(std::uint64_t{0xdead});
                    std::printf("[ Info     ] %s eval: %s\n", file.c_str(), e.what());
                }
                print_fingerprint("cpu", file,
                                  std::string("eval-") + (half ? "half" : "full") +
                                      (ref == mko::EvalReference::Vertex ? "-vertex" : "-pupil"),
                                  f);
            }
        }
    }
    EXPECT_GT(systems, 10);
}

TEST(CoaxialFingerprint, CpuAdjointJacobian) {
    for (const auto& c : mko::test::adjoint_cases()) {
        Fingerprint f;
        const mko::AdjointJacobian j =
            mko::adjoint_jacobian_cpu(c->p.set, c->p.base, c->p.x0, c->options);
        f.add(j.jacobian);
        f.add(j.residuals);
        f.add(j.status);
        print_fingerprint("cpu", c->name, "adjoint-jacobian", f);
    }
}
