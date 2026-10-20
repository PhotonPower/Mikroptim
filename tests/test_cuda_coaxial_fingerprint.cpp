// Fingerabdrücke der Kernel-Modi für koaxiale Systeme (M8-T3, Abnahme (1), nur mit
// MIKROPTIM_ENABLE_CUDA): RESIDUALS und MERIT (FP64 und df64, weich und hart, mit
// Max-Reduktionen), EVAL (volle und halbe Pupille, beide Referenzsphären), FP32-Vorfilterung,
// ADJOINT-Vorwärtsdurchlauf (vollständige Aufzeichnung und Checkpoints) und adjungierte
// Jacobi-Matrix. Nur APIs von vor M8-T3 (coaxial_fingerprint.hpp): Dieselbe Datei auf main
// übersetzt muss auf derselben GPU dieselben Zeilen ausgeben.

#include "adjoint_test_systems.hpp"
#include "coaxial_fingerprint.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cuda/adjoint.hpp"
#include "mikroptim/cuda/adjoint_jacobian.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "residual_test_systems.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace {

using mko::test::Fingerprint;
using mko::test::print_fingerprint;

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

constexpr int kFingerprintVariants = 12;

const char* precision_tag(mko::cuda::Precision p) {
    return p == mko::cuda::Precision::Df64 ? "df64" : "fp64";
}

} // namespace

TEST(CudaCoaxialFingerprint, KernelModesOfAllTestSystems) {
    MKO_REQUIRE_GPU();
    mko::cuda::TraceEngine engine(0);
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
        const mko::SystemBatchHost batch =
            mko::test::perturbed_batch(ls, kFingerprintVariants, ++seed);
        for (const mko::ResidualSettings& s :
             {mko::ResidualSettings{10, 8, mko::FailureMode::Soft},
              mko::ResidualSettings{6, 5, mko::FailureMode::Hard}}) {
            const std::string tag = s.failure == mko::FailureMode::Soft ? "soft" : "hard";
            for (const mko::cuda::Precision p :
                 {mko::cuda::Precision::Fp64, mko::cuda::Precision::Df64}) {
                mko::cuda::TraceSettings ts;
                ts.quadrature = s;
                ts.opd = true;
                ts.extrema = true;
                ts.precision = p;
                {
                    const mko::cuda::TraceOutput o =
                        engine.run(batch, mko::cuda::TraceMode::Residuals, ts);
                    Fingerprint f;
                    f.add(o.residuals.residuals);
                    f.add(o.residuals.opd);
                    f.add(o.residuals.status);
                    f.add(o.residuals.extrema.values);
                    print_fingerprint("gpu", file, "residuals-" + tag + "-" + precision_tag(p), f);
                }
                {
                    const mko::cuda::TraceOutput o =
                        engine.run(batch, mko::cuda::TraceMode::Merit, ts);
                    Fingerprint f;
                    f.add(o.merit.merit);
                    f.add(o.merit.failed);
                    f.add(o.merit.hard_failed);
                    f.add(o.merit.extrema.values);
                    print_fingerprint("gpu", file, "merit-" + tag + "-" + precision_tag(p), f);
                }
            }
            {
                mko::cuda::PrefilterSettings ps;
                ps.quadrature = s;
                const mko::PrefilterResult r = engine.prefilter_variants(batch, ps);
                Fingerprint f;
                f.add(r.stats.merit);
                f.add(r.stats.optical_length);
                f.add(r.stats.failed);
                f.add(r.stats.uncertain_ok);
                f.add(r.stats.uncertain_failed);
                f.add(r.reject);
                print_fingerprint("gpu", file, "prefilter-" + tag, f);
            }
            {
                const mko::ResidualLayout layout(batch.topology(), s);
                const mko::cuda::DeviceSystemBatch device(batch);
                for (const mko::cuda::RecordStorage st :
                     {mko::cuda::RecordStorage::Full, mko::cuda::RecordStorage::Checkpoint}) {
                    for (const mko::cuda::Precision p :
                         {mko::cuda::Precision::Fp64, mko::cuda::Precision::Df64}) {
                        mko::cuda::AdjointForwardOptions ao;
                        ao.storage = st;
                        ao.precision = p;
                        ao.opd = true;
                        ao.states = true;
                        const mko::cuda::AdjointForwardResult r =
                            mko::cuda::run_adjoint_forward(device, layout, ao);
                        Fingerprint f;
                        f.add(r.residuals.residuals);
                        f.add(r.residuals.opd);
                        f.add(r.residuals.status);
                        f.add(r.opl);
                        f.add(r.checksum);
                        f.add(r.count);
                        f.add(r.states);
                        f.add(r.branch);
                        f.add(r.image_s);
                        f.add(r.image_branch);
                        print_fingerprint(
                            "gpu", file,
                            "adjoint-forward-" + tag + "-" +
                                (st == mko::cuda::RecordStorage::Full ? "full" : "checkpoint") +
                                "-" + precision_tag(p),
                            f);
                    }
                }
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
                    const mko::EvalResult r = engine.evaluate(batch, es);
                    f.add(r.opd);
                    f.add(r.spot_x);
                    f.add(r.spot_y);
                    f.add(r.status);
                    f.add(r.field_rays);
                    f.add(r.field_status);
                    f.add(r.reference_curvature);
                    f.add(r.exit_pupil_position);
                } catch (const std::exception& e) {
                    f.add_value(std::uint64_t{0xdead});
                    std::printf("[ Info     ] %s eval: %s\n", file.c_str(), e.what());
                }
                print_fingerprint("gpu", file,
                                  std::string("eval-") + (half ? "half" : "full") +
                                      (ref == mko::EvalReference::Vertex ? "-vertex" : "-pupil"),
                                  f);
            }
        }
    }
    EXPECT_GT(systems, 10);
}

TEST(CudaCoaxialFingerprint, AdjointJacobian) {
    MKO_REQUIRE_GPU();
    mko::cuda::TraceEngine engine(0);
    for (const auto& c : mko::test::adjoint_cases()) {
        for (const mko::cuda::Precision p :
             {mko::cuda::Precision::Fp64, mko::cuda::Precision::Df64}) {
            const mko::AdjointJacobian j = mko::cuda::adjoint_jacobian_gpu(
                engine, c->p.set, c->p.base, c->p.x0, c->options, p);
            Fingerprint f;
            f.add(j.jacobian);
            f.add(j.residuals);
            f.add(j.status);
            print_fingerprint("gpu", c->name, std::string("adjoint-jacobian-") + precision_tag(p),
                              f);
        }
    }
}
