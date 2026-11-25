// Bildpunkt der Referenzsphäre am Hauptstrahl und Verzeichnung als Nebenbedingung (M6-T14,
// ADR-0019; docs/numerik.md, Abschnitt 7, docs/optimierung.md, Abschnitt 2.8): Höhe des realen
// Hauptstrahls gegen analytische Orakel (Planplatten, zur Blende konzentrisches System:
// verzeichnungsfrei in f·tan θ), Merit mit Mittelpunkt am Hauptstrahl gleich der Merit mit dem
// dorthin verschobenen idealen Punkt (bitgleich), apply gleich Neuaufbau, FD-Jacobi enthält
// ∂ȳ/∂x (Kettenregel gegen Differenzen), Nebenbedingung distortion, Bewertung (EVAL), Dateiformat,
// Kompatibilität ohne Feld und Ablehnung durch die adjungierte Jacobi.

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

using mko::ConstraintType;
using mko::FormatError;
using mko::LoadedSystem;
using mko::ReferencePointKind;
using mko::ReferenceSphereKind;
using mko::ResidualLayout;
using mko::SystemBatchHost;
using mko::SystemDescription;
using mko::test::kChiefRayAnalytic;
using mko::test::kChiefRayChainRule;

namespace {

const std::filesystem::path kSystems = std::filesystem::path(MKO_SOURCE_DIR) / "data" / "systems";
constexpr double kDeg = 3.14159265358979323846 / 180.0;

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

std::string material(double n) {
    return n == 1.0 ? std::string("\"AIR\"") : "{ \"model\": \"constant\", \"n\": " + num(n) + " }";
}

// Zur Blende konzentrisches System (wie test_sine_condition.cpp): Blende in Luft, Kugelflächen
// mit Krümmungsmittelpunkt in der Blendenmitte. Der Hauptstrahl trifft jede Fläche senkrecht und
// läuft unabgelenkt durch die Blendenmitte, die Knotenpunkte (in Luft = Hauptpunkte) liegen dort:
// ȳ = L tan θ mit L = Abstand Blende–Bildebene, h = f tan θ, also D = L/f − 1 exakt.
LoadedSystem concentric(const std::vector<double>& gaps, const std::vector<double>& n,
                        double image_distance, const std::vector<double>& fields_deg,
                        const std::string& extra) {
    std::string s = R"({
  "format": "mikroptim.system/1",
  "name": "Konzentrisch",
  "units": "mm",
  "glass_catalogs": [],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [)";
    std::string weights;
    for (std::size_t i = 0; i < fields_deg.size(); ++i) {
        s += (i ? ", " : "") + num(fields_deg[i]);
        weights += (i ? ", " : "") + std::string("1.0");
    }
    s += "], \"field_weights\": [" + weights + "] },\n" + extra + "  \"surfaces\": [\n";
    s += "    { \"type\": \"stop\", \"semi_diameter\": 2.0, \"thickness\": " + num(gaps[0]) +
         ", \"material\": \"AIR\" }";
    double z = 0.0;
    for (std::size_t k = 1; k < n.size(); ++k) {
        z += gaps[k - 1];
        const double t = k < gaps.size() ? gaps[k] : image_distance;
        s += ",\n    { \"type\": \"sphere\", \"curvature\": " + num(-1.0 / z) +
             ", \"thickness\": " + num(t) + ", \"material\": " + material(n[k]) +
             ", \"semi_diameter\": " + num(0.99 * z) + " }";
    }
    s += "\n  ]\n}\n";
    return mko::prepare_system(mko::read_system_json(s), kSystems);
}

const std::string kChief = "  \"reference_sphere\": \"image_infinite\",\n"
                           "  \"reference_point\": \"chief_ray\",\n";

// Einzellinse aus data/systems mit Referenzsphäre, Bildpunkt und Feldern.
LoadedSystem lens(const std::string& file, ReferenceSphereKind sphere,
                  std::optional<ReferencePointKind> point, std::vector<double> fields) {
    LoadedSystem l = mko::load_system(kSystems / file);
    l.system.reference_sphere = sphere;
    l.system.reference_point = point;
    l.system.source.field_angles_deg = fields;
    l.system.source.field_weights.assign(fields.size(), 1.0);
    return mko::prepare_system(l.system, kSystems);
}

} // namespace

// Planplatten: der Hauptstrahl wird nur parallel versetzt, ȳ = Σ t_i tan θ_i mit
// n_i sin θ_i = sin θ, bei λ_ref (paraxial_index) und bei jeder Wellenlänge des Batches (index).
TEST(ReferencePoint, ChiefRayHeightThroughPlatesIsAnalytic) {
    LoadedSystem l = mko::load_system(kSystems / "einzellinse_bk7.json");
    l.system.surfaces[1].curvature = 0.0;
    l.system.surfaces[2].curvature = 0.0;
    l.system.source.field_angles_deg = {0.0, 5.0, 20.0};
    l.system.source.field_weights = {1.0, 1.0, 1.0};
    l = mko::prepare_system(l.system, kSystems);
    const SystemBatchHost b = mko::make_system_batch(l, 1);
    for (int f = 0; f < b.fields(); ++f) {
        const double theta = b.field_angle(0, f);
        for (int l_ = -1; l_ < b.wavelengths(); ++l_) {
            const double n = l_ < 0 ? b.paraxial_index(0, 1) : b.index(0, 1, l_);
            const double tg = std::tan(std::asin(std::sin(theta) / n));
            const double expected =
                b.thickness(0, 0) * std::tan(theta) + b.thickness(0, 1) * tg +
                b.thickness(0, 2) * std::tan(theta);
            const double y = mko::batch_chief_ray_height(b, 0, f, l_);
            EXPECT_NEAR(y, expected, kChiefRayAnalytic * (1.0 + std::abs(expected)))
                << "Feld " << f << ", Wellenlänge " << l_;
        }
    }
    // Schwerpunkt: Hinter Planplatten bleibt das Bündel parallel und symmetrisch zum
    // Hauptstrahl, der Schwerpunkt ist der Hauptstrahl (halbe Pupille, Gewichte der Quadratur)
    for (int f = 1; f < b.fields(); ++f) {
        const double y = mko::batch_chief_ray_height(b, 0, f);
        EXPECT_NEAR(mko::batch_centroid_height(b, 0, f), y, kChiefRayAnalytic * (1.0 + std::abs(y)));
    }
    EXPECT_EQ(mko::batch_centroid_height(b, 0, 0), 0.0);
    // Dispersion: die Höhen der Wellenlängen unterscheiden sich (Querversatz der Platte)
    EXPECT_NE(mko::batch_chief_ray_height(b, 0, 2, 0), mko::batch_chief_ray_height(b, 0, 2, 2));
    EXPECT_THROW((void)mko::batch_chief_ray_height(b, 0, 3), std::out_of_range);
    EXPECT_THROW((void)mko::batch_chief_ray_height(b, 0, 0, 3), std::out_of_range);
}

// Analytisches Orakel der Verzeichnung: zur Blende konzentrisches System, D = L/f − 1; Bildebene
// im Brennpunkt (L = f) verzeichnungsfrei bis auf Rundung, um δ verschoben D = δ/f. Residuum der
// Nebenbedingung √w max(0, |D| − D_max), auf der Achse 0.
TEST(ReferencePoint, DistortionOfConcentricSystemIsAnalytic) {
    const std::vector<double> gaps = {10.0, 3.0};
    const std::vector<double> n = {1.0, 1.6, 1.0};
    const std::vector<double> fields = {0.0, 4.0, 12.0};
    // Brennweite (Hauptpunkte in der Blendenmitte): Bildebene in den Brennpunkt
    const LoadedSystem probe = concentric(gaps, n, 20.0, fields, kChief);
    const double f = mko::batch_paraxial(mko::make_system_batch(probe, 1), 0).focal_length;
    const double w0 = f - (gaps[0] + gaps[1]);
    ASSERT_TRUE(std::isfinite(w0)); // zerstreuende Schale: f < 0, virtuelle Bildebene (w < 0)
    for (const double delta : {0.0, 0.05, -0.2}) {
        const LoadedSystem l = concentric(gaps, n, w0 + delta, fields, kChief);
        const SystemBatchHost b = mko::make_system_batch(l, 1);
        const double L = gaps[0] + gaps[1] + w0 + delta;
        for (int fi = 1; fi < b.fields(); ++fi) {
            const double theta = fields[static_cast<std::size_t>(fi)] * kDeg;
            const double y = mko::batch_chief_ray_height(b, 0, fi);
            EXPECT_NEAR(y, L * std::tan(theta), kChiefRayAnalytic * std::abs(L));
            // Mittelpunkt der Referenzsphäre am Hauptstrahl
            EXPECT_TRUE(same_bits(b.reference_height(0, fi), y));
            const double h = b.field_target(0, fi);
            EXPECT_NEAR(h, f * std::tan(theta), kChiefRayAnalytic * std::abs(f));
            mko::ParaxialConstraint c;
            c.type = ConstraintType::Distortion;
            c.mode = mko::ConstraintMode::Max;
            c.weight = 4.0;
            c.target = 0.0;
            c.field = fi;
            const mko::ParaxialData p = mko::batch_paraxial(b, 0);
            const double q = mko::paraxial_deviation(c, b, 0, p);
            EXPECT_NEAR(q, std::abs(delta / f), kChiefRayAnalytic) << delta;
            EXPECT_NEAR(mko::paraxial_residual(c, b, 0, p), 2.0 * std::abs(delta / f),
                        2.0 * kChiefRayAnalytic);
            c.target = 0.5 * std::abs(delta / f) + 1e-9; // Grenze: nur der Überschuss
            EXPECT_NEAR(mko::paraxial_residual(c, b, 0, p),
                        2.0 * std::max(0.0, std::abs(delta / f) - c.target),
                        2.0 * kChiefRayAnalytic);
        }
        mko::ParaxialConstraint axis;
        axis.type = ConstraintType::Distortion;
        axis.mode = mko::ConstraintMode::Max;
        axis.field = 0;
        EXPECT_EQ(mko::paraxial_residual(axis, b, 0, mko::batch_paraxial(b, 0)), 0.0);
        axis.field = 7; // Feld, das der Batch nicht hat: inaktiv
        EXPECT_EQ(mko::paraxial_residual(axis, b, 0, mko::batch_paraxial(b, 0)), 0.0);
    }
}

// Die Merit mit Mittelpunkt am Hauptstrahl ist die Merit mit dem idealen Punkt an der Höhe ȳ_f
// (bitgleich, image_infinite und image_exit_pupil); bei λ_ref fehlt die Kippung der Verzeichnung
// (RMS kleiner), bei den übrigen Wellenlängen bleibt die Kippung relativ zu λ_ref (Querfarbe).
TEST(ReferencePoint, MeritEqualsIdealMeritAtChiefRayHeight) {
    for (const auto& [sphere, point] :
         {std::pair{ReferenceSphereKind::ImageInfinite, ReferencePointKind::ChiefRay},
          std::pair{ReferenceSphereKind::ImageExitPupil, ReferencePointKind::ChiefRay},
          std::pair{ReferenceSphereKind::ImageInfinite, ReferencePointKind::Centroid},
          std::pair{ReferenceSphereKind::ImageExitPupil, ReferencePointKind::Centroid}}) {
        const LoadedSystem chief = lens("einzellinse_bk7.json", sphere, point, {0.0, 3.0, 6.0});
        const LoadedSystem ideal = lens("einzellinse_bk7.json", sphere, std::nullopt,
                                        {0.0, 3.0, 6.0});
        const SystemBatchHost bc = mko::make_system_batch(chief, 1);
        SystemBatchHost bi = mko::make_system_batch(ideal, 1);
        EXPECT_EQ(bc.topology().reference_point, point);
        for (int f = 0; f < bc.fields(); ++f) {
            EXPECT_TRUE(same_bits(bc.reference_height(0, f),
                                  point == ReferencePointKind::Centroid
                                      ? mko::batch_centroid_height(bc, 0, f)
                                      : mko::batch_chief_ray_height(bc, 0, f)));
        }
        EXPECT_EQ(bi.topology().reference_point, ReferencePointKind::Ideal);
        EXPECT_TRUE(bi.reference_height_data().empty());
        for (int f = 0; f < bi.fields(); ++f) {
            bi.field_target(0, f) = bc.reference_height(0, f);
        }
        mko::update_reference_curvature(bi);
        const ResidualLayout layout(bc.topology());
        const mko::ResidualResult rc = mko::residuals_cpu(bc, 0, layout);
        const mko::ResidualResult ri = mko::residuals_cpu(bi, 0, layout);
        ASSERT_EQ(rc.residuals.size(), ri.residuals.size());
        for (std::size_t k = 0; k < rc.residuals.size(); ++k) {
            ASSERT_TRUE(same_bits(rc.residuals[k], ri.residuals[k])) << k;
        }
        EXPECT_TRUE(same_bits(mko::merit_cpu(bc, 0, layout).merit.at(0),
                              mko::merit_cpu(bi, 0, layout).merit.at(0)));
    }
    // RMS pro Tupel (trace-Pfad evaluate_wavefront) gegen den idealen Punkt
    const mko::WavefrontResult wc = mko::evaluate_wavefront(
        lens("einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite,
             ReferencePointKind::ChiefRay, {0.0, 3.0, 6.0}));
    const mko::WavefrontResult wi = mko::evaluate_wavefront(lens(
        "einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite, std::nullopt, {0.0, 3.0, 6.0}));
    const int lref = 1; // 587,56 nm
    // Achse: gleich (Hauptstrahl auf der Achse)
    for (int k = 0; k < wc.wavelengths; ++k) {
        EXPECT_NEAR(wc.at(0, k).rms_waves, wi.at(0, k).rms_waves, 1e-12);
    }
    // Feld bei λ_ref: anderer Mittelpunkt (Kippung der Verzeichnung fehlt; der Hauptstrahl ist
    // nicht der beste Punkt, bei Koma kann der RMS auch steigen)
    EXPECT_NE(wc.at(2, lref).rms_waves, wi.at(2, lref).rms_waves);
    // Querfarbe bleibt: bei λ ≠ λ_ref ist die Wellenfront nicht die um den eigenen Hauptstrahl
    const SystemBatchHost bc = mko::make_system_batch(
        lens("einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite,
             ReferencePointKind::ChiefRay, {0.0, 3.0, 6.0}),
        1);
    const double lateral =
        mko::batch_chief_ray_height(bc, 0, 2, 0) - mko::batch_chief_ray_height(bc, 0, 2, 2);
    EXPECT_GT(std::abs(lateral), 1e-6); // Einzellinse: Querfarbe (Blende vor der Linse)
}

// VariableSet::apply rechnet ȳ_f neu: der Batch nach apply ist bitgleich zum Neuaufbau aus der
// mit apply_to_description geänderten Datei (wie für alle übrigen Felder).
TEST(ReferencePoint, ApplyUpdatesChiefRayHeightLikeRebuild) {
    const LoadedSystem l = lens("einzellinse_bk7_biegung.json", ReferenceSphereKind::ImageInfinite,
                                ReferencePointKind::ChiefRay, {0.0, 2.0, 4.0});
    SystemBatchHost b = mko::make_system_batch(l, 1);
    const mko::VariableSet set(l.system, b.topology());
    std::vector<double> x = set.extract(b, 0);
    for (double& xi : x) {
        xi *= 1.02;
    }
    const std::vector<double> before(b.reference_height_data().begin(),
                                     b.reference_height_data().end());
    set.apply(b, 0, x);
    SystemDescription d = l.system;
    set.apply_to_description(d, x);
    const SystemBatchHost rebuilt = mko::make_system_batch(mko::prepare_system(d, kSystems), 1);
    int changed = 0;
    for (int f = 0; f < b.fields(); ++f) {
        EXPECT_TRUE(same_bits(b.reference_height(0, f), mko::batch_chief_ray_height(b, 0, f)));
        EXPECT_TRUE(same_bits(b.reference_height(0, f), rebuilt.reference_height(0, f)));
        changed += same_bits(before[static_cast<std::size_t>(f)], b.reference_height(0, f)) ? 0 : 1;
    }
    EXPECT_EQ(changed, 2); // beide Feldpunkte, die Achse bleibt 0
}

// Ableitungen gegen Differenzen: Die FD-Jacobi mit Bildpunkt am Hauptstrahl ist die Kettenregel
// J = ∂r/∂x|_ȳ + (∂r/∂ȳ_f)(∂ȳ_f/∂x), alle drei Teile aus eigenen zentralen Differenzen
// (Batch mit idealem Punkt an der Höhe ȳ_f, Verschiebung des Punkts, Hauptstrahl allein).
TEST(ReferencePoint, FdJacobianContainsChiefRayDerivative) {
    const LoadedSystem l = lens("einzellinse_bk7_biegung.json", ReferenceSphereKind::ImageInfinite,
                                ReferencePointKind::ChiefRay, {0.0, 4.0});
    const SystemBatchHost b = mko::make_system_batch(l, 1);
    const mko::VariableSet set(l.system, b.topology());
    const std::vector<double> x0 = set.extract(b, 0);
    mko::FdJacobianOptions o;
    o.quadrature.rings = 6;
    o.quadrature.spokes = 6;
    const mko::FdJacobian J = mko::fd_jacobian_cpu(set, b, x0, o);
    const ResidualLayout layout(b.topology(), o.quadrature);
    const std::int64_t R = layout.trace_rows();

    // Batch mit idealem Punkt an der Höhe ȳ_f(x₀), Höhe fest
    LoadedSystem li = l;
    li.system.reference_point.reset();
    li = mko::prepare_system(li.system, kSystems);
    SystemBatchHost bi = mko::make_system_batch(li, 1);
    const mko::VariableSet set_i(li.system, bi.topology());
    const int F = b.fields();
    for (int f = 0; f < F; ++f) {
        bi.field_target(0, f) = b.reference_height(0, f);
    }
    const auto residuals = [&](SystemBatchHost& batch) {
        return mko::residuals_cpu(batch, 0, layout).residuals;
    };
    const double hy = 1e-6; // Verschiebung des Punkts [mm]
    std::vector<std::vector<double>> dr_dh(static_cast<std::size_t>(F));
    for (int f = 1; f < F; ++f) {
        SystemBatchHost p = bi;
        SystemBatchHost m = bi;
        p.field_target(0, f) += hy;
        m.field_target(0, f) -= hy;
        const std::vector<double> rp = residuals(p);
        const std::vector<double> rm = residuals(m);
        dr_dh[static_cast<std::size_t>(f)].resize(static_cast<std::size_t>(R));
        for (std::int64_t k = 0; k < R; ++k) {
            dr_dh[static_cast<std::size_t>(f)][static_cast<std::size_t>(k)] =
                (rp[static_cast<std::size_t>(k)] - rm[static_cast<std::size_t>(k)]) / (2.0 * hy);
        }
    }
    double worst = 0.0;
    double scale = 0.0;
    for (int p = 0; p < set.size(); ++p) {
        const double h = J.patterns[0].nominal[static_cast<std::size_t>(p)];
        std::vector<double> xp = x0;
        std::vector<double> xm = x0;
        xp[static_cast<std::size_t>(p)] += h;
        xm[static_cast<std::size_t>(p)] -= h;
        // ∂r/∂x bei festem Punkt
        SystemBatchHost bp = bi;
        SystemBatchHost bm = bi;
        set_i.apply(bp, 0, xp);
        set_i.apply(bm, 0, xm);
        const std::vector<double> rp = residuals(bp);
        const std::vector<double> rm = residuals(bm);
        // ∂ȳ_f/∂x
        SystemBatchHost cp = b;
        SystemBatchHost cm = b;
        set.apply(cp, 0, xp);
        set.apply(cm, 0, xm);
        for (std::int64_t k = 0; k < R; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            double expected = (rp[ku] - rm[ku]) / (2.0 * h);
            for (int f = 1; f < F; ++f) {
                const double dy =
                    (cp.reference_height(0, f) - cm.reference_height(0, f)) / (2.0 * h);
                expected += dr_dh[static_cast<std::size_t>(f)][ku] * dy;
            }
            worst = std::max(worst, std::abs(J.at(0, k, p) - expected));
            scale = std::max(scale, std::abs(expected));
        }
    }
    std::printf("Kettenregel: max |J − (∂r/∂x + ∂r/∂ȳ ∂ȳ/∂x)| = %.3e, max |J| = %.3e\n", worst,
                scale);
    ASSERT_GT(scale, 0.0);
    EXPECT_LE(worst, kChiefRayChainRule * scale);
}

// Bewertung (EVAL): Mit Bildpunkt am Hauptstrahl sinkt der RMS bei λ_ref am Feldrand, die
// Kennzahlen Verzeichnung, Hauptstrahl und Schwerpunkt bleiben relativ zum idealen Punkt (gleich).
TEST(ReferencePoint, EvalKeepsDistortionRelativeToIdealPoint) {
    mko::EvalSettings s;
    s.grid = 32;
    s.reference = mko::EvalReference::Infinite;
    const mko::EvalResult ec = mko::eval_cpu(
        mko::make_system_batch(lens("einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite,
                                    ReferencePointKind::ChiefRay, {0.0, 6.0}),
                               1),
        s);
    const mko::EvalResult ei = mko::eval_cpu(
        mko::make_system_batch(lens("einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite,
                                    std::nullopt, {0.0, 6.0}),
                               1),
        s);
    const std::vector<mko::EvalTupleStats> sc = mko::eval_statistics(ec, 0);
    const std::vector<mko::EvalTupleStats> si = mko::eval_statistics(ei, 0);
    ASSERT_EQ(sc.size(), si.size());
    for (std::size_t i = 3; i < sc.size(); ++i) { // Feld 1 (auf der Achse ist D nicht definiert)
        EXPECT_NEAR(sc[i].distortion_chief, si[i].distortion_chief, 1e-9) << i;
        EXPECT_NEAR(sc[i].chief_y, si[i].chief_y, 1e-12) << i;
        EXPECT_NEAR(sc[i].centroid_y, si[i].centroid_y, 1e-12) << i;
    }
    const std::size_t edge_ref = 1 * 3 + 1; // Feld 1, λ_ref = 587,56 nm
    EXPECT_NE(si[edge_ref].distortion_chief, 0.0);
    EXPECT_NE(sc[edge_ref].rms_opd, si[edge_ref].rms_opd); // OPD um den anderen Mittelpunkt
    EXPECT_EQ(sc[0].rms_opd, si[0].rms_opd);               // Achse: derselbe Punkt
}

// Dateiformat: Lesen, Schreiben, Fehler; ohne Feld bzw. mit `ideal` bitgleich.
TEST(ReferencePoint, FileFormatAndCompatibility) {
    const LoadedSystem chief = lens("einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite,
                                    ReferencePointKind::ChiefRay, {0.0, 2.0});
    const std::string text = mko::write_system_json(chief.system);
    EXPECT_NE(text.find("\"reference_point\": \"chief_ray\""), std::string::npos);
    EXPECT_EQ(mko::read_system_json(text).reference_point, ReferencePointKind::ChiefRay);
    const LoadedSystem none = lens("einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite,
                                   std::nullopt, {0.0, 2.0});
    EXPECT_EQ(mko::write_system_json(none.system).find("reference_point"), std::string::npos);
    // ideal ausdrücklich = ohne Feld (bitgleicher Batch)
    const LoadedSystem ideal = lens("einzellinse_bk7.json", ReferenceSphereKind::ImageInfinite,
                                    ReferencePointKind::Ideal, {0.0, 2.0});
    EXPECT_TRUE(mko::make_system_batch(ideal, 2) == mko::make_system_batch(none, 2));
    // chief_ray nur mit Sphäre am Bildpunkt
    SystemDescription d = chief.system;
    d.reference_sphere = ReferenceSphereKind::LastSurface;
    try {
        mko::validate_system(d);
        FAIL() << "chief_ray mit last_surface angenommen";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "reference_point");
    }
    d.reference_sphere.reset();
    EXPECT_THROW(mko::validate_system(d), FormatError);
    // gesetzt nach dem Laden (CLI): der Aufbau des Batches lehnt ab
    LoadedSystem late = none;
    late.system.reference_sphere.reset();
    late.system.reference_point = ReferencePointKind::ChiefRay;
    EXPECT_THROW((void)mko::make_system_batch(late, 1), std::invalid_argument);
    // unbekannter Wert
    std::string bad = text;
    bad.replace(bad.find("chief_ray"), 9, "chief_ra");
    EXPECT_THROW((void)mko::read_system_json(bad), FormatError);

    // Nebenbedingung distortion: Lesen, Schreiben, Fehler
    SystemDescription c = none.system;
    mko::ConstraintSpec spec;
    spec.type = ConstraintType::Distortion;
    spec.max_distortion = 0.005;
    spec.weight = 1e4;
    c.constraints = mko::Constraints{};
    c.constraints->items.push_back(spec);
    mko::validate_system(c);
    const std::string ctext = mko::write_system_json(c);
    EXPECT_NE(ctext.find("\"max_distortion\": 0.005"), std::string::npos);
    const SystemDescription back = mko::read_system_json(ctext);
    ASSERT_TRUE(back.constraints.has_value());
    EXPECT_EQ(back.constraints->items.front(), spec);
    const mko::ParaxialConstraints pc = mko::make_paraxial_constraints(back);
    ASSERT_EQ(pc.size(), 2); // ein Residuum pro Feld
    EXPECT_EQ(pc.items[0].field, 0);
    EXPECT_EQ(pc.items[1].field, 1);
    EXPECT_EQ(pc.items[1].mode, mko::ConstraintMode::Max);
    EXPECT_EQ(pc.items[1].target, 0.005);
    EXPECT_FALSE(mko::supports_shift(spec));
    const auto expect_error = [&](SystemDescription e, const std::string& path) {
        try {
            mko::validate_system(e);
            FAIL() << "angenommen: " << path;
        } catch (const FormatError& err) {
            EXPECT_EQ(err.json_path(), path);
        }
    };
    SystemDescription e1 = c;
    e1.constraints->items[0].max_distortion.reset();
    expect_error(e1, "constraints.items[0].max_distortion");
    SystemDescription e2 = c;
    e2.constraints->items[0].max_distortion = -0.1;
    expect_error(e2, "constraints.items[0].max_distortion");
    SystemDescription e3 = c;
    e3.constraints->items[0].mode = mko::ConstraintMode::Min;
    expect_error(e3, "constraints.items[0].mode");
    SystemDescription e4 = c;
    e4.constraints->items.push_back(spec);
    expect_error(e4, "constraints.items[1].type");
    SystemDescription e5 = c;
    mko::ConstraintSpec petz;
    petz.type = ConstraintType::Petzval;
    petz.max_curvature = 0.01;
    petz.max_distortion = 0.01;
    e5.constraints->items.push_back(petz);
    expect_error(e5, "constraints.items[1].max_distortion");
}

// Die adjungierte Jacobi kennt ∂ȳ/∂x nicht und lehnt den Bildpunkt am Hauptstrahl ab.
TEST(ReferencePoint, AdjointJacobianRejectsChiefRay) {
    const LoadedSystem l = lens("einzellinse_bk7_biegung.json", ReferenceSphereKind::ImageInfinite,
                                ReferencePointKind::ChiefRay, {0.0, 2.0});
    const SystemBatchHost b = mko::make_system_batch(l, 1);
    const mko::VariableSet set(l.system, b.topology());
    const std::vector<double> x0 = set.extract(b, 0);
    EXPECT_THROW((void)mko::make_adjoint_batch(set, b, x0), std::invalid_argument);
    EXPECT_THROW((void)mko::adjoint_jacobian_cpu(set, b, x0), std::invalid_argument);
}
