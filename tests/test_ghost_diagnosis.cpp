// Diagnosebericht der Geisterplatten (M4-T6, docs/optimierung.md Abschnitt 5) an vorgegebenen
// Koeffizienten: Felder pro Platte (Σ|a|, dominante Ordnungen, Glas, φ_G, Lage, Strafe und
// Merit-Gewinn), Klassifikation pro Lage (A, B, B mit anomaler Teildispersion), Text und JSON.

#include "ghost_test_systems.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/trace/ghost_plate.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace mko::test {
namespace {

// Testsystem einzellinse_geisterplatten_exakt.json (Platten 0: CaF₂ und 1: Flint bei 1 mm nach
// der Blende, 2: N-BK7 bei 4 mm, 3: N-BK7 hinter der Linse; K = 4, ε = 1e-7) mit dem Spektrum
// `wavelengths` (Gewichte 1) und allen Koeffizienten 0.
struct DiagCase {
    LoadedSystem loaded;
    SystemBatchHost batch;
    GhostPenalty penalty;
};

DiagCase make_case(std::vector<double> wavelengths = {587.56}) {
    SystemDescription d = load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    d.spectrum.wavelengths_nm = wavelengths;
    d.spectrum.weights.assign(wavelengths.size(), 1.0);
    for (GhostPlate& g : d.ghost_plates) {
        std::fill(g.coefficients.begin(), g.coefficients.end(), 0.0);
    }
    DiagCase s;
    s.loaded = prepare_system(d, std::filesystem::path(kResidualSystems));
    s.batch = make_system_batch(s.loaded, 1);
    s.penalty = make_ghost_penalty(s.loaded.system);
    return s;
}

GhostDiagnosis diagnose(const DiagCase& s, const BatchMeritFn& merit = {}) {
    return diagnose_ghost_plates(s.loaded.system, s.loaded.catalog, s.batch, 0, s.penalty, merit);
}

// Einfache Merit als Orakel für Merit am Punkt und ohne Platte: 1 + Σ a².
std::vector<double> sum_of_squares(const SystemBatchHost& b) {
    std::vector<double> f(static_cast<std::size_t>(b.variants()), 1.0);
    for (int v = 0; v < b.variants(); ++v) {
        for (int g = 0; g < b.ghost_plates(); ++g) {
            for (int k = 0; k < b.ghost_terms(); ++k) {
                f[static_cast<std::size_t>(v)] += b.ghost_coeff(v, g, k) * b.ghost_coeff(v, g, k);
            }
        }
    }
    return f;
}

// Felder pro Platte gegen unabhängige Rechnungen.
TEST(GhostDiagnosis, PlateFields) {
    DiagCase s = make_case();
    const std::vector<double> a = {1e-4, -5e-4, 2e-5, 3e-4};
    for (int k = 0; k < 4; ++k) {
        s.batch.ghost_coeff(0, 2, k) = a[static_cast<std::size_t>(k)];
    }
    s.batch.ghost_coeff(0, 3, 1) = 5e-7; // unter der Schwelle 10 ε = 1e-6
    const GhostDiagnosis d = diagnose(s, sum_of_squares);
    ASSERT_EQ(d.plates.size(), 4u);
    const GhostPlateReport& r = d.plates[2];
    EXPECT_EQ(r.plate, 2);
    EXPECT_EQ(r.after_surface, 0);
    EXPECT_EQ(r.distance, 4.0);
    EXPECT_EQ(r.table_surface, s.batch.ghost_surface(2));
    EXPECT_EQ(r.norm_radius, 2.5);
    EXPECT_EQ(r.order, 4);
    EXPECT_EQ(r.coefficients, a);
    EXPECT_DOUBLE_EQ(r.sum_abs, 1e-4 + 5e-4 + 2e-5 + 3e-4);
    EXPECT_EQ(r.max_abs, 5e-4);
    EXPECT_EQ(r.activity_threshold, 1e-6);
    EXPECT_TRUE(r.active);
    // dominante Ordnungen: |a_k| ≥ 0,1 · max, absteigend (Ordnung 3 mit 2e-5 fällt heraus)
    EXPECT_EQ(r.dominant_orders, (std::vector<int>{2, 4, 1}));
    // φ_G = −2 (n_G − 1) t₂, t₂ = Σ a_k (−1)^{k+1} k (k + 1)/r_g²
    const double n_ref =
        material_index(s.loaded.system.ghost_plates[2].glass, 587.56, s.loaded.catalog);
    EXPECT_EQ(r.glass.n_ref, n_ref);
    const double t2 = (2.0 * 1e-4 - 6.0 * -5e-4 + 12.0 * 2e-5 - 20.0 * 3e-4) / (2.5 * 2.5);
    EXPECT_NEAR(r.power, -2.0 * (n_ref - 1.0) * t2, 1e-15);
    EXPECT_NEAR(r.petzval, r.power / n_ref, 1e-15);
    // Lage im kollimierten Strahl vor der Linse: Randstrahlhöhe = Pupillenradius, Hauptstrahl
    // (1 rad) = Abstand von der Blende
    EXPECT_EQ(r.marginal_height, 2.5);
    EXPECT_DOUBLE_EQ(r.chief_height_per_rad, 4.0);
    EXPECT_DOUBLE_EQ(d.plates[0].chief_height_per_rad, 1.0);
    // Strafe und Merit ohne Platte (Orakel 1 + Σ a²)
    EXPECT_TRUE(r.penalized);
    EXPECT_EQ(r.penalty, ghost_penalty_merit(s.penalty, s.batch, 0, 2));
    double sq = 0.0;
    for (const double x : a) {
        sq += x * x;
    }
    EXPECT_DOUBLE_EQ(d.merit, 1.0 + sq + 5e-7 * 5e-7);
    EXPECT_DOUBLE_EQ(r.merit_without, 1.0 + 5e-7 * 5e-7);
    EXPECT_NEAR(r.merit_gain, -sq, 1e-15); // Differenz zweier Werte nahe 1
    // inaktive Platten
    EXPECT_FALSE(d.plates[3].active);
    EXPECT_TRUE(d.plates[3].dominant_orders.empty());
    EXPECT_FALSE(d.plates[0].active);
    // Glasdaten aus den Linien g, F, d, C
    EXPECT_NEAR(d.plates[0].glass.abbe, 95.0, 0.05);            // CaF₂
    EXPECT_NEAR(d.plates[0].glass.delta_partial, 0.0546, 1e-3); // anomal
    EXPECT_NEAR(d.plates[1].glass.abbe, 26.94, 0.05);           // Flint (Buchdahl)
    EXPECT_NEAR(d.plates[1].glass.delta_partial, 0.0, 1e-3);
    EXPECT_NEAR(d.plates[2].glass.abbe, 64.17, 0.05); // N-BK7
    EXPECT_EQ(d.plates[0].glass.name, "CAF2");
    EXPECT_EQ(d.plates[1].glass.name, "buchdahl(n0=1.76, nu=-0.18/0.112)");
    // Lagen: (0, 1 mm) mit Platten 0 und 1, (0, 4 mm), (2, 10 mm)
    ASSERT_EQ(d.positions.size(), 3u);
    EXPECT_EQ(d.positions[0].plates, (std::vector<int>{0, 1}));
    EXPECT_TRUE(d.positions[0].active_plates.empty());
    EXPECT_EQ(d.positions[0].recommendation, GhostOperator::None);
    EXPECT_EQ(d.positions[1].active_plates, (std::vector<int>{2}));
    EXPECT_EQ(d.positions[1].recommendation, GhostOperator::A); // eine Wellenlänge
    EXPECT_EQ(d.recommended_position, 1);
    EXPECT_EQ(d.recommendation, GhostOperator::A);
    EXPECT_NE(d.recommendation_text.find("Operator A"), std::string::npos);
    EXPECT_NE(d.recommendation_text.find("nach Fläche 0 im Abstand 4 mm"), std::string::npos);
    EXPECT_NE(d.recommendation_text.find("N-BK7"), std::string::npos);
}

// Ohne aktive Platte keine Empfehlung; ohne Merit-Funktion NaN statt Merit-Werten.
TEST(GhostDiagnosis, NothingActive) {
    const DiagCase s = make_case();
    const GhostDiagnosis d = diagnose(s);
    EXPECT_TRUE(std::isnan(d.merit));
    EXPECT_TRUE(std::isnan(d.plates[0].merit_gain));
    EXPECT_EQ(d.recommended_position, -1);
    EXPECT_EQ(d.recommendation, GhostOperator::None);
    EXPECT_NE(d.recommendation_text.find("keine Empfehlung"), std::string::npos);
    for (const GhostPositionReport& p : d.positions) {
        EXPECT_EQ(p.recommendation, GhostOperator::None);
    }
}

// Zwei Wellenlängen, eine Platte: Farbanteil χ gleich dem eines einzelnen Glases → A.
TEST(GhostDiagnosis, SingleGlassIsMonochromatic) {
    DiagCase s = make_case({486.13, 656.27});
    s.batch.ghost_coeff(0, 1, 1) = 1e-3;
    const GhostDiagnosis d = diagnose(s);
    const GhostPositionReport& p = d.positions[0];
    EXPECT_EQ(d.merit_wavelengths, 2);
    EXPECT_EQ(d.short_nm, 486.13);
    EXPECT_EQ(d.long_nm, 656.27);
    EXPECT_NEAR(p.chromatic_ratio, p.single_glass_ratio, 1e-12);
    const double dn = s.batch.ghost_index(0, 1, 0) - s.batch.ghost_index(0, 1, 1);
    EXPECT_NEAR(p.single_glass_ratio, dn / (s.batch.paraxial_ghost_index(0, 1) - 1.0), 1e-15);
    EXPECT_EQ(p.recommendation, GhostOperator::A);
}

// Kron- und Flintplatte mit entgegengesetzter Brechkraft (monochromatisch kompensiert): Die
// Muster der Wellenlängen unterscheiden sich viel stärker als bei einem Glas → B. Mit CaF₂ als
// Kronglas bei drei Wellenlängen: B mit anomaler Teildispersion; bei zwei Wellenlängen nur B.
TEST(GhostDiagnosis, DispersionDifferenceIsChromatic) {
    for (const int wavelengths : {2, 3}) {
        DiagCase s =
            wavelengths == 2 ? make_case({486.13, 656.27}) : make_case({486.13, 587.56, 656.27});
        const double n0 = s.batch.paraxial_ghost_index(0, 0); // CaF₂
        const double n1 = s.batch.paraxial_ghost_index(0, 1); // Flint
        const double a1 = 1e-3;
        s.batch.ghost_coeff(0, 1, 0) = a1;
        s.batch.ghost_coeff(0, 0, 0) = -a1 * (n1 - 1.0) / (n0 - 1.0) * 0.98; // 2 % Restbrechkraft
        const GhostDiagnosis d = diagnose(s);
        const GhostPositionReport& p = d.positions[0];
        EXPECT_EQ(p.active_plates, (std::vector<int>{0, 1}));
        EXPECT_GT(p.chromatic_ratio, 2.0 * p.single_glass_ratio) << wavelengths;
        if (wavelengths == 2) {
            EXPECT_EQ(p.recommendation, GhostOperator::B);
            EXPECT_NE(p.reason.find("kein Nachweis eines sekundären Spektrums"), std::string::npos);
        } else {
            EXPECT_EQ(p.recommendation, GhostOperator::BAnomalous);
            EXPECT_NE(d.recommendation_text.find("Fluorit"), std::string::npos);
        }
        EXPECT_EQ(d.recommendation, p.recommendation);
    }
    // Verlangt man für die Fluorit-Platte einen Anteil von 90 % an Σ|a| der Lage, bleibt es bei
    // B ohne anomale Teildispersion (der Fluorit-Anteil ist hier etwa 63 %).
    DiagCase s = make_case({486.13, 587.56, 656.27});
    GhostDiagnosisOptions o;
    o.anomalous_share = 0.9; // Fluorit muss fast allein aktiv sein
    const double n0 = s.batch.paraxial_ghost_index(0, 0);
    const double n1 = s.batch.paraxial_ghost_index(0, 1);
    s.batch.ghost_coeff(0, 1, 0) = 1e-3;
    s.batch.ghost_coeff(0, 0, 0) = -1e-3 * (n1 - 1.0) / (n0 - 1.0) * 0.98;
    const GhostDiagnosis d =
        diagnose_ghost_plates(s.loaded.system, s.loaded.catalog, s.batch, 0, s.penalty, {}, o);
    EXPECT_EQ(d.positions[0].recommendation, GhostOperator::B);
}

// Text und JSON enthalten die Felder des Berichts.
TEST(GhostDiagnosis, TextAndJson) {
    DiagCase s = make_case({486.13, 587.56, 656.27});
    s.batch.ghost_coeff(0, 1, 1) = -2e-4;
    const GhostDiagnosis d = diagnose(s, sum_of_squares);
    std::ostringstream os;
    write_ghost_diagnosis_text(d, os);
    const std::string text = os.str();
    for (const char* expected :
         {"Geisterplatten (Diagnose", "buchdahl(n0=1.76",
          "Lagen:", "Befund:", "Begründung:", "Empfehlung: Operator A", "Platte 1 a_k [mm]:"}) {
        EXPECT_NE(text.find(expected), std::string::npos) << expected << '\n' << text;
    }
    const nlohmann::json j = nlohmann::json::parse(ghost_diagnosis_json(d));
    EXPECT_EQ(j.at("merit_wavelengths").get<int>(), 3);
    ASSERT_EQ(j.at("plates").size(), 4u);
    const nlohmann::json& p1 = j.at("plates").at(1);
    EXPECT_EQ(p1.at("plate").get<int>(), 1);
    EXPECT_EQ(p1.at("after_surface").get<int>(), 0);
    EXPECT_EQ(p1.at("distance").get<double>(), 1.0);
    EXPECT_EQ(p1.at("glass").at("name").get<std::string>(), "buchdahl(n0=1.76, nu=-0.18/0.112)");
    EXPECT_EQ(p1.at("dominant_orders"), nlohmann::json::array({2}));
    EXPECT_TRUE(p1.at("active").get<bool>());
    EXPECT_EQ(p1.at("coefficients").size(), 4u);
    EXPECT_DOUBLE_EQ(p1.at("sum_abs").get<double>(), 2e-4);
    EXPECT_TRUE(p1.at("merit_gain").is_number());
    EXPECT_EQ(j.at("positions").size(), 3u);
    EXPECT_EQ(j.at("positions").at(0).at("recommendation").get<std::string>(), "A");
    EXPECT_EQ(j.at("recommended_position").get<int>(), 0);
    EXPECT_EQ(j.at("recommendation").get<std::string>(), "A");
    // NaN als null
    const nlohmann::json j0 = nlohmann::json::parse(ghost_diagnosis_json(diagnose(make_case())));
    EXPECT_TRUE(j0.at("merit").is_null());
    EXPECT_TRUE(j0.at("recommended_position").is_null());
}

// Datei und Batch müssen zusammenpassen.
TEST(GhostDiagnosis, RejectsMismatch) {
    const DiagCase s = make_case();
    const SystemDescription other = load_residual_system("einzellinse_geisterplatte.json").system;
    EXPECT_THROW((void)diagnose_ghost_plates(other, s.loaded.catalog, s.batch, 0, s.penalty, {}),
                 std::invalid_argument);
    EXPECT_THROW(
        (void)diagnose_ghost_plates(s.loaded.system, s.loaded.catalog, s.batch, 1, s.penalty, {}),
        std::out_of_range);
    EXPECT_EQ(std::string(to_string(GhostOperator::BAnomalous)), "B_anomalous");
}

} // namespace
} // namespace mko::test
