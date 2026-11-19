// Adapter Raytatouille ↔ mikroptim.system/1 (M9-T2): Roundtrip rtt → mko → rtt und
// mko → rtt → mko (Gleichheit der abbildbaren Felder, Paraxialdaten bitgleich bzw. innerhalb
// kParaxialRelative), Querabgleich mit Zahlen aus der Raytatouille-Doku, Fehler und Warnungen
// für nicht abbildbare Elemente, Kommandozeile `mikroptim convert`.
//
// Die Testsysteme sind eigene. Das „Plankonvex-Singlet“ und das Skript-Singlet folgen den
// Beispielen in docs/dateiformat.md von Raytatouille (github.com/PhotonPower/Raytatouille, MIT);
// von dort stammen nur die veröffentlichten Zahlenwerte für den Querabgleich.

#include "cli/convert_command.hpp"
#include "mikroptim/glass/air_index.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/io/raytatouille.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "temp_paths.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

const std::string kRoot = std::string(MKO_SOURCE_DIR) + "/";
const std::string kBasis = kRoot + "data/glass/basis.json";

const mko::GlassCatalog& basis() {
    static const mko::GlassCatalog c = mko::GlassCatalog::from_file(kBasis);
    return c;
}

mko::RttImportOptions basis_import() {
    mko::RttImportOptions o;
    o.glass_catalogs = {"data/glass/basis.json"};
    o.glass_exists = [](std::string_view g) { return basis().contains(g); };
    o.glass_catalog_name = [](std::string_view g) {
        return basis().contains(g) ? std::string("BASIS") : std::string();
    };
    return o;
}

mko::RttExportOptions basis_export() {
    mko::RttExportOptions o;
    o.glass_catalog_name = [](std::string_view g) {
        return basis().contains(g) ? std::string("BASIS") : std::string();
    };
    return o;
}

json rel(double t) {
    return json{{"reference", "relative_to_preceding"}, {"position", {0.0, 0.0, t}}};
}
json conic(double r) {
    return json{{"base", {{"type", "conic"}, {"radius", r}}}};
}
json circ(double r) {
    return json{{"type", "circular"}, {"radius", r}};
}

// Kanonisches Testsystem in der Umgebung VACUUM: Singlet, verkittetes Triplet mit zwei
// Segmenten, Hilfsfläche in Luft (thin_element), Gewichte, Felder mit Vorzeichen.
json vacuum_system() {
    json s;
    s["schema_version"] = "0.11.0";
    s["name"] = "Testsystem Vakuum";
    s["units"] = {{"length", "mm"}, {"wavelength", "um"}};
    s["environment"] = {{"medium", "VACUUM"}};
    s["wavelengths"] = json::array({json{{"um", 0.4861}, {"weight", 0.5}},
                                    json{{"um", 0.5876}, {"reference", true}},
                                    json{{"um", 0.6563}, {"weight", 0.5}}});
    s["aperture"] = {{"type", "epd"}, {"value", 6.25}};
    s["fields"] = {{"type", "angle_deg"},
                   {"points", json::array({json::object(), json{{"y", 1.5}},
                                           json{{"y", -3.0}, {"weight", 2.0}}})}};
    json children = json::array();
    children.push_back({{"type", "stop"},
                        {"name", "stop"},
                        {"surfaces", json::array({{{"id", "STO"}, {"aperture", circ(3.125)}}})}});
    children.push_back(
        {{"type", "lens"},
         {"name", "L1"},
         {"pose", rel(4.0)},
         {"material", "CONST:1.7"},
         {"surfaces",
          json::array({{{"id", "L1.S1"}, {"shape", conic(41.3)}, {"aperture", circ(6.0)}},
                       {{"id", "L1.S2"}, {"pose", rel(3.5)}, {"aperture", circ(6.0)}}})}});
    children.push_back(
        {{"type", "lens"},
         {"name", "D"},
         {"pose", rel(0.8)},
         {"material", json::array({"CONST:1.62", "CONST:1.48"})},
         {"surfaces",
          json::array(
              {{{"id", "D.A"}, {"shape", conic(-120.0)}, {"aperture", circ(5.5)}},
               {{"id", "D.B"}, {"pose", rel(2.0)}, {"shape", conic(25.5)}, {"aperture", circ(5.5)}},
               {{"id", "D.C"},
                {"pose", rel(4.25)},
                {"shape", conic(-60.0)},
                {"aperture", circ(5.25)}}})}});
    children.push_back({{"type", "thin_element"},
                        {"name", "H1"},
                        {"pose", rel(1.0)},
                        {"surfaces", json::array({{{"id", "H1.S"}, {"aperture", circ(5.0)}}})}});
    children.push_back({{"type", "detector"},
                        {"name", "image"},
                        {"pose", rel(30.0)},
                        {"surfaces", json::array({{{"id", "IMG"}}})}});
    s["root"] = {{"type", "assembly"}, {"name", "system"}, {"children", children}};
    s["paths"] = json::array({{{"name", "main"}, {"events", "auto"}}});
    return s;
}

// Kanonisches Testsystem mit Katalog-Gläsern in Luft (Umgebung Standard).
json catalog_system() {
    json s = vacuum_system();
    s.erase("environment");
    s["name"] = "Testsystem Katalog";
    s["root"]["children"][1]["material"] = "BASIS:N-BK7";
    s["root"]["children"][2]["material"] = json::array({"BASIS:CAF2", "BASIS:F_SILICA"});
    return s;
}

mko::SystemDescription import(const json& j, const mko::RttImportOptions& o = basis_import(),
                              std::vector<std::string>* w = nullptr) {
    return mko::system_from_rtt_json(j.dump(), o, w);
}

std::vector<mko::RttIssue> import_issues(const json& j,
                                         const mko::RttImportOptions& o = basis_import()) {
    try {
        (void)import(j, o);
    } catch (const mko::RttConversionError& e) {
        return e.issues();
    }
    return {};
}

bool has_issue(const std::vector<mko::RttIssue>& issues, const std::string& location,
               const std::string& text) {
    return std::any_of(issues.begin(), issues.end(), [&](const mko::RttIssue& i) {
        return i.location == location && i.message.find(text) != std::string::npos;
    });
}

bool has_warning(const std::vector<std::string>& w, const std::string& text) {
    return std::any_of(w.begin(), w.end(),
                       [&](const std::string& s) { return s.find(text) != std::string::npos; });
}

// Zahl für RecordProperty (wissenschaftlich, 3 Stellen).
std::string sci(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3e", v);
    return buf;
}

std::string issues_text(const std::vector<mko::RttIssue>& issues) {
    std::string s;
    for (const auto& i : issues) {
        s += i.location + ": " + i.message + "\n";
    }
    return s;
}

} // namespace

// ---- Umrechnung der Zahlen ----------------------------------------------------------------

TEST(RttNumbers, WavelengthsShiftDecimalExactly) {
    for (const double um : {0.4861, 0.5876, 0.6563, 0.54607, 0.365, 1.014, 0.40466, 2.5}) {
        const double nm = mko::rtt_um_to_nm(um);
        EXPECT_EQ(mko::rtt_nm_to_um(nm), um);
    }
    EXPECT_EQ(mko::rtt_um_to_nm(0.5876), 587.6);
    EXPECT_EQ(mko::rtt_um_to_nm(0.54607), 546.07);
    EXPECT_EQ(mko::rtt_nm_to_um(587.56), 0.58756);
    EXPECT_EQ(mko::rtt_nm_to_um(486.13), 0.48613);
    // Dezimalzahlen bis 15 Stellen: Hin- und Rückweg in beiden Richtungen bitgleich
    for (int i = 0; i < 2000; ++i) {
        // std::to_string: sechs Nachkommastellen, also kurze Dezimalzahlen
        const double nm = std::stod(std::to_string(300.0 + 0.137137 * i));
        EXPECT_EQ(mko::rtt_um_to_nm(mko::rtt_nm_to_um(nm)), nm) << nm;
        const double um = std::stod(std::to_string(0.3 + 0.000371 * i));
        EXPECT_EQ(mko::rtt_nm_to_um(mko::rtt_um_to_nm(um)), um) << um;
    }
    // beliebige Doubles (17 Stellen): höchstens 2 ulp, weil die Binaden von µm und nm nicht
    // zusammenpassen
    double nm = 300.0;
    for (int i = 0; i < 2000; ++i) {
        nm = std::nextafter(nm + 0.3711, 1e9);
        const double back = mko::rtt_um_to_nm(mko::rtt_nm_to_um(nm));
        EXPECT_LE(std::abs(back - nm), 2.0 * (std::nextafter(nm, 1e9) - nm)) << nm;
    }
}

TEST(RttNumbers, RadiusCurvatureRoundtrip) {
    // kurze Radien aus Dateien kommen über c = 1/R exakt zurück
    for (const double r : {51.68, -413.41, 58.82, 41.3, -120.0, 25.5, 0.3, -1e4, 153.8}) {
        const double c = mko::rtt_curvature_from_radius(r);
        EXPECT_EQ(mko::rtt_radius_from_curvature(c), r);
    }
    // Krümmungen: 1/R == c, wo es ein solches R gibt (sonst höchstens 1 ulp Abweichung)
    int exact = 0;
    double c = 1e-4;
    for (int i = 0; i < 2000; ++i) {
        c = std::nextafter(c * 1.0137, 1.0);
        const double back = mko::rtt_curvature_from_radius(mko::rtt_radius_from_curvature(c));
        EXPECT_LE(std::abs(back - c), std::abs(std::nextafter(c, 0.0) - c)) << c;
        exact += back == c ? 1 : 0;
    }
    EXPECT_GT(exact, 1200); // die Mehrzahl ist bitgleich
}

// ---- Raytatouille → Mikroptim → Raytatouille ----------------------------------------------

TEST(RttImport, CanonicalVacuumSystemValues) {
    std::vector<std::string> w;
    const mko::SystemDescription s = import(vacuum_system(), basis_import(), &w);
    EXPECT_TRUE(w.empty()) << w.front();
    EXPECT_EQ(s.name, "Testsystem Vakuum");
    EXPECT_EQ(s.spectrum.wavelengths_nm, (std::vector<double>{486.1, 587.6, 656.3}));
    EXPECT_EQ(s.spectrum.weights, (std::vector<double>{0.5, 1.0, 0.5}));
    EXPECT_EQ(s.spectrum.reference_nm, 587.6);
    EXPECT_EQ(s.source.field_angles_deg, (std::vector<double>{0.0, 1.5, -3.0}));
    EXPECT_EQ(s.source.field_weights, (std::vector<double>{1.0, 1.0, 2.0}));
    ASSERT_EQ(s.surfaces.size(), 7u);
    const std::vector<double> t{4.0, 3.5, 0.8, 2.0, 4.25, 1.0, 30.0};
    const std::vector<double> sd{3.125, 6.0, 6.0, 5.5, 5.5, 5.25, 5.0};
    const std::vector<std::string> ids{"STO", "L1.S1", "L1.S2", "D.A", "D.B", "D.C", "H1.S"};
    for (std::size_t i = 0; i < s.surfaces.size(); ++i) {
        EXPECT_EQ(s.surfaces[i].thickness, t[i]) << i;
        EXPECT_EQ(s.surfaces[i].semi_diameter, sd[i]) << i;
        EXPECT_EQ(s.surfaces[i].comment, ids[i]) << i;
    }
    EXPECT_EQ(s.surfaces[0].type, mko::SurfaceType::Stop);
    EXPECT_EQ(s.surfaces[1].type, mko::SurfaceType::Sphere);
    EXPECT_EQ(s.surfaces[1].curvature, 1.0 / 41.3);
    EXPECT_EQ(s.surfaces[2].type, mko::SurfaceType::Plane);
    EXPECT_EQ(s.surfaces[3].curvature, 1.0 / -120.0);
    EXPECT_EQ(s.surfaces[6].type, mko::SurfaceType::Plane);
    EXPECT_EQ(s.surfaces[1].material.kind, mko::MaterialKind::Constant);
    EXPECT_EQ(s.surfaces[1].material.n, 1.7);
    EXPECT_EQ(s.surfaces[2].material.kind, mko::MaterialKind::Air);
    EXPECT_EQ(s.surfaces[3].material.n, 1.62);
    EXPECT_EQ(s.surfaces[4].material.n, 1.48);
    EXPECT_EQ(s.surfaces[5].material.kind, mko::MaterialKind::Air);
    EXPECT_EQ(s.surfaces[6].material.kind, mko::MaterialKind::Air);
}

TEST(RttRoundtrip, RttToMikroptimToRttIsIdentical) {
    for (const json& original : {vacuum_system(), catalog_system()}) {
        std::vector<std::string> w;
        const mko::SystemDescription s = import(original, basis_import(), &w);
        EXPECT_TRUE(w.empty()) << w.front();
        // Umgebung der Ausgabe wie die der Eingabe (Standard: Luft)
        mko::RttExportOptions eo = basis_export();
        eo.environment.vacuum = original.contains("environment");
        const std::string text = mko::system_to_rtt_json(s, eo, &w);
        EXPECT_TRUE(w.empty()) << w.front();
        EXPECT_EQ(json::parse(text), original) << text;
        // und das Mikroptim-Modell nach einem zweiten Durchgang ebenfalls gleich
        const mko::SystemDescription again = mko::system_from_rtt_json(text, basis_import());
        EXPECT_EQ(again, s);
        // die Datei ist eine gültige Mikroptim-Datei und schreibt sich stabil
        EXPECT_EQ(mko::read_system_json(mko::write_system_json(s)), s);
    }
}

TEST(RttRoundtrip, ParaxialDataUnchangedBitwise) {
    const mko::SystemDescription s = import(catalog_system());
    const std::string text = mko::system_to_rtt_json(s, basis_export());
    const mko::SystemDescription back = mko::system_from_rtt_json(text, basis_import());
    const mko::ParaxialData a = mko::compute_paraxial(s, basis());
    const mko::ParaxialData b = mko::compute_paraxial(back, basis());
    EXPECT_EQ(a.focal_length, b.focal_length);
    EXPECT_EQ(a.back_focal_distance, b.back_focal_distance);
    EXPECT_TRUE(std::isfinite(a.focal_length));
}

// Nicht kanonische Datei: verschachtelte Baugruppe, absolute Lagen, Geschwisterbezug,
// Params mit Variablen-Markierung und Bindung an die Parametertabelle, Drehung 0.
TEST(RttImport, GeneralPlacementAndParams) {
    json s = vacuum_system();
    s["parameters"] = json::array({{{"name", "D1"}, {"value", 3.5}},
                                   {{"name", "G"}, {"values", {2.0, 7.0}}, {"variable", true}}});
    s["configurations"] = json::array({{{"name", "a"}}, {{"name", "b"}}});
    json& ch = s["root"]["children"];
    // L1 absolut bei z = 4, zweite Fläche über den Parameter D1
    ch[1]["pose"] = {{"position", {0.0, 0.0, 4.0}}, {"rotation_deg", {0.0, 0.0, 0.0}}};
    ch[1]["surfaces"][1]["pose"] = {{"position", {0.0, 0.0, {{"param", "D1"}}}}};
    ch[1]["surfaces"][0]["shape"]["base"]["radius"] = {
        {"value", 41.3}, {"variable", true}, {"min", 20.0}};
    // D in einer Unterbaugruppe bei z = 8.3, Flächen absolut im Element
    json d = ch[2];
    d.erase("pose");
    d["surfaces"][1]["pose"] = {{"position", {0.0, 0.0, 2.0}}};
    d["surfaces"][2]["pose"] = {{"position", {0.0, 0.0, 6.25}}};
    json sub = {{"type", "assembly"},
                {"name", "hinten"},
                {"pose", {{"position", {0.0, 0.0, 8.3}}}},
                {"children", json::array({d})}};
    // H1 relativ zum Geschwister (Unterbaugruppe) um 7.25 + 1.0
    ch[3]["pose"] = {{"reference", "relative_to_sibling"}, {"position", {0.0, 0.0, 7.25}}};
    ch[2] = sub;
    ch[4]["pose"] = {{"position", {0.0, 0.0, {{"param", "G"}}}},
                     {"reference", "relative_to_preceding"}};
    std::vector<std::string> w;
    const mko::SystemDescription m = import(s, basis_import(), &w);
    const std::vector<double> t{4.0, 3.5, 8.3 - (4.0 + 3.5), 2.0, 6.25 - 2.0, 7.25 - 6.25, 2.0};
    ASSERT_EQ(m.surfaces.size(), t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(m.surfaces[i].thickness, t[i]) << i;
    }
    EXPECT_EQ(m.surfaces[1].curvature, 1.0 / 41.3);
    EXPECT_TRUE(has_warning(w, "Optimierungsvariable")) << w.size();
    EXPECT_TRUE(has_warning(w, "Parameter D1"));
    EXPECT_TRUE(has_warning(w, "ersten Konfiguration"));
    EXPECT_TRUE(has_warning(w, "Konfigurationen"));
}

// Querabgleich mit Raytatouille-Doku (docs/dateiformat.md, Beispiel 1 und Skript):
// Raytatouille rechnet absolut in Ciddor-Luft, also mit n_rel = n/n_Luft und EFL_rtt =
// EFL_rel/n_Luft. n_Luft folgt aus der veröffentlichten EFL 100,0527 mm von Beispiel 1; geprüft
// werden zwei weitere veröffentlichte Zahlen: die Bildfläche im paraxialen Fokus bei
// z = 106,442 mm (Beispiel 1) und EFL = 97,631 mm des Skript-Singlets (R1 = 100, R2 = −100).
TEST(RttImport, CrossCheckWithPublishedRaytatouilleValues) {
    const char* example1 = R"({
      "schema_version": "0.11.0", "name": "Plankonvex-Singlet",
      "units": {"length": "mm", "wavelength": "um"},
      "wavelengths": [{"um": 0.4861}, {"um": 0.5876, "reference": true}, {"um": 0.6563}],
      "aperture": {"type": "epd", "value": 20.0},
      "fields": {"type": "angle_deg", "points": [{}, {"y": 3.5}, {"y": 5.0}]},
      "root": {"type": "assembly", "name": "system", "children": [
        {"type": "stop", "name": "stop",
         "surfaces": [{"id": "STO", "aperture": {"type": "circular", "radius": 10.0}}]},
        {"type": "lens", "name": "L1", "pose": {"position": [0, 0, 5.0]}, "material": "CONST:1.5168",
         "surfaces": [
           {"id": "L1.S1", "shape": {"base": {"type": "conic", "radius": 51.68}},
            "aperture": {"type": "circular", "radius": 12.7}},
           {"id": "L1.S2", "pose": {"position": [0, 0, 4.0]},
            "aperture": {"type": "circular", "radius": 12.7}}]},
        {"type": "detector", "name": "image", "pose": {"position": [0, 0, 106.442]},
         "surfaces": [{"id": "IMG"}]}
      ]},
      "paths": [{"name": "main", "events": "auto"}]
    })";
    std::vector<std::string> w;
    const mko::SystemDescription s = mko::system_from_rtt_json(example1, {}, &w);
    EXPECT_TRUE(has_warning(w, "Dispersion der Luft")); // CONST bei drei Wellenlängen
    ASSERT_EQ(s.surfaces.size(), 3u);
    EXPECT_EQ(s.surfaces[0].thickness, 5.0);
    EXPECT_EQ(s.surfaces[1].thickness, 4.0);
    EXPECT_EQ(s.surfaces[2].thickness, 106.442 - 9.0);
    EXPECT_EQ(s.surfaces[0].semi_diameter, 10.0);
    // Umrechnung (Entscheidung Maintainer 2026-11-19): λ = λ_vac/n_Luft, n_rel = n/n_Luft(λ_ref)
    const double n_air = mko::ciddor_air_index(0.5876, 20.0, 1.0);
    EXPECT_EQ(s.spectrum.reference_nm, 0.5876 * 1000.0 / n_air);
    EXPECT_EQ(s.surfaces[1].material.n, 1.5168 / n_air);

    // Raytatouille: EFL = 1/Φ absolut = EFL_rel/n_Luft; Bildfläche im paraxialen Fokus
    const mko::ParaxialData p = mko::compute_paraxial(s, mko::GlassCatalog{});
    EXPECT_NEAR(p.focal_length / n_air, 100.0527, mko::test::kRttPublishedLength);
    EXPECT_NEAR(p.back_focal_distance, 106.442 - 9.0, mko::test::kRttPublishedLength);
    EXPECT_NEAR(p.image_defocus, 0.0, mko::test::kRttPublishedLength);
    RecordProperty("efl_abweichung_mm", sci(p.focal_length / n_air - 100.0527));
    RecordProperty("bfl_abweichung_mm", sci(p.back_focal_distance - 97.442));

    json script = json::parse(example1);
    script["root"]["children"][1]["surfaces"][0]["shape"]["base"]["radius"] = 100.0;
    script["root"]["children"][1]["surfaces"][1]["shape"] = conic(-100.0);
    script["root"]["children"][1]["surfaces"][1]["pose"]["position"][2] = 5.0;
    script["root"]["children"][2]["pose"]["position"][2] = 5.0 + 5.0 + 95.0;
    const mko::SystemDescription s2 = mko::system_from_rtt_json(script.dump(), {});
    const mko::ParaxialData p2 = mko::compute_paraxial(s2, mko::GlassCatalog{});
    EXPECT_NEAR(p2.focal_length / n_air, 97.631, mko::test::kRttPublishedLength);
    RecordProperty("efl_skript_abweichung_mm", sci(p2.focal_length / n_air - 97.631));

    // ohne Umrechnung (--no-air-conversion): Zahlen unverändert, f = R/(n − 1)
    mko::RttImportOptions raw;
    raw.air_conversion = false;
    w.clear();
    const mko::SystemDescription r = mko::system_from_rtt_json(example1, raw, &w);
    EXPECT_TRUE(has_warning(w, "CONST")); // Umgebung Luft, CONST relativ übernommen
    EXPECT_EQ(r.spectrum.reference_nm, 587.6);
    EXPECT_EQ(r.surfaces[1].material.n, 1.5168);
    const mko::ParaxialData direct = mko::compute_paraxial(r, mko::GlassCatalog{});
    EXPECT_NEAR(direct.focal_length, 51.68 / 0.5168, mko::test::kParaxialRelative * 100.0);
}

// Katalogglas in Luft bei 20 °C, 1 atm: Raytatouille rechnet die Brechzahl relativ zur Umgebung
// als n_rel(λ_vac/n_L)·n_L/n_L (Katalogformel bei der Wellenlänge in Luft); mit der umgerechneten
// Wellenlänge ist das genau die Mikroptim-Brechzahl. Ohne Umrechnung blieb die Abweichung
// n_rel(λ_vac) − n_rel(λ_vac/n_L) von einigen 10⁻⁶.
TEST(RttAir, CatalogGlassMatchesRaytatouilleRelativeIndex) {
    const mko::SystemDescription s = import(catalog_system());
    mko::RttImportOptions raw = basis_import();
    raw.air_conversion = false;
    const mko::SystemDescription old = import(catalog_system(), raw);
    const std::vector<double> um{0.4861, 0.5876, 0.6563};
    double worst_new = 0.0;
    double worst_old = 0.0;
    for (const char* glass : {"N-BK7", "CAF2", "F_SILICA"}) {
        const mko::Glass& g = basis().at(glass);
        for (std::size_t l = 0; l < um.size(); ++l) {
            const double n_air = mko::ciddor_air_index(um[l], 20.0, 1.0);
            // Raytatouille (CatalogMaterial, Umgebung = Bezugsluft): n_abs/n_U
            const double n_abs = g.index(1000.0 * um[l] / n_air) * n_air;
            const double rtt_relative = n_abs / n_air;
            worst_new =
                std::max(worst_new, std::abs(g.index(s.spectrum.wavelengths_nm[l]) - rtt_relative));
            worst_old = std::max(worst_old,
                                 std::abs(g.index(old.spectrum.wavelengths_nm[l]) - rtt_relative));
        }
    }
    EXPECT_LE(worst_new, mko::test::kRttAirConversion);
    EXPECT_GT(worst_old, 1e-6); // bisheriger Restfehler
    RecordProperty("abweichung_neu", sci(worst_new));
    RecordProperty("abweichung_alt", sci(worst_old));
}

// Import: Wellenlängen und CONST über die Umgebung; Vakuum ändert nichts; Temperatur und Druck
// gehen in die Luft ein; Katalogglas außerhalb der Bezugsluft und CONST bei mehreren
// Wellenlängen geben eine Warnung.
TEST(RttAir, ImportUsesEnvironment) {
    // Vakuum: Zahlen unverändert, keine Warnung (CanonicalVacuumSystemValues)
    const mko::SystemDescription v = import(vacuum_system());
    EXPECT_EQ(v.spectrum.wavelengths_nm[1], 587.6);
    EXPECT_EQ(v.surfaces[1].material.n, 1.7);
    // dasselbe System in Luft bei 30 °C und 0,9 atm
    json warm = vacuum_system();
    warm["environment"] = {{"temperature_c", 30.0}, {"pressure_atm", 0.9}};
    std::vector<std::string> w;
    const mko::SystemDescription a = import(warm, basis_import(), &w);
    const double n_ref = mko::ciddor_air_index(0.5876, 30.0, 0.9);
    EXPECT_EQ(a.spectrum.wavelengths_nm[0],
              0.4861 * 1000.0 / mko::ciddor_air_index(0.4861, 30.0, 0.9));
    EXPECT_EQ(a.spectrum.reference_nm, 0.5876 * 1000.0 / n_ref);
    EXPECT_EQ(a.surfaces[1].material.n, 1.7 / n_ref);
    EXPECT_EQ(a.surfaces[3].material.n, 1.62 / n_ref);
    EXPECT_TRUE(has_warning(w, "Dispersion der Luft"));
    EXPECT_FALSE(has_warning(w, "Katalogglas"));
    // Katalogglas: in der Bezugsluft keine Warnung, bei 30 °C bzw. im Vakuum eine
    w.clear();
    (void)import(catalog_system(), basis_import(), &w);
    EXPECT_TRUE(w.empty()) << w.front();
    json cat_warm = catalog_system();
    cat_warm["environment"] = {{"temperature_c", 30.0}};
    (void)import(cat_warm, basis_import(), &w);
    EXPECT_TRUE(has_warning(w, "Katalogglas: Raytatouille bezieht Katalogdaten auf Luft"));
    w.clear();
    json cat_vac = catalog_system();
    cat_vac["environment"] = {{"medium", "VACUUM"}};
    (void)import(cat_vac, basis_import(), &w);
    EXPECT_TRUE(has_warning(w, "die Umgebung ist Vakuum"));
    // ungültige Umgebung
    json bad = vacuum_system();
    bad["environment"] = {{"pressure_atm", -1.0}};
    EXPECT_TRUE(has_issue(import_issues(bad), "/environment/pressure_atm", "Druck"));
}

// Export: Umgebung Luft (Standard), Wellenlängen und CONST absolut, bitgleich zurück; Vakuum und
// --no-air-conversion schreiben die Zahlen unverändert.
TEST(RttAir, ExportConvertsAndRoundtrips) {
    const mko::SystemDescription v = import(vacuum_system());
    std::vector<std::string> w;
    const json air = json::parse(mko::system_to_rtt_json(v, basis_export(), &w));
    EXPECT_FALSE(air.contains("environment"));
    const double um = air["wavelengths"][1]["um"].get<double>();
    EXPECT_EQ(um * 1000.0 / mko::ciddor_air_index(um, 20.0, 1.0), 587.6);
    const std::string c = air["root"]["children"][1]["material"].get<std::string>();
    const double n_abs = std::stod(c.substr(6));
    EXPECT_EQ(n_abs / mko::ciddor_air_index(um, 20.0, 1.0), 1.7);
    // zurück: dieselbe Beschreibung
    const mko::SystemDescription back = mko::system_from_rtt_json(air.dump(), basis_import());
    EXPECT_EQ(back.spectrum, v.spectrum);
    for (std::size_t i = 0; i < v.surfaces.size(); ++i) {
        EXPECT_EQ(back.surfaces[i].material.n, v.surfaces[i].material.n) << i;
    }
    // Vakuum: unverändert
    mko::RttExportOptions vac = basis_export();
    vac.environment.vacuum = true;
    const json jv = json::parse(mko::system_to_rtt_json(v, vac));
    EXPECT_EQ(jv["environment"], (json{{"medium", "VACUUM"}}));
    EXPECT_EQ(jv["wavelengths"][1]["um"], 0.5876);
    // Temperatur und Druck werden geschrieben und beim Import benutzt
    mko::RttExportOptions warm = basis_export();
    warm.environment.temperature_c = 25.0;
    warm.environment.pressure_atm = 0.95;
    const std::string tw = mko::system_to_rtt_json(v, warm);
    EXPECT_EQ(json::parse(tw)["environment"],
              (json{{"temperature_c", 25.0}, {"pressure_atm", 0.95}}));
    const mko::SystemDescription bw = mko::system_from_rtt_json(tw, basis_import());
    EXPECT_EQ(bw.spectrum, v.spectrum);
    EXPECT_EQ(bw.surfaces[1].material.n, 1.7);
    // ohne Umrechnung: wie vor 2026-11-19 (ohne Katalogglas VACUUM, Zahlen unverändert)
    mko::RttExportOptions raw = basis_export();
    raw.air_conversion = false;
    EXPECT_EQ(json::parse(mko::system_to_rtt_json(v, raw)), vacuum_system());
}

// --estimate-apertures: fehlende Aperturen aus paraxialer Rand- und Hauptstrahlhöhe (volles
// Feld) mit Zuschlag; Orakel: planparallele Platte (Randstrahl parallel, Hauptstrahl im Glas
// mit dem Winkel u/n).
TEST(RttAir, EstimateAperturesFromParaxialHeights) {
    json s = vacuum_system();
    json& ch = s["root"]["children"];
    ch[1]["surfaces"][0].erase("shape");
    ch[1]["surfaces"][0].erase("aperture");
    ch[1]["surfaces"][1].erase("aperture");
    // ohne Option: Fehler (Standard)
    EXPECT_TRUE(has_issue(import_issues(s), "/root/children/1/surfaces/0", "ohne Apertur"));
    mko::RttImportOptions o = basis_import();
    const double margin = 0.1;
    o.aperture_estimate = [margin](const mko::SystemDescription& d) {
        return mko::estimate_clear_semi_diameters(d, basis(), margin);
    };
    std::vector<std::string> w;
    const mko::SystemDescription m = import(s, o, &w);
    EXPECT_TRUE(has_warning(w, "geschätzt"));
    // Platte n = 1,7 (Vakuum: unverändert), Abstand 4 mm von der Blende, Dicke 3,5 mm;
    // größter Feldwinkel 3°, Pupillenradius 3,125 mm
    const double a = 3.125;
    const double u = std::tan(3.0 * std::acos(-1.0) / 180.0);
    const double front = (a + 4.0 * u) * (1.0 + margin);
    const double back = (a + (4.0 + 3.5 / 1.7) * u) * (1.0 + margin);
    EXPECT_NEAR(m.surfaces[1].semi_diameter, front, mko::test::kParaxialRelative * front);
    EXPECT_NEAR(m.surfaces[2].semi_diameter, back, mko::test::kParaxialRelative * back);
    EXPECT_EQ(m.surfaces[3].semi_diameter, 5.5); // vorhandene Aperturen bleiben
    EXPECT_THROW((void)mko::estimate_clear_semi_diameters(m, basis(), -0.1), std::invalid_argument);
}

// ---- Mikroptim → Raytatouille → Mikroptim -------------------------------------------------

namespace {

struct LoadedCatalogs {
    mko::GlassCatalog merged;
    std::vector<std::pair<std::string, mko::GlassCatalog>> files;
    std::string name_of(std::string_view g) const {
        for (const auto& [n, c] : files) {
            if (c.contains(g)) {
                return n;
            }
        }
        return {};
    }
};

LoadedCatalogs load(const mko::SystemDescription& s, const fs::path& dir) {
    LoadedCatalogs l;
    for (const fs::path& f : mko::resolve_glass_catalogs(s, dir)) {
        std::string stem = f.stem().string();
        std::transform(stem.begin(), stem.end(), stem.begin(), [](char c) {
            return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        });
        l.files.emplace_back(stem, mko::GlassCatalog::from_file(f));
        l.merged.merge(l.files.back().second);
    }
    return l;
}

} // namespace

TEST(RttRoundtrip, MikroptimSystemsToRttAndBack) {
    const fs::path dir = fs::path(kRoot) / "data" / "systems";
    std::vector<std::string> converted;
    std::vector<std::string> rejected;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".json") {
            continue;
        }
        const std::string file = entry.path().filename().string();
        SCOPED_TRACE(file);
        const mko::SystemDescription s = mko::read_system_file(entry.path());
        const LoadedCatalogs cat = load(s, dir);
        mko::RttExportOptions eo;
        eo.glass_catalog_name = [&cat](std::string_view g) { return cat.name_of(g); };
        std::vector<std::string> w;
        std::string text;
        try {
            text = mko::system_to_rtt_json(s, eo, &w);
        } catch (const mko::RttConversionError& e) {
            EXPECT_FALSE(e.issues().empty());
            rejected.push_back(file);
            continue;
        }
        converted.push_back(file);
        EXPECT_EQ(has_warning(w, "tolerances: Toleranzsatz entfällt"), s.tolerances.has_value());
        mko::RttImportOptions io;
        io.glass_catalogs = s.glass_catalogs;
        io.glass_exists = [&cat](std::string_view g) { return cat.merged.contains(g); };
        io.glass_catalog_name = [&cat](std::string_view g) { return cat.name_of(g); };
        std::vector<std::string> wi;
        const mko::SystemDescription b = mko::system_from_rtt_json(text, io, &wi);
        for (const std::string& x : wi) {
            // einzige erwartete Warnung: CONST neben Katalogglas in Luft
            EXPECT_NE(x.find("Ciddor"), std::string::npos) << x;
        }

        EXPECT_EQ(b.name, s.name);
        EXPECT_EQ(b.glass_catalogs, s.glass_catalogs);
        EXPECT_EQ(b.source, s.source);
        const std::size_t nl = s.spectrum.wavelengths_nm.size();
        ASSERT_GE(b.spectrum.wavelengths_nm.size(), nl);
        EXPECT_EQ(b.spectrum.reference_nm, s.spectrum.reference_nm);
        for (std::size_t l = 0; l < nl; ++l) {
            EXPECT_EQ(b.spectrum.wavelengths_nm[l], s.spectrum.wavelengths_nm[l]);
            EXPECT_EQ(b.spectrum.weights[l], s.spectrum.weights[l]);
        }
        if (b.spectrum.wavelengths_nm.size() > nl) { // Referenz angehängt
            EXPECT_TRUE(has_warning(w, "Referenzwellenlänge"));
            EXPECT_EQ(b.spectrum.weights.back(), 0.0);
        }
        ASSERT_EQ(b.surfaces.size(), s.surfaces.size());
        const bool ulp_warning = has_warning(w, "1/R weicht");
        for (std::size_t i = 0; i < s.surfaces.size(); ++i) {
            const mko::Surface& x = s.surfaces[i];
            const mko::Surface& y = b.surfaces[i];
            const bool flat_sphere = x.type == mko::SurfaceType::Sphere && x.curvature == 0.0;
            EXPECT_EQ(y.type, flat_sphere ? mko::SurfaceType::Plane : x.type) << i;
            if (ulp_warning) {
                EXPECT_LE(std::abs(y.curvature - x.curvature),
                          std::abs(std::nextafter(x.curvature, 0.0) - x.curvature))
                    << i;
            } else {
                EXPECT_EQ(y.curvature, x.curvature) << i;
            }
            EXPECT_EQ(y.thickness, x.thickness) << i;
            EXPECT_EQ(y.semi_diameter, x.semi_diameter) << i;
            EXPECT_EQ(y.material.kind, x.material.kind) << i;
            EXPECT_EQ(y.material.name, x.material.name) << i;
            EXPECT_EQ(y.material.n, x.material.n) << i;
            EXPECT_EQ(y.decenter, x.decenter);
            EXPECT_EQ(y.tilt_deg, x.tilt_deg);
        }
        EXPECT_TRUE(b.ghost_plates.empty());
        const mko::ParaxialData pa = mko::compute_paraxial(s, cat.merged);
        const mko::ParaxialData pb = mko::compute_paraxial(b, cat.merged);
        if (!std::isfinite(pa.focal_length)) { // afokal (Planplatte)
            EXPECT_EQ(pb.afocal, pa.afocal);
            EXPECT_EQ(std::isnan(pb.back_focal_distance), std::isnan(pa.back_focal_distance));
        } else if (!ulp_warning && s.ghost_plates.empty()) {
            EXPECT_EQ(pb.focal_length, pa.focal_length);
            EXPECT_EQ(pb.back_focal_distance, pa.back_focal_distance);
        } else {
            EXPECT_NEAR(pb.focal_length, pa.focal_length,
                        mko::test::kParaxialRelative * std::abs(pa.focal_length));
            EXPECT_NEAR(pb.back_focal_distance, pa.back_focal_distance,
                        mko::test::kParaxialRelative * std::abs(pa.back_focal_distance));
        }
    }
    for (const char* f : {"einzellinse_bk7.json", "planplatte_bk7.json", "zweilinser_petzval.json",
                          "zweilinser_mechanik.json", "einzellinse_bk7_biegung.json"}) {
        EXPECT_NE(std::find(converted.begin(), converted.end(), f), converted.end()) << f;
    }
    for (const char* f : {"testsystem_mehrlinser_dezentriert.json",
                          "einzellinse_geisterplatte.json", "achromat_glasvariablen.json"}) {
        EXPECT_NE(std::find(rejected.begin(), rejected.end(), f), rejected.end()) << f;
    }
}

TEST(RttExport, WarningsForDroppedData) {
    mko::SystemDescription s =
        mko::read_system_file(kRoot + "data/systems/einzellinse_bk7_nebenbedingungen.json");
    s.spectrum.reference_nm = 546.07; // nicht in der Liste
    s.surfaces[2].type = mko::SurfaceType::Sphere;
    s.surfaces[2].curvature = 0.0;
    s.surfaces[2].comment = "Rückfläche mit Leerzeichen";
    s.surfaces[1].group = "front"; // Feld group (Entscheidung Maintainer 2026-11-19)
    std::vector<std::string> w;
    const json r = json::parse(mko::system_to_rtt_json(s, basis_export(), &w));
    for (const char* text : {"description:", "target:", "variables:", "constraints:",
                             "spectrum.reference_nm:", "surfaces[2]: Kugel mit Krümmung 0",
                             "surfaces[2].comment:", "surfaces[1].group: Gruppe entfällt"}) {
        EXPECT_TRUE(has_warning(w, text)) << text;
    }
    const json& ref = r["wavelengths"].back();
    EXPECT_EQ(ref["weight"], 0.0);
    EXPECT_EQ(ref["reference"], true);
    const double ref_um = ref["um"].get<double>();
    EXPECT_EQ(ref_um * 1000.0 / mko::ciddor_air_index(ref_um, 20.0, 1.0), 546.07);
    EXPECT_FALSE(r.contains("environment")); // Katalogglas: Umgebung Luft
    EXPECT_EQ(r["root"]["children"][1]["material"], "BASIS:N-BK7");
}

TEST(RttExport, NonMappableElementsAreErrors) {
    const mko::SystemDescription base =
        mko::read_system_file(kRoot + "data/systems/einzellinse_bk7.json");
    struct Case {
        const char* location;
        const char* text;
        void (*mutate)(mko::SystemDescription&);
    };
    const Case cases[] = {
        {"surfaces[1].decenter", "Dezentrierung",
         [](mko::SystemDescription& s) { s.surfaces[1].decenter = {0.01, 0.0}; }},
        {"surfaces[2].tilt_deg", "Verkippung",
         [](mko::SystemDescription& s) { s.surfaces[2].tilt_deg = {0.0, 0.1}; }},
        {"surfaces[1].material", "Buchdahl",
         [](mko::SystemDescription& s) {
             s.surfaces[1].material.kind = mko::MaterialKind::Buchdahl;
             s.surfaces[1].material.n0 = 1.6;
         }},
        {"source.type", "Punktquelle",
         [](mko::SystemDescription& s) {
             s.source.type = mko::SourceType::Point;
             s.source.field_angles_deg.clear();
             s.source.field_weights.clear();
             s.source.numerical_aperture = 0.1;
             s.source.point = {0.0, 0.0, -50.0};
         }},
        {"surfaces[2].material", "Bildraum",
         [](mko::SystemDescription& s) {
             s.surfaces[2].material.kind = mko::MaterialKind::Constant;
             s.surfaces[2].material.n = 1.33;
         }},
        {"ghost_plates[0]", "Geisterplatte",
         [](mko::SystemDescription& s) {
             mko::GhostPlate g;
             g.after_surface = 0;
             g.distance = 1.0;
             g.glass.kind = mko::MaterialKind::Catalog;
             g.glass.name = "CAF2";
             g.norm_radius = 10.0;
             g.coefficients = {0.0, 1e-4};
             s.ghost_plates.push_back(g);
         }},
    };
    for (const Case& c : cases) {
        mko::SystemDescription s = base;
        c.mutate(s);
        try {
            (void)mko::system_to_rtt_json(s, basis_export());
            ADD_FAILURE() << c.location;
        } catch (const mko::RttConversionError& e) {
            EXPECT_TRUE(has_issue(e.issues(), c.location, c.text)) << c.location << "\n"
                                                                   << e.what();
        }
    }
    // Katalogglas ohne Katalognamen
    try {
        (void)mko::system_to_rtt_json(base, {});
        ADD_FAILURE();
    } catch (const mko::RttConversionError& e) {
        EXPECT_TRUE(has_issue(e.issues(), "surfaces[1].material", "Katalog des Glases N-BK7"));
    }
    // Geisterplatten ohne Wirkung entfallen mit Warnung
    mko::SystemDescription zero = base;
    mko::GhostPlate g;
    g.after_surface = 0;
    g.distance = 1.0;
    g.glass.kind = mko::MaterialKind::Catalog;
    g.glass.name = "CAF2";
    g.norm_radius = 10.0;
    g.coefficients = {0.0, 0.0};
    zero.ghost_plates.push_back(g);
    std::vector<std::string> w;
    (void)mko::system_to_rtt_json(zero, basis_export(), &w);
    EXPECT_TRUE(has_warning(w, "ghost_plates: Geisterplatten ohne Wirkung"));
}

// ---- nicht abbildbare Raytatouille-Elemente ------------------------------------------------

TEST(RttImport, NonMappableElementsAreErrors) {
    struct Case {
        const char* location;
        const char* text;
        void (*mutate)(json&);
    };
    const Case cases[] = {
        {"/root/children/1/surfaces/0/shape/base/conic", "Konik",
         [](json& s) { s["root"]["children"][1]["surfaces"][0]["shape"]["base"]["conic"] = -1.0; }},
        {"/root/children/1/surfaces/0/shape/base/type", "Flächenform",
         [](json& s) {
             s["root"]["children"][1]["surfaces"][0]["shape"]["base"] = {
                 {"type", "even_asphere"}, {"radius", 41.3}, {"coefficients", {1e-6}}};
         }},
        {"/root/children/1/surfaces/0/shape/terms", "Zernike",
         [](json& s) {
             s["root"]["children"][1]["surfaces"][0]["shape"]["terms"] = json::array({{{"n", 2}}});
         }},
        {"/root/children/2/type", "Spiegel",
         [](json& s) { s["root"]["children"][2]["type"] = "mirror"; }},
        {"/object", "endliches Objekt",
         [](json& s) { s["object"] = {{"at_infinity", false}, {"distance", 100.0}}; }},
        {"/aperture/type", "Systemapertur",
         [](json& s) { s["aperture"] = {{"type", "image_fnumber"}, {"value", 4.0}}; }},
        {"/aperture/stop", "Blende auf einer Linsenfläche",
         [](json& s) { s["aperture"]["stop"] = {{"surface", "L1.S1"}}; }},
        {"/fields/points/1/x", "Feldwinkel in x",
         [](json& s) { s["fields"]["points"][1]["x"] = 1.0; }},
        {"/fields/type", "Feldart", [](json& s) { s["fields"]["type"] = "object_height"; }},
        {"/root/children/1/surfaces/1/pose/position/1", "Dezentrierung",
         [](json& s) { s["root"]["children"][1]["surfaces"][1]["pose"]["position"][1] = 0.02; }},
        {"/root/children/2/pose/rotation_deg/0", "Verkippung",
         [](json& s) { s["root"]["children"][2]["pose"]["rotation_deg"] = {0.1, 0.0, 0.0}; }},
        {"/environment/medium", "Umgebung",
         [](json& s) { s["environment"]["medium"] = "SCHOTT:N-BK7"; }},
        {"/root/children/1/material", "komplexe Brechzahl",
         [](json& s) { s["root"]["children"][1]["material"] = "CONST:1.2,7.26"; }},
        {"/root/children/1/material", "Kristalle",
         [](json& s) {
             s["root"]["children"][1]["material"] = {{"ordinary", "BIREFRINGENT:CALCITE"},
                                                     {"extraordinary", "BIREFRINGENT:CALCITE-E"}};
         }},
        {"/root/children/1/material", "nicht in den Mikroptim-Glaskatalogen",
         [](json& s) { s["root"]["children"][1]["material"] = "SCHOTT:N-SF66"; }},
        {"/root/children/2/material/0", "AIR in der Umgebung VACUUM",
         [](json& s) { s["root"]["children"][2]["material"][0] = "AIR"; }},
        {"/paths", "kein Pfad",
         [](json& s) {
             s["paths"] = json::array(
                 {{{"name", "main"}, {"events", json::array({{{"surface", "L1.S1"}}})}}});
         }},
        {"/root/children/0/surfaces/0", "keine Blende",
         [](json& s) { s["root"]["children"].erase(0); }},
        {"/root", "Bildfläche (detector) fehlt", [](json& s) { s["root"]["children"].erase(4); }},
        {"/root/children/4/surfaces/0", "hinter der Bildfläche",
         [](json& s) {
             json& ch = s["root"]["children"];
             std::swap(ch[3], ch[4]);
         }},
        {"/root/children/1/surfaces/0/phases", "Phasenschichten",
         [](json& s) {
             s["root"]["children"][1]["surfaces"][0]["phases"] =
                 json::array({{{"type", "linear_grating"}, {"lines_per_mm", 100.0}}});
         }},
        {"/root/children/3/surfaces/0/interaction", "Interaktion",
         [](json& s) {
             s["root"]["children"][3]["surfaces"][0]["interaction"] = {{"type", "ideal_lens"},
                                                                       {"focal_length", 50.0}};
         }},
        {"/root/children/1/surfaces/0/aperture", "Apertur",
         [](json& s) {
             s["root"]["children"][1]["surfaces"][0]["aperture"] = {
                 {"type", "rectangular"}, {"half_width_x", 5.0}, {"half_width_y", 5.0}};
         }},
        {"/root/children/1/surfaces/0/aperture/inner_radius", "Ringapertur",
         [](json& s) {
             s["root"]["children"][1]["surfaces"][0]["aperture"]["inner_radius"] = 1.0;
         }},
        {"/root/children/1/surfaces/0/obscurations", "Obskurationen",
         [](json& s) {
             s["root"]["children"][1]["surfaces"][0]["obscurations"] =
                 json::array({{{"type", "circular"}, {"radius", 1.0}}});
         }},
        {"/root/children/1/surfaces/0", "ohne Apertur",
         [](json& s) { s["root"]["children"][1]["surfaces"][0].erase("aperture"); }},
        {"/root/children/1/pose/position/2", "Ausdruck",
         [](json& s) {
             s["parameters"] = json::array({{{"name", "T"}, {"expression", "1 + 3"}}});
             s["root"]["children"][1]["pose"]["position"][2] = {{"param", "T"}};
         }},
        {"/root/children/1/pose/position/2", "unbekannter Parameter",
         [](json& s) { s["root"]["children"][1]["pose"]["position"][2] = {{"param", "X"}}; }},
        {"/schema_version", "nicht unterstützt", [](json& s) { s["schema_version"] = "0.12.0"; }},
        {"/units", "Einheiten", [](json& s) { s["units"]["wavelength"] = "nm"; }},
        {"/wavelengths", "genau eine Wellenlänge",
         [](json& s) { s["wavelengths"][0]["reference"] = true; }},
        {"/root/children/1/surfaces", "mindestens zwei Flächen",
         [](json& s) { s["root"]["children"][1]["surfaces"].erase(1); }},
        {"/root/children/1/surfaces/0/pose", "erste Fläche eines Elements",
         [](json& s) { s["root"]["children"][1]["surfaces"][0]["pose"] = rel(1.0); }},
    };
    for (const Case& c : cases) {
        json s = vacuum_system();
        c.mutate(s);
        const std::vector<mko::RttIssue> issues = import_issues(s);
        EXPECT_TRUE(has_issue(issues, c.location, c.text)) << c.location << " / " << c.text << "\n"
                                                           << issues_text(issues);
    }
}

TEST(RttImport, AllIssuesAreReportedTogether) {
    json s = vacuum_system();
    s["root"]["children"][1]["surfaces"][0]["shape"]["base"]["conic"] = -0.5;
    s["root"]["children"][2]["pose"]["rotation_deg"] = {0.0, 0.2, 0.0};
    s["object"] = {{"at_infinity", false}, {"distance", 50.0}};
    try {
        (void)import(s);
        FAIL();
    } catch (const mko::RttConversionError& e) {
        EXPECT_EQ(e.issues().size(), 3u) << e.what();
        const std::string what = e.what();
        EXPECT_NE(what.find("nicht abbildbar (3 Stellen)"), std::string::npos) << what;
        EXPECT_NE(what.find("/root/children/1/surfaces/0/shape/base/conic"), std::string::npos);
        EXPECT_NE(what.find("/root/children/2/pose/rotation_deg/1"), std::string::npos);
        EXPECT_NE(what.find("/object"), std::string::npos);
    }
}

TEST(RttImport, WarningsForDroppedData) {
    json s = vacuum_system();
    s["optimization"] = {{"operands", json::array({{{"type", "efl"}, {"target", 50.0}}})}};
    s["tolerances"] = {{"class", "normal"}};
    s["paths"].push_back({{"name", "ghost"}, {"events", json::array({{{"surface", "L1.S1"}}})}});
    s["environment"]["temperature_c"] = 25.0;
    s["root"]["children"][1]["surfaces"][0]["interaction"] = {{"type", "coating"},
                                                              {"name", "C:AR"}};
    s["root"]["children"][4]["surfaces"][0]["aperture"] = circ(1.0);
    json inactive = s["root"]["children"][3];
    inactive["name"] = "Filter";
    inactive["active"] = false;
    inactive["type"] = "mirror"; // inaktiv: kein Fehler
    inactive["pose"] = rel(0.0);
    s["root"]["children"].insert(s["root"]["children"].begin() + 4, inactive);
    s["root"]["children"][1]["surfaces"][1].erase("aperture");
    mko::RttImportOptions o = basis_import();
    o.default_semi_diameter = 6.0;
    o.air_conversion = false; // Temperatur entfällt nur ohne Umrechnung (sonst RttAir.*)
    std::vector<std::string> w;
    const mko::SystemDescription m = import(s, o, &w);
    for (const char* text :
         {"/optimization:", "/tolerances:", "/paths: übernommen wird nur",
          "/environment: Temperatur", "/interaction: Interaktion coating",
          "/root/children/5/surfaces/0/aperture:", "/root/children/4: inaktiver Knoten",
          "/root/children/1/surfaces/1: ohne Apertur"}) {
        EXPECT_TRUE(has_warning(w, text)) << text;
    }
    EXPECT_EQ(m, import(vacuum_system()));
    // CONST in Luft: Hinweis auf die absolute Rechnung in Raytatouille
    json air = vacuum_system();
    air.erase("environment");
    w.clear();
    (void)import(air, basis_import(), &w);
    EXPECT_TRUE(has_warning(w, "Ciddor"));
    // Katalogglas in VACUUM (ohne Umrechnung; mit Umrechnung RttAir.ImportUsesEnvironment)
    json vac = catalog_system();
    vac["environment"] = {{"medium", "VACUUM"}};
    w.clear();
    mko::RttImportOptions raw = basis_import();
    raw.air_conversion = false;
    (void)import(vac, raw, &w);
    EXPECT_TRUE(has_warning(w, "Katalogglas in der Umgebung VACUUM"));
    // Katalogglas aus einem anderen Katalog als dem von Mikroptim
    json other = catalog_system();
    other["root"]["children"][1]["material"] = "SCHOTT:N-BK7";
    w.clear();
    (void)import(other, basis_import(), &w);
    EXPECT_TRUE(has_warning(w, "aus Katalog SCHOTT wird aus dem Mikroptim-Katalog BASIS"));
    // ohne Glaskataloge ist ein Katalogglas ein Fehler
    EXPECT_TRUE(has_issue(import_issues(catalog_system(), {}), "/root/children/1/material",
                          "keine Mikroptim-Glaskataloge"));
}

// ---- Kommandozeile ------------------------------------------------------------------------

namespace {

struct RunResult {
    int code = -1;
    std::string out;
    std::string err;
};

RunResult run_convert(const std::vector<std::string>& args) {
    const std::vector<std::string_view> v(args.begin(), args.end());
    std::ostringstream out;
    std::ostringstream err;
    RunResult r;
    r.code = mko::cli::run_convert_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

} // namespace

TEST(RttCli, ConvertBothDirections) {
    const fs::path dir = mko::test::temp_dir() / "rtt_cli";
    fs::create_directories(dir);
    const std::string input = kRoot + "data/systems/einzellinse_bk7.json";
    const fs::path rtt = dir / "einzellinse.rtt.json";
    RunResult r = run_convert({input, "-o", rtt.string()});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Mikroptim → Raytatouille"), std::string::npos);
    EXPECT_NE(r.err.find("Warnung: description"), std::string::npos) << r.err;
    const json j = json::parse(read_file(rtt));
    EXPECT_EQ(j["root"]["children"][1]["material"], "BASIS:N-BK7");

    const fs::path back = dir / "einzellinse.json";
    r = run_convert({rtt.string(), "-o", back.string(), "--glass-catalog", kBasis});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Raytatouille → Mikroptim"), std::string::npos);
    EXPECT_TRUE(r.err.empty()) << r.err;
    const mko::SystemDescription a = mko::read_system_file(input);
    const mko::SystemDescription b = mko::read_system_file(back); // Katalog wird gefunden
    ASSERT_EQ(b.surfaces.size(), a.surfaces.size());
    for (std::size_t i = 0; i < a.surfaces.size(); ++i) {
        EXPECT_EQ(b.surfaces[i].curvature, a.surfaces[i].curvature);
        EXPECT_EQ(b.surfaces[i].thickness, a.surfaces[i].thickness);
    }
    const mko::GlassCatalog cat =
        mko::GlassCatalog::from_file(mko::resolve_glass_catalogs(b, back.parent_path()).at(0));
    EXPECT_EQ(mko::compute_paraxial(b, cat).focal_length,
              mko::compute_paraxial(a, basis()).focal_length);

    // --rtt-catalog setzt den Katalognamen
    r = run_convert({input, "-o", rtt.string(), "--rtt-catalog", "SCHOTT"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_EQ(json::parse(read_file(rtt))["root"]["children"][1]["material"], "SCHOTT:N-BK7");
}

TEST(RttCli, ErrorsAndUsage) {
    const fs::path dir = mko::test::temp_dir() / "rtt_cli_errors";
    fs::create_directories(dir);
    const std::string dezentriert = kRoot + "data/systems/testsystem_mehrlinser_dezentriert.json";
    RunResult r = run_convert({dezentriert, "-o", (dir / "x.rtt.json").string()});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("nicht abbildbar"), std::string::npos) << r.err;
    EXPECT_NE(r.err.find("surfaces[1].decenter"), std::string::npos) << r.err;
    EXPECT_NE(r.err.find("surfaces[3].tilt_deg"), std::string::npos) << r.err;
    EXPECT_FALSE(fs::exists(dir / "x.rtt.json"));

    const std::string input = kRoot + "data/systems/einzellinse_bk7.json";
    EXPECT_EQ(run_convert({input}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run_convert({input, "-o", input}).code, mko::cli::kExitUsage);
    EXPECT_EQ(
        run_convert({input, "-o", (dir / "y.rtt.json").string(), "--glass-catalog", kBasis}).code,
        mko::cli::kExitUsage);
    EXPECT_EQ(run_convert({input, "-o", "z", "--bogus"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run_convert({"--help"}).code, mko::cli::kExitOk);
    // Optionen der Umrechnung und der Aperturschätzung (Entscheidung Maintainer 2026-11-19)
    const std::string y = (dir / "y.rtt.json").string();
    EXPECT_EQ(run_convert({input, "-o", y, "--rtt-environment", "mars"}).code,
              mko::cli::kExitUsage);
    EXPECT_EQ(run_convert({input, "-o", y, "--estimate-apertures"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(
        run_convert({input, "-o", y, "--rtt-environment", "vacuum", "--no-air-conversion"}).code,
        mko::cli::kExitUsage);
    EXPECT_EQ(run_convert({input, "-o", y, "--aperture-margin", "0.2"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run_convert({input, "-o", y, "--estimate-apertures", "--semi-diameter", "3"}).code,
              mko::cli::kExitUsage);

    // Raytatouille-Datei mit Katalogglas, aber ohne --glass-catalog
    const fs::path rtt = dir / "k.rtt.json";
    std::ofstream(rtt, std::ios::binary) << catalog_system().dump();
    r = run_convert({rtt.string(), "-o", (dir / "k.json").string()});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("--glass-catalog"), std::string::npos) << r.err;

    // weder Raytatouille noch Mikroptim
    const fs::path other = dir / "other.json";
    std::ofstream(other, std::ios::binary) << R"({"format": "mikroptim.glass/1"})";
    EXPECT_EQ(run_convert({other.string(), "-o", (dir / "o.json").string()}).code,
              mko::cli::kExitError);
}

// Kommandozeile der Umrechnung (Entscheidung Maintainer 2026-11-19): --estimate-apertures mit
// --aperture-margin, --rtt-environment vacuum, --no-air-conversion.
TEST(RttCli, AirConversionAndApertureOptions) {
    const fs::path dir = mko::test::temp_dir() / "rtt_cli_air";
    fs::create_directories(dir);
    json s = vacuum_system();
    s["root"]["children"][1]["surfaces"][1].erase("aperture");
    const fs::path rtt = dir / "ohne_apertur.rtt.json";
    std::ofstream(rtt, std::ios::binary) << s.dump();
    const fs::path out = dir / "m.json";
    RunResult r = run_convert({rtt.string(), "-o", out.string()});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("--estimate-apertures"), std::string::npos) << r.err;
    r = run_convert(
        {rtt.string(), "-o", out.string(), "--estimate-apertures", "--aperture-margin", "0.25"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.err.find("geschätzt"), std::string::npos) << r.err;
    const mko::SystemDescription m = mko::read_system_file(out);
    const std::vector<double> h = mko::estimate_clear_semi_diameters(m, basis(), 0.25);
    EXPECT_EQ(m.surfaces[2].semi_diameter, h[2]);
    // zurück nach Raytatouille in der Umgebung VACUUM: Zahlen unverändert
    const fs::path back = dir / "zurueck.rtt.json";
    r = run_convert({out.string(), "-o", back.string(), "--rtt-environment", "vacuum"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json jb = json::parse(read_file(back));
    EXPECT_EQ(jb["environment"], (json{{"medium", "VACUUM"}}));
    EXPECT_EQ(jb["wavelengths"], s["wavelengths"]);
    // Standard Luft: Vakuumwellenlängen; ohne Umrechnung wie früher (VACUUM ohne Katalogglas)
    r = run_convert({out.string(), "-o", back.string()});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_FALSE(json::parse(read_file(back)).contains("environment"));
    EXPECT_GT(json::parse(read_file(back))["wavelengths"][1]["um"].get<double>(), 0.5876);
    r = run_convert({out.string(), "-o", back.string(), "--no-air-conversion"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_EQ(json::parse(read_file(back))["wavelengths"], s["wavelengths"]);
    EXPECT_EQ(json::parse(read_file(back))["environment"], (json{{"medium", "VACUUM"}}));
    // Import einer Luft-Datei ohne Umrechnung: Wellenlängen unverändert
    json air = vacuum_system();
    air.erase("environment");
    std::ofstream(rtt, std::ios::binary) << air.dump();
    r = run_convert({rtt.string(), "-o", out.string(), "--no-air-conversion"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_EQ(mko::read_system_file(out).spectrum.reference_nm, 587.6);
    EXPECT_EQ(run_convert({rtt.string(), "-o", out.string(), "--rtt-environment", "air"}).code,
              mko::cli::kExitUsage);
}
