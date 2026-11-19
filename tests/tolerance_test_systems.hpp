#pragma once

// Testsysteme und Hilfen der Monte-Carlo-Tolerierung (M8-T4): das Testsystem
// data/systems/testsystem_toleranz.json (frei erfunden), Varianten mit anderem Toleranzsatz und
// ein Kittglied mit planparalleler Platte im konvergenten Strahl (Orakel des Kompensators).

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace mko::test {

inline const std::string kToleranceSystemFile =
    std::string(MKO_SOURCE_DIR) + "/data/systems/testsystem_toleranz.json";

// JSON des Testsystems.
inline nlohmann::json tolerance_system_json() {
    std::ifstream in(kToleranceSystemFile);
    std::stringstream ss;
    ss << in.rdbuf();
    return nlohmann::json::parse(ss.str());
}

// Lädt ein System aus JSON mit den Glaskatalogen relativ zum Verzeichnis des Testsystems.
inline LoadedSystem load_tolerance_json(const nlohmann::json& j) {
    return prepare_system(read_system_json(j.dump()),
                          std::filesystem::path(kToleranceSystemFile).parent_path());
}

// Testsystem mit eigener Liste `tolerances.items` (übrige Felder des Abschnitts bleiben).
inline nlohmann::json with_tolerance_items(nlohmann::json items) {
    nlohmann::json j = tolerance_system_json();
    j["tolerances"]["items"] = std::move(items);
    return j;
}

// Kittglied des Testsystems, danach 4 mm Luft, eine planparallele Platte aus Quarzglas (1 mm)
// und der Arbeitsabstand. Eine Platte ist längs z verschiebungsinvariant: Ein um δ längerer
// Luftspalt vor der Platte wird durch einen um δ kürzeren Arbeitsabstand exakt ausgeglichen
// (gleiche Strahlen, gleiche OPD bis auf Rundung). Toleranzsatz: nur dieser Luftspalt
// (Fläche 3).
inline nlohmann::json plate_system_json() {
    nlohmann::json j = tolerance_system_json();
    nlohmann::json& s = j["surfaces"];
    const double wd = s[3]["thickness"].get<double>();
    s[3]["thickness"] = 4.0;
    s.push_back(
        {{"type", "plane"}, {"thickness", 1.0}, {"material", "F_SILICA"}, {"semi_diameter", 3.0}});
    // Fokus hinter der Platte: Arbeitsabstand − 4 mm − Plattenversatz t (1 − 1/n) ≈ 0,32 mm
    s.push_back({{"type", "plane"},
                 {"thickness", wd - 4.0 - 1.0 + 0.32},
                 {"material", "AIR"},
                 {"semi_diameter", 3.0}});
    j["name"] = "Testsystem Toleranz mit Platte";
    j["tolerances"] = {{"distribution", "normal"},
                       {"cutoff_sigma", 2.0},
                       {"compensator", "working_distance"},
                       {"items", nlohmann::json::array(
                                     {{{"type", "thickness"}, {"surface", 3}, {"delta", 0.05}}})}};
    return j;
}

} // namespace mko::test
