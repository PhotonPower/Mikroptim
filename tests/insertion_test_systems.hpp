#pragma once

// Einfüge-Operatoren (M6-T2, CPU und GPU): Merit eines Systems wie in mikroptim optimize und die
// Fälle der Abnahme „Merit nach dem Einfügen relativ unverändert“ (docs/optimierung.md,
// Abschnitt 6).

#include "fd_test_systems.hpp"
#include "mikroptim/model/material.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/insertion.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/trace_extra_residuals.hpp"

#include <cmath>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace mko::test {

inline LoadedSystem prepare_test_system(const SystemDescription& d) {
    return prepare_system(d, std::filesystem::path(kResidualSystems));
}

// Zusatzresiduen wie in mikroptim optimize (ohne Glas-Hülle): paraxiale Nebenbedingungen, die
// L1-Strafe der Geisterkoeffizienten und der geometrische Teil der mechanischen Bedingungen
// (M4-T3); der trace-abhängige Teil steht in insertion_trace_extra.
inline ExtraResiduals insertion_extra(const SystemDescription& d) {
    return concat_extra_residuals(
        {make_paraxial_residuals(make_paraxial_constraints(d)),
         make_ghost_penalty_residuals(make_ghost_penalty(d)),
         make_mechanical_residuals(make_mechanical_constraints(d)).geometric});
}

// Trace-abhängige mechanische Residuen (aus den Extrema: Strahlhöhe, Einfallswinkel, Randdicke
// bei den Strahlhöhen).
inline TraceExtraResiduals insertion_trace_extra(const SystemDescription& d) {
    return make_mechanical_residuals(make_mechanical_constraints(d)).traced;
}

// F = Strahl-, Straf- und Zusatzresiduen (Wellen²) des Systems auf der CPU.
inline double system_merit(const SystemDescription& d, const ResidualSettings& q = {}) {
    const LoadedSystem l = prepare_test_system(d);
    const SystemBatchHost b = make_system_batch(l, 1);
    return make_cpu_merit(q, insertion_extra(l.system), 0, insertion_trace_extra(l.system))(b).at(
        0);
}

inline double relative_change(double before, double after) {
    return std::abs(after - before) / std::abs(before);
}

inline MaterialSpec catalog_glass(const char* name) {
    MaterialSpec m;
    m.kind = MaterialKind::Catalog;
    m.name = name;
    return m;
}

inline MaterialSpec constant_glass(double n) {
    MaterialSpec m;
    m.kind = MaterialKind::Constant;
    m.n = n;
    return m;
}

// Flint der Geisterplatten-Testsysteme (Buchdahl-Modell, n_d = 1,753, ν_d = 26,9).
inline MaterialSpec flint_glass() {
    MaterialSpec m;
    m.kind = MaterialKind::Buchdahl;
    m.n0 = 1.76;
    m.nu = {-0.18, 0.112, 0.0};
    return m;
}

inline MeniscusInsertion meniscus(int after, double position, double c0, MaterialSpec glass) {
    MeniscusInsertion m;
    m.after_surface = after;
    m.position = position;
    m.curvature = c0;
    m.glass = std::move(glass);
    return m;
}

inline LensSplit lens_split(int lens, double cs, double fraction) {
    LensSplit s;
    s.lens_surface = lens;
    s.curvature = cs;
    s.fraction = fraction;
    return s;
}

// Einzellinse mit Hilfsebene hinter der Linse (einzellinse_geisterplatten_exakt.json ohne
// Geisterplatten, Geistervariablen und Strafe), damit hinter der Linse ein Luftspalt liegt.
inline SystemDescription lens_with_aux_plane() {
    SystemDescription d = load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    d.ghost_plates.clear();
    d.variables->items.pop_back();   // ghost_coefficient
    d.constraints->items.pop_back(); // ghost_penalty
    validate_system(d);
    return d;
}

struct InsertionCase {
    std::string name;
    SystemDescription before;
    SystemDescription after;
};

// Fälle der Merit-Invarianz: Operator A an verschiedenen Lagen, Krümmungen und Gläsern (auch an
// Lagen von Geisterplatten mit Koeffizienten ≠ 0, vor, zwischen und hinter ihnen), Operator B mit
// verschiedenen Kittflächen und Dickenaufteilungen; Einzellinse (eine Wellenlänge), Mehrlinser
// (drei Felder, vier Wellenlängen), Systeme mit Geisterplatten, paraxialen Nebenbedingungen und
// L1-Strafe.
inline std::vector<InsertionCase> insertion_cases() {
    std::vector<InsertionCase> out;
    const auto add_a = [&](const std::string& name, const SystemDescription& d,
                           const MeniscusInsertion& m) {
        out.push_back({name, d, insert_meniscus(d, m).system});
    };
    const auto add_b = [&](const std::string& name, const SystemDescription& d,
                           const LensSplit& s) {
        out.push_back({name, d, split_lens(d, s).system});
    };
    const SystemDescription lens = load_residual_system("einzellinse_bk7_biegung.json").system;
    const SystemDescription aux = lens_with_aux_plane();
    const SystemDescription multi = load_residual_system("testsystem_mehrlinser.json").system;
    const SystemDescription exact =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    const SystemDescription na05 =
        load_residual_system("einzellinse_na05_geisterplatten.json").system;
    const SystemDescription mech = load_residual_system("zweilinser_mechanik.json").system;

    add_a("A Einzellinse Blende, Pos. 0, c0 0, N-BK7", lens,
          meniscus(0, 0.0, 0.0, catalog_glass("N-BK7")));
    add_a("A Einzellinse Blende, Pos. 2,5, c0 0,05, CaF2", lens,
          meniscus(0, 2.5, 0.05, catalog_glass("CAF2")));
    add_a("A Einzellinse Blende, Pos. 5 (Spaltende), c0 -0,2, n = 1,9", lens,
          meniscus(0, 5.0, -0.2, constant_glass(1.9)));
    add_a("A Einzellinse hinter Linse, Pos. 15, c0 0,3, Flint", aux,
          meniscus(2, 15.0, 0.3, flint_glass()));
    add_a("A Einzellinse hinter Linse, Pos. 1, c0 -0,01, Flint", aux,
          meniscus(2, 1.0, -0.01, flint_glass()));
    add_a("A Mehrlinser Spalt 3, Pos. 0,5, c0 0,02, N-BK7", multi,
          meniscus(3, 0.5, 0.02, catalog_glass("N-BK7")));
    add_a("A Mehrlinser Spalt 5, Pos. 1,2, c0 -0,08, Flint", multi,
          meniscus(5, 1.2, -0.08, flint_glass()));
    add_a("A Mehrlinser Spalt 0, Pos. 1, c0 0,1, CaF2", multi,
          meniscus(0, 1.0, 0.1, catalog_glass("CAF2")));
    add_a("A Geisterplatten zwischen den Platten (2,5), c0 -0,1, Flint", exact,
          meniscus(0, 2.5, -0.1, flint_glass()));
    add_a("A Geisterplatten an Plattenlage (1,0), c0 0,2, N-BK7", exact,
          meniscus(0, 1.0, 0.2, catalog_glass("N-BK7")));
    add_a("A Geisterplatten hinter Linse vor der Platte (5), c0 0,04, CaF2", exact,
          meniscus(2, 5.0, 0.04, catalog_glass("CAF2")));
    add_a("A Geisterplatten hinter Linse nach der Platte (15), c0 -0,05, Flint", exact,
          meniscus(2, 15.0, -0.05, flint_glass()));
    add_a("A NA 0,5 an der Lage der Platten (0,2), c0 0,6, Flint", na05,
          meniscus(2, 0.2, 0.6, flint_glass()));
    add_a("A NA 0,5 vor den Platten (0,1), c0 -0,4, N-BK7", na05,
          meniscus(2, 0.1, -0.4, catalog_glass("N-BK7")));
    add_a("A Mechanik (M4-T3) Luftspalt 2, Pos. 0,5, c0 -0,04, N-BK7", mech,
          meniscus(2, 0.5, -0.04, catalog_glass("N-BK7")));
    add_a("A Mechanik (M4-T3) vor der ersten Linse, Pos. 0,3, c0 0,02, n = 1,6", mech,
          meniscus(0, 0.3, 0.02, constant_glass(1.6)));

    add_b("B Einzellinse Mitte, c_s 0", lens, lens_split(1, 0.0, 0.5));
    add_b("B Einzellinse 0,25, c_s -0,05", lens, lens_split(1, -0.05, 0.25));
    add_b("B Einzellinse 0 (Kittfläche am Scheitel vorn), c_s 0,00967", lens,
          lens_split(1, 0.00967, 0.0));
    add_b("B Einzellinse 1 (Kittfläche am Scheitel hinten), c_s -0,00967", lens,
          lens_split(1, -0.00967, 1.0));
    add_b("B Mehrlinser CaF2, c_s 0,01", multi, lens_split(1, 0.01, 0.4));
    add_b("B Mehrlinser Buchdahl, c_s -0,03", multi, lens_split(2, -0.03, 0.5));
    add_b("B Mehrlinser N-BK7, c_s 0,06", multi, lens_split(4, 0.06, 0.7));
    add_b("B Mehrlinser n = 1,72, c_s 0,03", multi, lens_split(6, 0.03, 0.5));
    add_b("B Geisterplatten, c_s -0,02", exact, lens_split(1, -0.02, 0.5));
    add_b("B NA 0,5, c_s 0,3", na05, lens_split(1, 0.3, 0.5));
    return out;
}

} // namespace mko::test
