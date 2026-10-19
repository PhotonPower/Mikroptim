// Diagnosebericht der Geisterplatten (docs/optimierung.md, Abschnitt 5 „Diagnose“).

#include "mikroptim/optim/ghost_diagnosis.hpp"

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/paraxial/ynu_trace.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "parallel_variants.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::size_t to_size(int i) {
    return static_cast<std::size_t>(i);
}

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Geisterplatten-Diagnose: " + message);
}

std::string num(double v, int precision = 6) {
    std::ostringstream os;
    os << std::setprecision(precision) << v;
    return os.str();
}

std::string glass_name(const MaterialSpec& m) {
    switch (m.kind) {
    case MaterialKind::Air:
        return "AIR";
    case MaterialKind::Catalog:
        return m.name;
    case MaterialKind::Constant:
        return "constant(n=" + num(m.n) + ")";
    case MaterialKind::Buchdahl:
        return "buchdahl(n0=" + num(m.n0) + ", nu=" + num(m.nu[0]) + "/" + num(m.nu[1]) +
               (m.nu[2] != 0.0 ? "/" + num(m.nu[2]) : std::string()) + ")";
    }
    return "?";
}

// Brechzahl bei λ oder NaN, wenn sie nicht auswertbar ist (z. B. außerhalb des Katalogbereichs).
double index_or_nan(const MaterialSpec& m, double lambda_nm, const GlassCatalog& catalog) {
    try {
        return material_index(m, lambda_nm, catalog);
    } catch (const std::exception&) {
        return kNaN;
    }
}

// Abbezahl und Teildispersion aus den Linien g, F, d, C (NaN bei dispersionsfreiem Glas).
void line_data(const MaterialSpec& m, const GlassCatalog& catalog, GhostGlassData& g) {
    const double ng = index_or_nan(m, kLineG, catalog);
    const double nf = index_or_nan(m, kLineF, catalog);
    const double nd = index_or_nan(m, kLineD, catalog);
    const double nc = index_or_nan(m, kLineC, catalog);
    g.n_d = nd;
    const double dfc = nf - nc;
    if (!(std::isfinite(dfc) && dfc != 0.0)) {
        g.abbe = kNaN;
        g.partial_gf = kNaN;
        g.delta_partial = kNaN;
        return;
    }
    g.abbe = (nd - 1.0) / dfc;
    g.partial_gf = (ng - nf) / dfc;
    // Normalgerade P_g,F = 0,6438 − 0,001682 ν_d (Schott, Bezugsgläser K7 und F2)
    g.delta_partial = g.partial_gf - (0.6438 - 0.001682 * g.abbe);
}

// Norm der Legendre-Koeffizienten x_k (k = 1 … K) als RMS des Profils über ξ ∈ [−1, 1]:
// Σ x_k² / (2k + 1) (Orthogonalität der P_k).
double legendre_norm(const std::vector<double>& x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        s += x[i] * x[i] / (2.0 * static_cast<double>(i + 1) + 1.0);
    }
    return std::sqrt(s);
}

// Paraxiale Höhen (Randstrahl volle Pupille, Hauptstrahl 1 rad) an allen Tabellenflächen der
// Variante v, mit den Geisterplatten als dünne Elemente wie batch_paraxial.
void paraxial_heights(const SystemBatchHost& b, int v, std::vector<double>& marginal,
                      std::vector<double>& chief) {
    const int S = b.surfaces();
    std::vector<double> c(to_size(S)), t(to_size(S)), n(to_size(S)), power(to_size(S), 0.0);
    for (int s = 0; s < S; ++s) {
        c[to_size(s)] = b.curvature(v, s);
        t[to_size(s)] = b.thickness(v, s);
        n[to_size(s)] = b.paraxial_index(v, s);
    }
    std::vector<double> coeff(to_size(std::max(b.ghost_terms(), 1)), 0.0);
    for (int g = 0; g < b.ghost_plates(); ++g) {
        const int s = b.ghost_surface(g);
        for (int k = 0; k < b.ghost_terms(); ++k) {
            coeff[to_size(k)] = b.ghost_coeff(v, g, k);
        }
        power[to_size(s)] = ghost_plate_power(coeff.data(), b.ghost_terms(), b.ghost_norm_radius(g),
                                              b.paraxial_ghost_index(v, g), n[to_size(s)]);
    }
    const double a = b.pupil_radius(v);
    ParaxialRay<double> m0{a, 0.0};
    if (b.source() == SourceKind::Point && b.source_point[2] != 0.0) {
        m0.u = a / -b.source_point[2]; // Randstrahl vom Quellpunkt auf der Achse
    }
    std::vector<ParaxialRay<double>> at_m(to_size(S)), at_c(to_size(S));
    paraxial_trace(c.data(), t.data(), n.data(), S, m0, at_m.data(), power.data());
    paraxial_trace(c.data(), t.data(), n.data(), S, ParaxialRay<double>{0.0, 1.0}, at_c.data(),
                   power.data());
    marginal.resize(to_size(S));
    chief.resize(to_size(S));
    for (int s = 0; s < S; ++s) {
        marginal[to_size(s)] = at_m[to_size(s)].y;
        chief[to_size(s)] = at_c[to_size(s)].y;
    }
}

std::string position_text(const GhostPositionReport& p) {
    return "nach Fläche " + std::to_string(p.after_surface) + " im Abstand " + num(p.distance) +
           " mm";
}

std::string orders_text(const std::vector<int>& orders) {
    std::string s;
    for (std::size_t i = 0; i < orders.size(); ++i) {
        s += (i > 0 ? ", " : "") + std::to_string(orders[i]);
    }
    return s.empty() ? "-" : s;
}

// Klassifikation einer Lage nach der Tabelle in Abschnitt 5.
void classify(const GhostDiagnosis& d, const GhostDiagnosisOptions& o,
              const std::vector<GhostGlassData>& glass, const std::vector<double>& n_medium,
              GhostPositionReport& p) {
    if (p.active_plates.empty()) {
        p.recommendation = GhostOperator::None;
        p.finding = "keine aktive Geisterplatte";
        p.reason = "alle |a_k| unter der Aktivitätsschwelle";
        return;
    }
    std::size_t k_max = 0;
    for (const int g : p.active_plates) {
        k_max = std::max(k_max, d.plates[to_size(g)].coefficients.size());
    }
    std::vector<double> mono(k_max, 0.0), chroma(k_max, 0.0);
    p.single_glass_ratio = 0.0;
    for (const int g : p.active_plates) {
        const GhostPlateReport& r = d.plates[to_size(g)];
        const double dn_ref = glass[to_size(g)].n_ref - n_medium[to_size(g)];
        const double dn = glass[to_size(g)].dispersion;
        for (std::size_t k = 0; k < r.coefficients.size(); ++k) {
            mono[k] += dn_ref * r.coefficients[k];
            chroma[k] += dn * r.coefficients[k];
        }
        if (dn_ref != 0.0) {
            p.single_glass_ratio = std::max(p.single_glass_ratio, std::abs(dn / dn_ref));
        }
    }
    if (d.merit_wavelengths < 2) {
        p.chromatic_ratio = kNaN;
        p.recommendation = GhostOperator::A;
        p.finding = "monochromatische Korrektur (sphärische Aberration höherer Ordnung, "
                    "Aplanasie, Petzval)";
        p.reason = "nur eine Merit-Wellenlänge: die Platten korrigieren das Muster dieser "
                   "Wellenlänge";
        return;
    }
    const double nm = legendre_norm(mono);
    const double nc = legendre_norm(chroma);
    p.chromatic_ratio =
        nm > 0.0 ? nc / nm : (nc > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
    const double limit = o.chromatic_factor * p.single_glass_ratio;
    if (!(p.chromatic_ratio > limit)) {
        p.recommendation = GhostOperator::A;
        p.finding = "hauptsächlich monochromatische Korrektur (gleiches Muster über alle "
                    "Wellenlängen)";
        p.reason = "Farbanteil χ = " + num(p.chromatic_ratio, 3) + " ≤ " +
                   num(o.chromatic_factor, 3) + " · " + num(p.single_glass_ratio, 3) +
                   " (χ einer einzelnen Platte der aktiven Gläser)";
        return;
    }
    p.recommendation = GhostOperator::B;
    p.finding = "Korrektur über Gläser mit unterschiedlicher Dispersion (Farbfehler 1. Ordnung)";
    p.reason = "Farbanteil χ = " + num(p.chromatic_ratio, 3) + " > " + num(o.chromatic_factor, 3) +
               " · " + num(p.single_glass_ratio, 3) +
               ": nur durch Kombination verschiedener Dispersionen erreichbar";
    for (const int g : p.active_plates) {
        const GhostGlassData& gl = glass[to_size(g)];
        const double share = p.sum_abs > 0.0 ? d.plates[to_size(g)].sum_abs / p.sum_abs : 0.0;
        if (std::isfinite(gl.delta_partial) && gl.delta_partial >= o.anomalous_delta &&
            share >= o.anomalous_share) {
            if (d.merit_wavelengths >= 3) {
                p.recommendation = GhostOperator::BAnomalous;
                p.finding = "Bedarf an anomaler Teildispersion (Fluorit-Platte aktiv): "
                            "sekundäres Spektrum";
                p.reason += "; Platte " + std::to_string(g) + " (" + gl.name +
                            ", ΔP_g,F = " + num(gl.delta_partial, 3) + ") trägt " +
                            num(100.0 * share, 3) + " % von Σ|a| der Lage bei " +
                            std::to_string(d.merit_wavelengths) + " Wellenlängen";
                return;
            }
            p.reason += "; Platte " + std::to_string(g) + " (" + gl.name +
                        ") mit anomaler Teildispersion aktiv, bei zwei Wellenlängen aber kein "
                        "Nachweis eines sekundären Spektrums (großer Abbe-Unterschied genügt)";
        }
    }
}

std::string recommendation_text(const GhostDiagnosis& d) {
    if (d.recommended_position < 0) {
        return "keine Empfehlung: keine Geisterplatte aktiv (das reale System erreicht das Ziel "
               "ohne Geisterkorrektur)";
    }
    const GhostPositionReport& p = d.positions[to_size(d.recommended_position)];
    // aktivste Platte der Lage (größte Σ|a|)
    int best = p.active_plates.front();
    for (const int g : p.active_plates) {
        if (d.plates[to_size(g)].sum_abs > d.plates[to_size(best)].sum_abs) {
            best = g;
        }
    }
    const GhostPlateReport& r = d.plates[to_size(best)];
    const std::string where = position_text(p);
    switch (p.recommendation) {
    case GhostOperator::A:
        return "Operator A (Meniskus ohne Brechkraft) " + where + ", Glas wie Platte " +
               std::to_string(best) + " (" + r.glass.name + "), dominante Ordnung " +
               orders_text(r.dominant_orders) + "; Biegung aus dem Vorzeichen der Korrektur";
    case GhostOperator::B:
        return "Operator B (Aufspalten in ein Kittglied aus normalen Gläsern) an einer Linse "
               "nahe " +
               where + "; aktive Gläser der Lage zeigen die Dispersionsdifferenz";
    case GhostOperator::BAnomalous:
        return "Operator B mit Fluorit- oder FK51-artigem Glas an einer Linse nahe " + where +
               " (sekundäres Spektrum)";
    case GhostOperator::None:
        break;
    }
    return "keine Empfehlung";
}

} // namespace

const char* to_string(GhostOperator op) {
    switch (op) {
    case GhostOperator::None:
        return "none";
    case GhostOperator::A:
        return "A";
    case GhostOperator::B:
        return "B";
    case GhostOperator::BAnomalous:
        return "B_anomalous";
    }
    return "?";
}

BatchMeritFn make_cpu_merit(const ResidualSettings& quadrature, ExtraResiduals extra, int threads,
                            TraceExtraResiduals trace_extra) {
    check_trace_extra_residuals(trace_extra, "make_cpu_merit");
    return [quadrature, extra = std::move(extra), threads,
            traced = std::move(trace_extra)](const SystemBatchHost& batch) {
        const ResidualLayout layout(batch.topology(), quadrature);
        const std::vector<double> e = extra_merit_values(extra, batch, threads);
        std::vector<double> f(to_size(batch.variants()), 0.0);
        detail::parallel_for_variants(0, batch.variants(), threads, [&](int v) {
            const MeritResult m = merit_cpu(batch, v, layout, traced.count > 0);
            f[to_size(v)] = e.empty() ? m.merit[0] : m.merit[0] + e[to_size(v)];
            if (traced.count > 0) {
                f[to_size(v)] += trace_extra_merit(traced, batch, v, variant_extrema(m.extrema, 0));
            }
        });
        return f;
    };
}

GhostDiagnosis diagnose_ghost_plates(const SystemDescription& system, const GlassCatalog& catalog,
                                     const SystemBatchHost& batch, int v,
                                     const GhostPenalty& penalty, const BatchMeritFn& merit,
                                     const GhostDiagnosisOptions& options) {
    const int G = static_cast<int>(system.ghost_plates.size());
    if (batch.ghost_plates() != G) {
        invalid("der Batch hat " + std::to_string(batch.ghost_plates()) +
                " Geisterplatten, die Systembeschreibung " + std::to_string(G));
    }
    if (v < 0 || v >= batch.variants()) {
        throw std::out_of_range("Geisterplatten-Diagnose: Variante " + std::to_string(v) +
                                " außerhalb des Batches");
    }
    check_ghost_penalty(penalty, batch.topology());
    const SurfaceMap map = make_surface_map(system);

    GhostDiagnosis d;
    d.reference_nm = batch.reference_wavelength_nm;
    // Merit-Wellenlängen (Gewicht > 0): kürzeste und längste für den Farbanteil
    int l_short = -1;
    int l_long = -1;
    for (int l = 0; l < batch.wavelengths(); ++l) {
        if (!(batch.wavelength_weight(l) > 0.0)) {
            continue;
        }
        ++d.merit_wavelengths;
        if (l_short < 0 || batch.wavelength_nm(l) < batch.wavelength_nm(l_short)) {
            l_short = l;
        }
        if (l_long < 0 || batch.wavelength_nm(l) > batch.wavelength_nm(l_long)) {
            l_long = l;
        }
    }
    d.short_nm = l_short >= 0 ? batch.wavelength_nm(l_short) : kNaN;
    d.long_nm = l_long >= 0 ? batch.wavelength_nm(l_long) : kNaN;

    // Merit am Punkt und je Platte mit a = 0 (Varianten 0 … G eines eigenen Batches)
    std::vector<double> f;
    if (merit) {
        // Kopie mit allen gemeinsamen Feldern, Variante v in 0 … G
        SystemBatchHost probe = batch;
        if (v != 0) {
            probe.copy_variant(v, 0);
        }
        probe.resize(options.merit_without ? G + 1 : 1);
        for (int i = 1; i < probe.variants(); ++i) {
            probe.copy_variant(0, i);
        }
        if (options.merit_without) {
            for (int g = 0; g < G; ++g) {
                for (int k = 0; k < batch.ghost_terms(); ++k) {
                    probe.ghost_coeff(g + 1, g, k) = 0.0;
                }
            }
        }
        f = merit(probe);
        if (f.size() != to_size(probe.variants())) {
            invalid("Merit-Funktion liefert " + std::to_string(f.size()) + " Werte, erwartet " +
                    std::to_string(probe.variants()));
        }
        d.merit = f[0];
    } else {
        d.merit = kNaN;
    }

    std::vector<double> marginal, chief;
    paraxial_heights(batch, v, marginal, chief);

    std::vector<GhostGlassData> glass(to_size(G));
    std::vector<double> n_medium(to_size(G));
    for (int g = 0; g < G; ++g) {
        const GhostPlate& gp = system.ghost_plates[to_size(g)];
        GhostPlateReport r;
        r.plate = g;
        r.after_surface = gp.after_surface;
        r.distance = gp.distance;
        r.table_surface = map.table_of_ghost[to_size(g)];
        if (r.table_surface != batch.ghost_surface(g)) {
            invalid("Lage der Geisterplatte " + std::to_string(g) + " passt nicht zum Batch");
        }
        r.norm_radius = batch.ghost_norm_radius(g);
        r.order = static_cast<int>(gp.coefficients.size());
        if (r.order > batch.ghost_terms()) {
            invalid("Ordnung der Geisterplatte " + std::to_string(g) + " größer als K des Batches");
        }

        GhostGlassData& gl = glass[to_size(g)];
        gl.name = glass_name(gp.glass);
        gl.n_ref = batch.paraxial_ghost_index(v, g);
        gl.dispersion = (l_short >= 0 && l_long >= 0 && l_short != l_long)
                            ? batch.ghost_index(v, g, l_short) - batch.ghost_index(v, g, l_long)
                            : 0.0;
        line_data(gp.glass, catalog, gl);
        r.glass = gl;
        n_medium[to_size(g)] = batch.paraxial_index(v, r.table_surface);

        // Strafe der Platte (Glättung für die Aktivitätsschwelle)
        double smoothing = kDefaultGhostSmoothing;
        for (const GhostPenaltyTerm& t : penalty.terms) {
            if (t.plate == g) {
                r.penalized = true;
                smoothing = t.smoothing;
            }
        }
        r.penalty = ghost_penalty_merit(penalty, batch, v, g);
        r.activity_threshold =
            options.activity_threshold > 0.0 ? options.activity_threshold : 10.0 * smoothing;

        for (int k = 0; k < r.order; ++k) {
            const double a = batch.ghost_coeff(v, g, k);
            r.coefficients.push_back(a);
            r.sum_abs += std::abs(a);
            r.max_abs = std::max(r.max_abs, std::abs(a));
        }
        r.active = r.max_abs > r.activity_threshold;
        if (r.active) {
            for (int k = 1; k <= r.order; ++k) {
                const double a = std::abs(r.coefficients[to_size(k - 1)]);
                if (a > r.activity_threshold && a >= options.dominant_fraction * r.max_abs) {
                    r.dominant_orders.push_back(k);
                }
            }
            std::stable_sort(r.dominant_orders.begin(), r.dominant_orders.end(), [&](int a, int b) {
                return std::abs(r.coefficients[to_size(a - 1)]) >
                       std::abs(r.coefficients[to_size(b - 1)]);
            });
        }
        r.power = ghost_plate_power(r.coefficients.data(), r.order, r.norm_radius, gl.n_ref,
                                    n_medium[to_size(g)]);
        r.petzval = ghost_plate_petzval(r.coefficients.data(), r.order, r.norm_radius, gl.n_ref,
                                        n_medium[to_size(g)]);
        r.marginal_height = marginal[to_size(r.table_surface)];
        r.chief_height_per_rad = chief[to_size(r.table_surface)];
        if (!f.empty() && options.merit_without) {
            r.merit_without = f[to_size(g + 1)];
            r.merit_gain = r.merit_without - d.merit;
        } else {
            r.merit_without = kNaN;
            r.merit_gain = kNaN;
        }
        d.plates.push_back(std::move(r));
    }

    // Lagen: gleiche (after_surface, distance), in Dateireihenfolge
    for (int g = 0; g < G; ++g) {
        const GhostPlateReport& r = d.plates[to_size(g)];
        GhostPositionReport* pos = nullptr;
        for (GhostPositionReport& p : d.positions) {
            if (p.after_surface == r.after_surface && p.distance == r.distance) {
                pos = &p;
            }
        }
        if (pos == nullptr) {
            d.positions.emplace_back();
            pos = &d.positions.back();
            pos->after_surface = r.after_surface;
            pos->distance = r.distance;
        }
        pos->plates.push_back(g);
        if (r.active) {
            pos->active_plates.push_back(g);
            pos->merit_gain += r.merit_gain;
            pos->sum_abs += r.sum_abs;
        }
    }
    for (GhostPositionReport& p : d.positions) {
        if (p.active_plates.empty()) {
            p.merit_gain = f.empty() || !options.merit_without ? kNaN : 0.0;
        }
        classify(d, options, glass, n_medium, p);
    }

    // Empfehlung: Lage mit dem größten Merit-Gewinn (ohne Merit: größte Σ|a|)
    for (std::size_t i = 0; i < d.positions.size(); ++i) {
        const GhostPositionReport& p = d.positions[i];
        if (p.active_plates.empty()) {
            continue;
        }
        if (d.recommended_position < 0) {
            d.recommended_position = static_cast<int>(i);
            continue;
        }
        const GhostPositionReport& b = d.positions[to_size(d.recommended_position)];
        const bool better = std::isfinite(p.merit_gain) && std::isfinite(b.merit_gain)
                                ? p.merit_gain > b.merit_gain
                                : p.sum_abs > b.sum_abs;
        if (better) {
            d.recommended_position = static_cast<int>(i);
        }
    }
    d.recommendation = d.recommended_position >= 0
                           ? d.positions[to_size(d.recommended_position)].recommendation
                           : GhostOperator::None;
    d.recommendation_text = recommendation_text(d);
    return d;
}

// ---- Ausgabe ------------------------------------------------------------------------------

namespace {

std::string sci(double v, int digits = 3) {
    if (!std::isfinite(v)) {
        return "-";
    }
    std::ostringstream os;
    os << std::scientific << std::setprecision(digits) << v;
    return os.str();
}

std::string fix(double v, int digits = 4) {
    if (!std::isfinite(v)) {
        return "-";
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(digits) << v;
    return os.str();
}

std::string pad(const std::string& s, std::size_t w) {
    // Breite in Zeichen (UTF-8: Folgebytes zählen nicht)
    std::size_t chars = 0;
    for (const char c : s) {
        if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) {
            ++chars;
        }
    }
    return chars >= w ? s : std::string(w - chars, ' ') + s;
}

using Json = nlohmann::ordered_json;

Json number(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

} // namespace

void write_ghost_diagnosis_text(const GhostDiagnosis& d, std::ostream& out) {
    out << "Geisterplatten (Diagnose, docs/optimierung.md Abschnitt 5)\n";
    out << "  Merit F [Wellen²]: " << sci(d.merit, 9)
        << ", Merit-Wellenlängen: " << d.merit_wavelengths;
    if (d.merit_wavelengths >= 2) {
        out << " (" << fix(d.short_nm, 2) << " … " << fix(d.long_nm, 2) << " nm)";
    }
    out << '\n';
    out << "  " << pad("Pl", 3) << pad("Fläche", 7) << pad("Abstand", 9) << pad("K", 3)
        << pad("Summe|a| [mm]", 15) << pad("dominant", 10) << pad("phi_G [1/mm]", 14)
        << pad("y_R [mm]", 10) << pad("y_H/rad", 10) << pad("Strafe", 11) << pad("Gewinn", 11)
        << pad("aktiv", 7) << pad("n_ref", 8) << pad("nu_d", 7) << pad("dP_gF", 8) << "  Glas\n";
    for (const GhostPlateReport& r : d.plates) {
        out << "  " << pad(std::to_string(r.plate), 3) << pad(std::to_string(r.after_surface), 7)
            << pad(fix(r.distance, 3), 9) << pad(std::to_string(r.order), 3)
            << pad(sci(r.sum_abs), 15) << pad(orders_text(r.dominant_orders), 10)
            << pad(sci(r.power), 14) << pad(fix(r.marginal_height, 3), 10)
            << pad(fix(r.chief_height_per_rad, 3), 10) << pad(sci(r.penalty, 2), 11)
            << pad(sci(r.merit_gain, 2), 11) << pad(r.active ? "ja" : "nein", 7)
            << pad(fix(r.glass.n_ref, 4), 8) << pad(fix(r.glass.abbe, 1), 7)
            << pad(fix(r.glass.delta_partial, 4), 8) << "  " << r.glass.name << '\n';
    }
    out << "  (Strafe = Σ r² der Platte, Gewinn = F(a = 0 für diese Platte) − F in Wellen²; "
           "y_R Randstrahl-, y_H Hauptstrahlhöhe paraxial)\n";
    for (const GhostPlateReport& r : d.plates) {
        if (!r.active) {
            continue;
        }
        out << "  Platte " << r.plate << " a_k [mm]:";
        for (std::size_t k = 0; k < r.coefficients.size(); ++k) {
            out << ' ' << sci(r.coefficients[k], 4);
        }
        out << '\n';
    }
    out << "  Lagen:\n";
    for (const GhostPositionReport& p : d.positions) {
        out << "    " << position_text(p) << ": Platten " << orders_text(p.plates) << ", aktiv "
            << orders_text(p.active_plates) << " -> " << to_string(p.recommendation) << '\n';
        out << "      Befund: " << p.finding << '\n';
        out << "      Begründung: " << p.reason << '\n';
    }
    out << "  Empfehlung: " << d.recommendation_text << '\n';
}

std::string ghost_diagnosis_json(const GhostDiagnosis& d, int indent) {
    Json j;
    j["merit"] = number(d.merit);
    j["reference_nm"] = number(d.reference_nm);
    j["merit_wavelengths"] = d.merit_wavelengths;
    j["short_nm"] = number(d.short_nm);
    j["long_nm"] = number(d.long_nm);
    Json plates = Json::array();
    for (const GhostPlateReport& r : d.plates) {
        Json e;
        e["plate"] = r.plate;
        e["after_surface"] = r.after_surface;
        e["distance"] = number(r.distance);
        e["table_surface"] = r.table_surface;
        e["norm_radius"] = number(r.norm_radius);
        e["order"] = r.order;
        Json g;
        g["name"] = r.glass.name;
        g["n_ref"] = number(r.glass.n_ref);
        g["dispersion"] = number(r.glass.dispersion);
        g["n_d"] = number(r.glass.n_d);
        g["abbe"] = number(r.glass.abbe);
        g["partial_gf"] = number(r.glass.partial_gf);
        g["delta_partial_gf"] = number(r.glass.delta_partial);
        e["glass"] = std::move(g);
        Json c = Json::array();
        for (const double a : r.coefficients) {
            c.push_back(number(a));
        }
        e["coefficients"] = std::move(c);
        e["sum_abs"] = number(r.sum_abs);
        e["max_abs"] = number(r.max_abs);
        e["activity_threshold"] = number(r.activity_threshold);
        e["active"] = r.active;
        e["dominant_orders"] = r.dominant_orders;
        e["power"] = number(r.power);
        e["petzval"] = number(r.petzval);
        e["marginal_height"] = number(r.marginal_height);
        e["chief_height_per_rad"] = number(r.chief_height_per_rad);
        e["penalized"] = r.penalized;
        e["penalty"] = number(r.penalty);
        e["merit_without"] = number(r.merit_without);
        e["merit_gain"] = number(r.merit_gain);
        plates.push_back(std::move(e));
    }
    j["plates"] = std::move(plates);
    Json positions = Json::array();
    for (const GhostPositionReport& p : d.positions) {
        Json e;
        e["after_surface"] = p.after_surface;
        e["distance"] = number(p.distance);
        e["plates"] = p.plates;
        e["active_plates"] = p.active_plates;
        e["chromatic_ratio"] = number(p.chromatic_ratio);
        e["single_glass_ratio"] = number(p.single_glass_ratio);
        e["merit_gain"] = number(p.merit_gain);
        e["sum_abs"] = number(p.sum_abs);
        e["recommendation"] = to_string(p.recommendation);
        e["finding"] = p.finding;
        e["reason"] = p.reason;
        positions.push_back(std::move(e));
    }
    j["positions"] = std::move(positions);
    j["recommended_position"] =
        d.recommended_position >= 0 ? Json(d.recommended_position) : Json(nullptr);
    j["recommendation"] = to_string(d.recommendation);
    j["recommendation_text"] = d.recommendation_text;
    return j.dump(indent, ' ', false, Json::error_handler_t::replace);
}

} // namespace mko
