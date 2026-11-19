// Adapter Raytatouille (`*.rtt.json`) ↔ `mikroptim.system/1` (M9-T2, docs/dateiformat.md,
// Abschnitt „Raytatouille-Adapter“). nlohmann/json bleibt wie in system_json.cpp privat.

#include "mikroptim/io/raytatouille.hpp"

#include "mikroptim/io/system_json.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace mko {

namespace {

using Json = nlohmann::ordered_json;

// ---- Zahlen verlustfrei ------------------------------------------------------------------

// Kürzeste Darstellung, die beim Lesen denselben Wert ergibt (std::to_chars).
std::string shortest(double x) {
    std::array<char, 64> buf{};
    const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), x);
    return std::string(buf.data(), r.ptr);
}

std::size_t significant_digits(double x) {
    const std::string s = shortest(x);
    std::size_t n = 0;
    bool leading = true;
    for (const char ch : s) {
        if (ch == 'e' || ch == 'E') {
            break;
        }
        if (ch >= '0' && ch <= '9') {
            if (ch == '0' && leading) {
                continue;
            }
            leading = false;
            ++n;
        }
    }
    return n;
}

bool parse_number(std::string_view text, double& value) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto r = std::from_chars(first, last, value);
    return r.ec == std::errc() && r.ptr == last;
}

// Verschiebt das Dezimalkomma der kürzesten Darstellung von x um e Stellen.
double shift_decimal(double x, int e) {
    if (!std::isfinite(x) || x == 0.0) {
        return x;
    }
    const std::string s = shortest(x);
    const std::size_t pos = s.find_first_of("eE");
    int exp10 = 0;
    std::string mantissa = s;
    if (pos != std::string::npos) {
        mantissa = s.substr(0, pos);
        const std::string ex = s.substr(pos + 1);
        const char* b = ex.data();
        if (!ex.empty() && ex.front() == '+') {
            ++b;
        }
        const auto r = std::from_chars(b, ex.data() + ex.size(), exp10);
        if (r.ec != std::errc()) {
            return x;
        }
    }
    const std::string t = mantissa + "e" + std::to_string(exp10 + e);
    double y = 0.0;
    parse_number(t, y);
    return y;
}

// Sucht y in wenigen ulp um y0, für das back(y) == x gilt; unter mehreren das mit der
// kürzesten Dezimaldarstellung, bei Gleichstand das nächste an y0. Ohne Treffer y0.
template <class Back>
double exact_preimage(double x, double y0, Back back) {
    constexpr int kSteps = 4;
    double best = y0;
    bool found = false;
    std::size_t best_digits = 0;
    int best_distance = 0;
    double below = y0;
    double above = y0;
    for (int k = 0; k <= kSteps; ++k) {
        for (const double y : {below, above}) {
            if (!std::isfinite(y) || !(back(y) == x)) {
                continue;
            }
            const std::size_t d = significant_digits(y);
            if (!found || d < best_digits || (d == best_digits && k < best_distance)) {
                found = true;
                best = y;
                best_digits = d;
                best_distance = k;
            }
        }
        below = std::nextafter(below, -std::numeric_limits<double>::infinity());
        above = std::nextafter(above, std::numeric_limits<double>::infinity());
    }
    return best;
}

std::string pointer(const std::string& base, std::string_view key) {
    std::string k(key);
    std::string escaped;
    for (const char ch : k) {
        if (ch == '~') {
            escaped += "~0";
        } else if (ch == '/') {
            escaped += "~1";
        } else {
            escaped += ch;
        }
    }
    return base + "/" + escaped;
}

std::string pointer(const std::string& base, std::size_t i) {
    return base + "/" + std::to_string(i);
}

std::string pointer_text(const std::string& p) {
    return p.empty() ? std::string("/") : p;
}

// ---- Raytatouille → Mikroptim ------------------------------------------------------------

constexpr int kMaxSchemaMinor = 11;
constexpr int kMinSchemaMinor = 4;

// Lage eines Koordinatensystems entlang z als Summanden in Baumreihenfolge. Abstände werden
// nach Abzug des gemeinsamen Anfangs gebildet, damit relative Lagen exakt bleiben.
using Terms = std::vector<double>;

double sum(const Terms& t, std::size_t from) {
    double s = 0.0;
    for (std::size_t i = from; i < t.size(); ++i) {
        s += t[i];
    }
    return s;
}

double distance(const Terms& from, const Terms& to) {
    std::size_t k = 0;
    while (k < from.size() && k < to.size() && from[k] == to[k]) {
        ++k;
    }
    if (k == from.size()) {
        return sum(to, k);
    }
    return sum(to, k) - sum(from, k);
}

enum class ElementKind { Stop, Lens, Thin, Detector };

struct SurfaceRec {
    std::string ptr; // Pointer der Fläche
    Terms frame;     // Lage des Scheitels
    ElementKind kind;
    bool active = true;
    Surface surface; // Krümmung, Typ, Halbdurchmesser, Medium danach
};

class Importer {
public:
    Importer(const RttImportOptions& options, std::vector<std::string>* warnings)
        : options_(options), warnings_(warnings) {}

    SystemDescription run(const Json& root) {
        if (!root.is_object()) {
            error("", "Objekt auf oberster Ebene erwartet");
            throw RttConversionError(issues_);
        }
        root_ = &root;
        check_header(root);
        read_parameters(root);

        SystemDescription sys;
        sys.name = options_.fallback_name;
        if (const Json* n = find(root, "name")) {
            if (n->is_string() && !n->get<std::string>().empty()) {
                sys.name = n->get<std::string>();
            }
        }
        sys.glass_catalogs = options_.glass_catalogs;
        read_environment(root);
        sys.spectrum = read_wavelengths(root);
        sys.source = read_fields(root);
        read_object(root);
        read_paths(root);
        for (const char* key : {"optimization", "tolerances"}) {
            if (find(root, key) != nullptr) {
                warn(pointer("", key), std::string("Abschnitt ") + key +
                                           " hat in mikroptim.system/1 kein Gegenstück und "
                                           "entfällt");
            }
        }
        if (const Json* c = find(root, "configurations")) {
            if (c->is_array() && c->size() > 1) {
                warn("/configurations", "mehrere Konfigurationen: übernommen wird nur die erste (" +
                                            std::to_string(c->size()) + " vorhanden)");
            }
        }

        const Json* tree = find(root, "root");
        if (tree == nullptr || !tree->is_object()) {
            error("/root", "Baugruppe `root` fehlt");
        } else {
            const Terms origin;
            Terms preceding;
            bool has_preceding = false;
            (void)walk_node(*tree, "/root", origin, nullptr, preceding, has_preceding, true);
        }
        read_aperture(root);

        build_surfaces(sys);
        if (!issues_.empty()) {
            throw RttConversionError(issues_);
        }
        try {
            validate_system(sys);
        } catch (const FormatError& e) {
            throw RttConversionError(
                {{e.json_path(), "ungültige Systembeschreibung: " + e.message()}});
        }
        if (options_.glass_exists) {
            try {
                validate_materials(sys, options_.glass_exists);
            } catch (const FormatError& e) {
                throw RttConversionError({{e.json_path(), e.message()}});
            }
        }
        return sys;
    }

private:
    static const Json* find(const Json& j, std::string_view key) {
        if (!j.is_object()) {
            return nullptr;
        }
        const auto it = j.find(std::string(key));
        return it == j.end() ? nullptr : &*it;
    }

    void error(const std::string& ptr, std::string message) {
        issues_.push_back({pointer_text(ptr), std::move(message)});
    }

    void warn(const std::string& ptr, const std::string& message) {
        if (warnings_ != nullptr) {
            warnings_->push_back(pointer_text(ptr) + ": " + message);
        }
    }

    // Zahl oder Param (Zahl, {"value", "variable", "min", "max"}, {"param": NAME}).
    std::optional<double> param(const Json& j, const std::string& ptr) {
        if (j.is_number()) {
            return j.get<double>();
        }
        if (j.is_object()) {
            if (const Json* v = find(j, "value")) {
                if (!v->is_number()) {
                    error(pointer(ptr, "value"), "Zahl erwartet");
                    return std::nullopt;
                }
                const Json* var = find(j, "variable");
                if ((var != nullptr && var->is_boolean() && var->get<bool>()) ||
                    find(j, "min") != nullptr || find(j, "max") != nullptr) {
                    warn(ptr, "Optimierungsvariable bzw. Grenzen entfallen, übernommen wird "
                              "nur der Wert (Variablen in mikroptim.system/1 selbst anlegen)");
                }
                return v->get<double>();
            }
            if (const Json* p = find(j, "param")) {
                if (!p->is_string()) {
                    error(pointer(ptr, "param"), "Zeichenkette erwartet");
                    return std::nullopt;
                }
                return parameter_value(p->get<std::string>(), ptr);
            }
        }
        error(ptr, "Zahl oder Param erwartet");
        return std::nullopt;
    }

    double param_or(const Json& parent, std::string_view key, const std::string& ptr,
                    double fallback) {
        if (const Json* j = find(parent, key)) {
            if (const auto v = param(*j, pointer(ptr, key))) {
                return *v;
            }
        }
        return fallback;
    }

    void read_parameters(const Json& root) {
        const Json* p = find(root, "parameters");
        if (p == nullptr) {
            return;
        }
        if (!p->is_array()) {
            error("/parameters", "Liste erwartet");
            return;
        }
        for (std::size_t i = 0; i < p->size(); ++i) {
            const Json& row = (*p)[i];
            const Json* n = find(row, "name");
            if (n != nullptr && n->is_string()) {
                parameters_.emplace_back(n->get<std::string>(), i);
            }
        }
    }

    std::optional<double> parameter_value(const std::string& name, const std::string& ptr) {
        for (const auto& [row_name, index] : parameters_) {
            if (row_name != name) {
                continue;
            }
            const Json& row = root_->at("parameters").at(index);
            const std::string row_ptr = pointer("/parameters", index);
            if (const Json* v = find(row, "value"); v != nullptr && v->is_number()) {
                warn(ptr,
                     "Bindung an den Parameter " + name + " entfällt, übernommen wird sein Wert");
                return v->get<double>();
            }
            if (const Json* v = find(row, "values");
                v != nullptr && v->is_array() && !v->empty() && (*v)[0].is_number()) {
                warn(ptr, "Bindung an den Parameter " + name +
                              " entfällt, übernommen wird sein Wert der ersten Konfiguration");
                return (*v)[0].get<double>();
            }
            if (find(row, "expression") != nullptr) {
                error(ptr, "Parameter " + name +
                               " ist ein Ausdruck; Ausdrücke wertet der Adapter nicht aus");
                return std::nullopt;
            }
            error(row_ptr, "Parameter " + name + " ohne Zahlenwert");
            return std::nullopt;
        }
        error(ptr, "unbekannter Parameter " + name);
        return std::nullopt;
    }

    void check_header(const Json& root) {
        const Json* v = find(root, "schema_version");
        if (v == nullptr || !v->is_string()) {
            error("/schema_version", "schema_version fehlt (keine Raytatouille-Datei?)");
        } else {
            const std::string s = v->get<std::string>();
            int minor = -1;
            if (s.rfind("0.", 0) == 0) {
                const std::size_t dot = s.find('.', 2);
                const std::string m =
                    s.substr(2, dot == std::string::npos ? std::string::npos : dot - 2);
                const auto r = std::from_chars(m.data(), m.data() + m.size(), minor);
                if (r.ec != std::errc() || r.ptr != m.data() + m.size()) {
                    minor = -1;
                }
            }
            if (minor < kMinSchemaMinor || minor > kMaxSchemaMinor) {
                error("/schema_version",
                      "Schema " + s + " nicht unterstützt (unterstützt: 0.4 bis 0.11)");
            }
        }
        const Json* u = find(root, "units");
        const bool units_ok = u != nullptr && u->is_object() && find(*u, "length") != nullptr &&
                              (*u)["length"] == "mm" && find(*u, "wavelength") != nullptr &&
                              (*u)["wavelength"] == "um";
        if (!units_ok) {
            error("/units", "Einheiten {\"length\": \"mm\", \"wavelength\": \"um\"} erwartet");
        }
    }

    void read_environment(const Json& root) {
        const Json* env = find(root, "environment");
        if (env == nullptr) {
            return;
        }
        if (const Json* m = find(*env, "medium")) {
            const std::string medium = m->is_string() ? m->get<std::string>() : std::string();
            if (medium == "VACUUM") {
                vacuum_ = true;
            } else if (medium != "AIR") {
                error("/environment/medium",
                      "Umgebung " + m->dump() +
                          " nicht abbildbar (Mikroptim: Luft mit n = 1; erlaubt AIR, VACUUM)");
            }
        }
        const Json* t = find(*env, "temperature_c");
        const Json* p = find(*env, "pressure_atm");
        if ((t != nullptr && !(t->is_number() && t->get<double>() == 20.0)) ||
            (p != nullptr && !(p->is_number() && p->get<double>() == 1.0))) {
            warn("/environment", "Temperatur und Druck entfallen (Mikroptim rechnet ohne "
                                 "Temperaturmodell, Brechzahlen relativ zu Luft)");
        }
    }

    Spectrum read_wavelengths(const Json& root) {
        Spectrum s;
        const Json* w = find(root, "wavelengths");
        if (w == nullptr || !w->is_array() || w->empty()) {
            error("/wavelengths", "mindestens eine Wellenlänge erwartet");
            return s;
        }
        int references = 0;
        for (std::size_t i = 0; i < w->size(); ++i) {
            const Json& e = (*w)[i];
            const std::string ptr = pointer("/wavelengths", i);
            const Json* um = find(e, "um");
            if (um == nullptr || !um->is_number() || !(um->get<double>() > 0.0)) {
                error(pointer(ptr, "um"), "Wellenlänge > 0 in µm erwartet");
                continue;
            }
            const double nm = rtt_um_to_nm(um->get<double>());
            double weight = 1.0;
            if (const Json* g = find(e, "weight")) {
                if (!g->is_number()) {
                    error(pointer(ptr, "weight"), "Zahl erwartet");
                } else {
                    weight = g->get<double>();
                }
            }
            s.wavelengths_nm.push_back(nm);
            s.weights.push_back(weight);
            if (const Json* r = find(e, "reference");
                r != nullptr && r->is_boolean() && r->get<bool>()) {
                ++references;
                s.reference_nm = nm;
            }
        }
        if (references != 1) {
            error("/wavelengths", "genau eine Wellenlänge mit \"reference\": true erwartet, "
                                  "gefunden " +
                                      std::to_string(references));
        }
        return s;
    }

    Source read_fields(const Json& root) {
        Source src;
        src.type = SourceType::Collimated;
        const Json* f = find(root, "fields");
        if (f == nullptr || !f->is_object()) {
            error("/fields", "Abschnitt fields fehlt");
            return src;
        }
        if (const Json* t = find(*f, "type")) {
            if (!(t->is_string() && t->get<std::string>() == "angle_deg")) {
                error("/fields/type", "Feldart " + t->dump() +
                                          " nicht abbildbar (Mikroptim: Feldwinkel, "
                                          "angle_deg)");
                return src;
            }
        }
        const Json* pts = find(*f, "points");
        if (pts == nullptr || !pts->is_array() || pts->empty()) {
            error("/fields/points", "mindestens ein Feldpunkt erwartet");
            return src;
        }
        for (std::size_t i = 0; i < pts->size(); ++i) {
            const Json& p = (*pts)[i];
            const std::string ptr = pointer("/fields/points", i);
            if (!p.is_object()) {
                error(ptr, "Objekt erwartet");
                continue;
            }
            double x = 0.0;
            double y = 0.0;
            double weight = 1.0;
            for (const auto& [key, out] : {std::pair<const char*, double*>{"x", &x},
                                           std::pair<const char*, double*>{"y", &y},
                                           std::pair<const char*, double*>{"weight", &weight}}) {
                if (const Json* v = find(p, key)) {
                    if (!v->is_number()) {
                        error(pointer(ptr, key), "Zahl erwartet");
                    } else {
                        *out = v->get<double>();
                    }
                }
            }
            if (x != 0.0) {
                error(pointer(ptr, "x"), "Feldwinkel in x nicht abbildbar (Mikroptim: Felder "
                                         "in der y-z-Ebene)");
            }
            src.field_angles_deg.push_back(y);
            src.field_weights.push_back(weight);
        }
        return src;
    }

    void read_object(const Json& root) {
        const Json* o = find(root, "object");
        if (o == nullptr) {
            return;
        }
        const Json* inf = find(*o, "at_infinity");
        if (inf == nullptr || !inf->is_boolean() || !inf->get<bool>()) {
            error("/object", "endliches Objekt nicht abbildbar (Mikroptim: kollimierte Quelle, "
                             "Objekt im Unendlichen, ADR-0006)");
        }
    }

    void read_paths(const Json& root) {
        const Json* p = find(root, "paths");
        if (p == nullptr || !p->is_array() || p->empty()) {
            error("/paths", "mindestens ein Pfad erwartet");
            return;
        }
        std::optional<std::size_t> automatic;
        for (std::size_t i = 0; i < p->size(); ++i) {
            const Json* e = find((*p)[i], "events");
            if (e != nullptr && e->is_string() && e->get<std::string>() == "auto") {
                automatic = i;
                break;
            }
        }
        if (!automatic) {
            error("/paths", "kein Pfad mit \"events\": \"auto\"; explizite Ereignislisten "
                            "(Doppeldurchgang, Strahlteiler, Beugungsordnungen) sind nicht "
                            "abbildbar");
            return;
        }
        if (p->size() > 1) {
            warn("/paths", "übernommen wird nur der automatische Pfad " +
                               pointer("/paths", *automatic) + ", weitere Pfade entfallen");
        }
    }

    void read_aperture(const Json& root) {
        const Json* a = find(root, "aperture");
        if (a == nullptr || !a->is_object()) {
            error("/aperture", "Systemapertur fehlt");
            return;
        }
        if (find(*a, "stop") != nullptr) {
            error("/aperture/stop", "Blende auf einer Linsenfläche nicht abbildbar (Mikroptim: "
                                    "Blende als eigene erste Fläche in Luft, ADR-0006)");
        }
        const Json* t = find(*a, "type");
        const std::string type = t != nullptr && t->is_string() ? t->get<std::string>() : "";
        if (type == "epd") {
            if (const Json* v = find(*a, "value")) {
                if (const auto epd = param(*v, "/aperture/value")) {
                    if (!(*epd > 0.0)) {
                        error("/aperture/value", "Durchmesser der Eintrittspupille > 0 erwartet");
                    } else {
                        pupil_radius_ = *epd / 2.0;
                    }
                }
            } else {
                error("/aperture/value", "Durchmesser der Eintrittspupille fehlt");
            }
        } else if (type == "stop_size") {
            pupil_radius_ = stop_radius_;
            if (!stop_radius_) {
                error("/aperture", "stop_size verlangt eine kreisförmige Apertur der Blende");
            }
        } else {
            error("/aperture/type", "Systemapertur " + (t != nullptr ? t->dump() : "") +
                                        " nicht abbildbar (unterstützt: epd, stop_size)");
        }
        if (type == "epd" && pupil_radius_ && stop_radius_ && *stop_radius_ != *pupil_radius_) {
            warn(stop_ptr_ + "/aperture", "Blendenradius " + shortest(*stop_radius_) +
                                              " mm weicht von EPD/2 = " + shortest(*pupil_radius_) +
                                              " mm ab; Mikroptim nimmt EPD/2 als Pupillenradius");
        }
    }

    // Lage aus einer Pose. Nur Verschiebungen entlang z; x, y und Drehungen sind Fehler.
    struct PoseInfo {
        std::string reference = "absolute";
        double z = 0.0;
        bool present = false;
    };

    PoseInfo read_pose(const Json& node, const std::string& ptr) {
        PoseInfo info;
        const Json* pose = find(node, "pose");
        if (pose == nullptr) {
            return info;
        }
        info.present = true;
        const std::string pp = pointer(ptr, "pose");
        if (!pose->is_object()) {
            error(pp, "Objekt erwartet");
            return info;
        }
        if (const Json* r = find(*pose, "reference")) {
            info.reference = r->is_string() ? r->get<std::string>() : "";
            if (info.reference != "absolute" && info.reference != "relative_to_preceding" &&
                info.reference != "relative_to_sibling") {
                error(pointer(pp, "reference"), "unbekannter Bezug " + r->dump());
            }
        }
        if (const Json* pos = find(*pose, "position")) {
            const std::string pq = pointer(pp, "position");
            if (!pos->is_array() || pos->size() != 3) {
                error(pq, "Liste aus drei Zahlen erwartet");
            } else {
                for (std::size_t k = 0; k < 3; ++k) {
                    const auto v = param((*pos)[k], pointer(pq, k));
                    if (!v) {
                        continue;
                    }
                    if (k < 2 && *v != 0.0) {
                        error(pointer(pq, k), "Dezentrierung nicht abbildbar (der Adapter "
                                              "übernimmt nur koaxiale Systeme)");
                    }
                    if (k == 2) {
                        info.z = *v;
                    }
                }
            }
        }
        if (const Json* rot = find(*pose, "rotation_deg")) {
            const std::string pr = pointer(pp, "rotation_deg");
            if (!rot->is_array() || rot->size() != 3) {
                error(pr, "Liste aus drei Zahlen erwartet");
            } else {
                for (std::size_t k = 0; k < 3; ++k) {
                    const auto v = param((*rot)[k], pointer(pr, k));
                    if (v && *v != 0.0) {
                        error(pointer(pr, k), "Verkippung bzw. Drehung nicht abbildbar (der "
                                              "Adapter übernimmt nur koaxiale Systeme)");
                    }
                }
            }
        }
        return info;
    }

    // Lage des Knotens aus Pose und Bezug.
    Terms place(const PoseInfo& pose, const std::string& ptr, const Terms& parent,
                const Terms* sibling, const Terms& preceding, bool has_preceding) {
        Terms t;
        if (pose.reference == "relative_to_preceding") {
            if (!has_preceding) {
                error(pointer(ptr, "pose"), "relative_to_preceding ohne vorangehende Fläche");
            }
            t = preceding;
        } else if (pose.reference == "relative_to_sibling") {
            if (sibling == nullptr) {
                error(pointer(ptr, "pose"), "relative_to_sibling ohne vorangehendes Geschwister");
                t = parent;
            } else {
                t = *sibling;
            }
        } else {
            t = parent;
        }
        if (pose.present) {
            t.push_back(pose.z);
        }
        return t;
    }

    void check_active(const Json& node, const std::string& ptr, bool& active) {
        if (const Json* a = find(node, "active")) {
            if (a->is_boolean() && !a->get<bool>()) {
                if (active) {
                    warn(ptr, "inaktiver Knoten entfällt (er wird nicht durchlaufen)");
                }
                active = false;
            }
        }
    }

    // Durchläuft einen Knoten und liefert sein KS (für Geschwisterbezüge).
    Terms walk_node(const Json& node, const std::string& ptr, const Terms& parent,
                    const Terms* sibling, Terms& preceding, bool& has_preceding, bool active) {
        if (!node.is_object()) {
            error(ptr, "Objekt erwartet");
            return parent;
        }
        check_active(node, ptr, active);
        const PoseInfo pose = read_pose(node, ptr);
        const Terms frame = place(pose, ptr, parent, sibling, preceding, has_preceding);
        const Json* t = find(node, "type");
        const std::string type = t != nullptr && t->is_string() ? t->get<std::string>() : "";
        if (type == "assembly") {
            const Json* children = find(node, "children");
            if (children == nullptr || !children->is_array()) {
                error(pointer(ptr, "children"), "Liste erwartet");
                return frame;
            }
            Terms previous;
            for (std::size_t i = 0; i < children->size(); ++i) {
                const std::string cp = pointer(pointer(ptr, "children"), i);
                previous = walk_node((*children)[i], cp, frame, i > 0 ? &previous : nullptr,
                                     preceding, has_preceding, active);
            }
            return frame;
        }
        walk_element(node, ptr, type, frame, preceding, has_preceding, active);
        return frame;
    }

    std::optional<MaterialSpec> material(const std::string& ref, const std::string& ptr) {
        if (ref == "AIR" || ref == "VACUUM") {
            if ((ref == "VACUUM") == vacuum_) {
                return MaterialSpec{};
            }
            error(ptr, ref + " in der Umgebung " + (vacuum_ ? "VACUUM" : "AIR") +
                           " nicht abbildbar (Mikroptim: genau ein Medium Luft mit n = 1)");
            return std::nullopt;
        }
        if (ref.rfind("CONST:", 0) == 0) {
            const std::string value = ref.substr(6);
            double n = 0.0;
            if (value.find(',') != std::string::npos) {
                error(ptr, "komplexe Brechzahl " + ref + " nicht abbildbar");
                return std::nullopt;
            }
            if (!parse_number(value, n) || !(n > 0.0) || !std::isfinite(n)) {
                error(ptr, "ungültige konstante Brechzahl " + ref);
                return std::nullopt;
            }
            if (!vacuum_) {
                constant_in_air_ = true;
            }
            MaterialSpec m;
            m.kind = MaterialKind::Constant;
            m.n = n;
            return m;
        }
        const std::size_t colon = ref.find(':');
        const std::string catalog = colon == std::string::npos ? "" : ref.substr(0, colon);
        const std::string name = colon == std::string::npos ? ref : ref.substr(colon + 1);
        if (name.empty()) {
            error(ptr, "leerer Glasname in " + ref);
            return std::nullopt;
        }
        if (!options_.glass_exists) {
            error(ptr, "Katalogglas " + ref +
                           ": keine Mikroptim-Glaskataloge angegeben (--glass-catalog)");
            return std::nullopt;
        }
        if (!options_.glass_exists(name)) {
            error(ptr, "Glas " + name + " (" + ref + ") nicht in den Mikroptim-Glaskatalogen");
            return std::nullopt;
        }
        if (options_.glass_catalog_name) {
            const std::string own = options_.glass_catalog_name(name);
            if (!catalog.empty() && !own.empty() && own != catalog) {
                warn(ptr, "Glas " + name + " aus Katalog " + catalog +
                              " wird aus dem Mikroptim-Katalog " + own + " genommen");
            }
        }
        if (vacuum_) {
            catalog_in_vacuum_ = true;
        }
        MaterialSpec m;
        m.kind = MaterialKind::Catalog;
        m.name = name;
        return m;
    }

    // Form einer Fläche: Typ und Krümmung. Nur Ebene und Kugel (conic mit k = 0).
    void read_shape(const Json& surface, const std::string& ptr, Surface& s, bool plane_only) {
        s.type = SurfaceType::Plane;
        s.curvature = 0.0;
        const Json* shape = find(surface, "shape");
        if (shape == nullptr) {
            return;
        }
        const std::string sp = pointer(ptr, "shape");
        if (find(*shape, "terms") != nullptr && !(*shape)["terms"].empty()) {
            error(pointer(sp, "terms"), "Zernike-Terme nicht abbildbar");
        }
        const Json* base = find(*shape, "base");
        if (base == nullptr) {
            return;
        }
        const std::string bp = pointer(sp, "base");
        const Json* t = find(*base, "type");
        const std::string type = t != nullptr && t->is_string() ? t->get<std::string>() : "";
        if (type == "plane") {
            return;
        }
        if (type != "conic") {
            error(pointer(bp, "type"), "Flächenform " + (t != nullptr ? t->dump() : "?") +
                                           " nicht abbildbar (Mikroptim: Ebene und Kugel)");
            return;
        }
        if (find(*base, "conic") != nullptr) {
            const auto k = param((*base)["conic"], pointer(bp, "conic"));
            if (k && *k != 0.0) {
                error(pointer(bp, "conic"),
                      "Konik k = " + shortest(*k) + " nicht abbildbar (nur Kugeln, k = 0)");
            }
        }
        const Json* r = find(*base, "radius");
        if (r == nullptr) {
            error(pointer(bp, "radius"), "Radius fehlt");
            return;
        }
        const auto radius = param(*r, pointer(bp, "radius"));
        if (!radius) {
            return;
        }
        if (!std::isfinite(*radius) || *radius == 0.0) {
            error(pointer(bp, "radius"), "Radius endlich und ≠ 0 erwartet");
            return;
        }
        if (plane_only) {
            error(bp, "gekrümmte Fläche hier nicht abbildbar (Blende und Bildfläche sind eben)");
            return;
        }
        s.type = SurfaceType::Sphere;
        s.curvature = rtt_curvature_from_radius(*radius);
    }

    // Interaktion: fresnel (Standard) ohne Meldung; nur die Transmission betreffende Arten mit
    // Warnung; alles andere ist ein Fehler.
    void read_interaction(const Json& surface, const std::string& ptr) {
        if (const Json* ph = find(surface, "phases"); ph != nullptr && !ph->empty()) {
            error(pointer(ptr, "phases"), "Phasenschichten (Gitter, DOE) nicht abbildbar");
        }
        if (find(surface, "diffraction_efficiency") != nullptr) {
            error(pointer(ptr, "diffraction_efficiency"), "Beugungseffizienz nicht abbildbar");
        }
        if (const Json* ob = find(surface, "obscurations"); ob != nullptr && !ob->empty()) {
            error(pointer(ptr, "obscurations"), "Obskurationen nicht abbildbar");
        }
        const Json* in = find(surface, "interaction");
        if (in == nullptr) {
            return;
        }
        const std::string ip = pointer(ptr, "interaction");
        const Json* t = find(*in, "type");
        const std::string type = t != nullptr && t->is_string() ? t->get<std::string>() : "";
        if (type == "fresnel") {
            return;
        }
        if (type == "ideal_anti_reflection" || type == "coating") {
            warn(ip, "Interaktion " + type +
                         " entfällt (Mikroptim rechnet ohne Fresnel-Verluste "
                         "und Schichten)");
            return;
        }
        error(ip, "Interaktion " + (t != nullptr ? t->dump() : "?") + " nicht abbildbar");
    }

    std::optional<double> read_aperture_radius(const Json& surface, const std::string& ptr,
                                               bool required) {
        const Json* a = find(surface, "aperture");
        if (a == nullptr) {
            if (!required) {
                return std::nullopt;
            }
            if (options_.default_semi_diameter) {
                warn(ptr, "ohne Apertur: Halbdurchmesser " +
                              shortest(*options_.default_semi_diameter) + " mm angenommen");
                return options_.default_semi_diameter;
            }
            error(ptr, "Fläche ohne Apertur: Mikroptim braucht einen Halbdurchmesser "
                       "(--semi-diameter)");
            return std::nullopt;
        }
        const std::string ap = pointer(ptr, "aperture");
        const Json* t = find(*a, "type");
        if (t == nullptr || !t->is_string() || t->get<std::string>() != "circular") {
            error(ap, "Apertur " + (t != nullptr ? t->dump() : "?") +
                          " nicht abbildbar (Mikroptim: kreisförmiger Halbdurchmesser)");
            return std::nullopt;
        }
        if (find(*a, "inner_radius") != nullptr) {
            const auto inner = param((*a)["inner_radius"], pointer(ap, "inner_radius"));
            if (inner && *inner != 0.0) {
                error(pointer(ap, "inner_radius"), "Ringapertur nicht abbildbar");
            }
        }
        const Json* r = find(*a, "radius");
        if (r == nullptr) {
            error(pointer(ap, "radius"), "Radius fehlt");
            return std::nullopt;
        }
        const auto radius = param(*r, pointer(ap, "radius"));
        if (radius && !(*radius > 0.0)) {
            error(pointer(ap, "radius"), "Radius > 0 erwartet");
            return std::nullopt;
        }
        return radius;
    }

    void walk_element(const Json& node, const std::string& ptr, const std::string& type,
                      const Terms& frame, Terms& preceding, bool& has_preceding, bool active) {
        ElementKind kind = ElementKind::Lens;
        if (type == "stop") {
            kind = ElementKind::Stop;
        } else if (type == "lens" || type == "plate") {
            kind = ElementKind::Lens;
        } else if (type == "thin_element") {
            kind = ElementKind::Thin;
        } else if (type == "detector") {
            kind = ElementKind::Detector;
        } else if (type == "mirror") {
            if (active) {
                error(pointer(ptr, "type"), "Spiegel nicht abbildbar (Mikroptim: nur "
                                            "brechende Flächen)");
                return;
            }
            kind = ElementKind::Thin; // inaktiv: nur die Lagen der Flächen zählen
        } else {
            error(pointer(ptr, "type"),
                  "unbekannter Knotentyp " + (find(node, "type") ? node["type"].dump() : "?"));
            return;
        }
        const Json* surfaces = find(node, "surfaces");
        if (surfaces == nullptr || !surfaces->is_array() || surfaces->empty()) {
            error(pointer(ptr, "surfaces"), "mindestens eine Fläche erwartet");
            return;
        }
        const std::size_t n = surfaces->size();
        if (active) {
            if (kind == ElementKind::Lens && n < 2) {
                error(pointer(ptr, "surfaces"), "Linse bzw. Platte mit mindestens zwei Flächen "
                                                "erwartet");
            }
            if (kind != ElementKind::Lens && n != 1) {
                error(pointer(ptr, "surfaces"), "Element " + type +
                                                    " mit genau einer Fläche "
                                                    "erwartet");
            }
            if (find(node, "optic_axis") != nullptr) {
                error(pointer(ptr, "optic_axis"), "Kristalle nicht abbildbar");
            }
        }

        // Medien der Segmente (Linse und Platte).
        std::vector<MaterialSpec> segments(n > 0 ? n - 1 : 0);
        const Json* mat = find(node, "material");
        if (active && kind == ElementKind::Lens) {
            const std::string mp = pointer(ptr, "material");
            if (mat == nullptr) {
                error(mp, "Material fehlt");
            } else if (mat->is_string()) {
                const auto m = material(mat->get<std::string>(), mp);
                for (MaterialSpec& s : segments) {
                    if (m) {
                        s = *m;
                    }
                }
            } else if (mat->is_array()) {
                if (mat->size() != segments.size()) {
                    error(mp, "ein Material je Segment erwartet (" +
                                  std::to_string(segments.size()) + ")");
                } else {
                    for (std::size_t k = 0; k < segments.size(); ++k) {
                        const std::string kp = pointer(mp, k);
                        if (!(*mat)[k].is_string()) {
                            error(kp, "Zeichenkette erwartet");
                        } else if (const auto m = material((*mat)[k].get<std::string>(), kp)) {
                            segments[k] = *m;
                        }
                    }
                }
            } else {
                error(mp, "Kristalle bzw. Materialobjekte nicht abbildbar");
            }
        } else if (active && mat != nullptr && kind != ElementKind::Lens) {
            error(pointer(ptr, "material"), "Material an " + type + " nicht abbildbar");
        }

        Terms previous_surface;
        for (std::size_t k = 0; k < n; ++k) {
            const Json& surface = (*surfaces)[k];
            const std::string sp = pointer(pointer(ptr, "surfaces"), k);
            if (!surface.is_object()) {
                error(sp, "Objekt erwartet");
                continue;
            }
            const PoseInfo pose = read_pose(surface, sp);
            if (k == 0 && pose.present && pose.reference != "absolute") {
                error(pointer(sp, "pose"), "erste Fläche eines Elements relativ platziert");
            }
            const Terms sf = place(pose, sp, frame, k > 0 ? &previous_surface : nullptr, preceding,
                                   has_preceding);
            previous_surface = sf;
            preceding = sf;
            has_preceding = true;
            if (!active) {
                continue;
            }
            SurfaceRec rec;
            rec.ptr = sp;
            rec.frame = sf;
            rec.kind = kind;
            rec.active = active;
            read_interaction(surface, sp);
            const bool plane_only = kind == ElementKind::Stop || kind == ElementKind::Detector;
            read_shape(surface, sp, rec.surface, plane_only);
            if (kind == ElementKind::Stop) {
                rec.surface.type = SurfaceType::Stop;
                stop_ptr_ = sp;
                if (const auto r = read_aperture_radius(surface, sp, false)) {
                    stop_radius_ = *r;
                }
            } else if (kind == ElementKind::Detector) {
                if (find(surface, "aperture") != nullptr) {
                    warn(pointer(sp, "aperture"), "Apertur der Bildfläche entfällt");
                }
            } else if (const auto r = read_aperture_radius(surface, sp, true)) {
                rec.surface.semi_diameter = *r;
            }
            if (kind == ElementKind::Lens && k + 1 < n) {
                rec.surface.material = segments[k];
            }
            if (const Json* id = find(surface, "id"); id != nullptr && id->is_string()) {
                rec.surface.comment = id->get<std::string>();
            }
            records_.push_back(std::move(rec));
        }
    }

    void build_surfaces(SystemDescription& sys) {
        if (constant_in_air_) {
            warn("/environment", "Raytatouille rechnet Brechzahlen absolut (Luft nach Ciddor, "
                                 "n ≈ 1,0003); CONST-Werte werden als Brechzahl relativ zu Luft "
                                 "übernommen. Für gleiche Paraxialdaten die Umgebung VACUUM "
                                 "verwenden");
        }
        if (catalog_in_vacuum_) {
            warn("/environment/medium",
                 "Katalogglas in der Umgebung VACUUM: Raytatouille rechnet n_rel · n_Luft "
                 "gegen n = 1, Mikroptim n_rel gegen Luft (Abweichung etwa 3·10⁻⁴)");
        }
        if (records_.empty()) {
            error("/root", "keine Flächen");
            return;
        }
        if (records_.front().kind != ElementKind::Stop) {
            error(records_.front().ptr,
                  "die erste Fläche ist keine Blende (Mikroptim: Aperturblende als erste "
                  "Fläche in Luft, ADR-0006)");
        }
        std::size_t detectors = 0;
        for (std::size_t i = 0; i < records_.size(); ++i) {
            const SurfaceRec& r = records_[i];
            if (r.kind == ElementKind::Stop && i != 0) {
                error(r.ptr, "nur eine Blende als erste Fläche abbildbar");
            }
            if (r.kind == ElementKind::Detector) {
                ++detectors;
                if (i + 1 != records_.size()) {
                    error(records_[i + 1].ptr, "Fläche hinter der Bildfläche nicht abbildbar");
                }
            }
        }
        if (detectors == 0) {
            error("/root", "Bildfläche (detector) fehlt");
        }
        if (!issues_.empty()) {
            return;
        }
        if (records_.size() < 3) {
            error("/root", "mindestens eine Fläche zwischen Blende und Bildfläche erwartet");
            return;
        }
        if (!pupil_radius_) {
            return;
        }
        for (std::size_t i = 0; i + 1 < records_.size(); ++i) {
            Surface s = records_[i].surface;
            s.thickness = distance(records_[i].frame, records_[i + 1].frame);
            if (i == 0) {
                s.semi_diameter = *pupil_radius_;
            }
            sys.surfaces.push_back(std::move(s));
        }
    }

    const RttImportOptions& options_;
    std::vector<std::string>* warnings_;
    std::vector<RttIssue> issues_;
    const Json* root_ = nullptr;
    std::vector<std::pair<std::string, std::size_t>> parameters_;
    std::vector<SurfaceRec> records_;
    bool vacuum_ = false;
    bool constant_in_air_ = false;
    bool catalog_in_vacuum_ = false;
    std::optional<double> pupil_radius_;
    std::optional<double> stop_radius_;
    std::string stop_ptr_;
};

// ---- Ausgabe ----------------------------------------------------------------------------

constexpr std::size_t kLineWidth = 100;

// Einzeilige Darstellung: `{"a": 1, "b": [1.0, 2.0]}`.
std::string inline_text(const Json& j) {
    if (j.is_object() && !j.empty()) {
        std::string s = "{";
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) {
                s += ", ";
            }
            first = false;
            s += Json(it.key()).dump() + ": " + inline_text(it.value());
        }
        return s + "}";
    }
    if (j.is_array() && !j.empty()) {
        std::string s = "[";
        for (std::size_t i = 0; i < j.size(); ++i) {
            if (i > 0) {
                s += ", ";
            }
            s += inline_text(j[i]);
        }
        return s + "]";
    }
    return j.dump();
}

// Eingerückte Ausgabe; Listen und Objekte bleiben einzeilig, wenn sie in die Zeile passen.
void pretty(const Json& j, std::size_t indent, std::size_t column, bool expand, std::string& out) {
    if (!j.is_structured() || j.empty()) {
        out += inline_text(j);
        return;
    }
    if (!expand) {
        const std::string s = inline_text(j);
        if (column + s.size() + 1 <= kLineWidth) {
            out += s;
            return;
        }
    }
    const std::string inner(indent + 2, ' ');
    out += j.is_object() ? "{\n" : "[\n";
    std::size_t i = 0;
    for (auto it = j.begin(); it != j.end(); ++it, ++i) {
        out += inner;
        std::size_t col = inner.size();
        if (j.is_object()) {
            const std::string key = Json(it.key()).dump() + ": ";
            out += key;
            col += key.size();
        }
        pretty(it.value(), indent + 2, col, false, out);
        if (i + 1 < j.size()) {
            out += ',';
        }
        out += '\n';
    }
    out += std::string(indent, ' ');
    out += j.is_object() ? '}' : ']';
}

// ---- Mikroptim → Raytatouille ------------------------------------------------------------

std::string at(const std::string& path, std::size_t i) {
    return path + "[" + std::to_string(i) + "]";
}

bool id_like(const std::string& s) {
    if (s.empty() || s.size() > 64) {
        return false;
    }
    for (const char ch : s) {
        const bool ok = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                        (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-' ||
                        ch == '+' || ch == ':';
        if (!ok) {
            return false;
        }
    }
    return true;
}

class Exporter {
public:
    Exporter(const RttExportOptions& options, std::vector<std::string>* warnings)
        : options_(options), warnings_(warnings) {}

    std::string run(const SystemDescription& sys) {
        validate_system(sys);
        check(sys);
        if (!issues_.empty()) {
            throw RttConversionError(issues_);
        }
        Json root;
        root["schema_version"] = std::string(kRttSchemaVersion);
        root["name"] = sys.name;
        root["units"] = Json{{"length", "mm"}, {"wavelength", "um"}};
        if (vacuum_) {
            root["environment"] = Json{{"medium", "VACUUM"}};
        }
        root["wavelengths"] = wavelengths(sys);
        root["aperture"] = Json{{"type", "epd"}, {"value", 2.0 * sys.surfaces[0].semi_diameter}};
        root["fields"] = fields(sys);
        root["root"] = tree(sys);
        root["paths"] = Json::array({Json{{"name", "main"}, {"events", "auto"}}});
        if (!issues_.empty()) {
            throw RttConversionError(issues_);
        }
        std::string out;
        pretty(root, 0, 0, true, out);
        out += '\n';
        return out;
    }

private:
    void error(const std::string& path, std::string message) {
        issues_.push_back({path, std::move(message)});
    }
    void warn(const std::string& path, const std::string& message) {
        if (warnings_ != nullptr) {
            warnings_->push_back((path.empty() ? std::string("(System)") : path) + ": " + message);
        }
    }

    void check(const SystemDescription& sys) {
        if (sys.source.type != SourceType::Collimated) {
            error("source.type", "Punktquelle nicht abbildbar (Raytatouille-Adapter: Objekt im "
                                 "Unendlichen, kollimierte Quelle)");
        }
        bool catalog = false;
        bool constant = false;
        for (std::size_t i = 0; i < sys.surfaces.size(); ++i) {
            const Surface& s = sys.surfaces[i];
            const std::string p = at("surfaces", i);
            if (s.decenter[0] != 0.0 || s.decenter[1] != 0.0) {
                error(p + ".decenter", "Dezentrierung nicht abbildbar (der Adapter übernimmt "
                                       "nur koaxiale Systeme)");
            }
            if (s.tilt_deg[0] != 0.0 || s.tilt_deg[1] != 0.0) {
                error(p + ".tilt_deg", "Verkippung nicht abbildbar (der Adapter übernimmt nur "
                                       "koaxiale Systeme)");
            }
            if (s.group) {
                warn(p + ".group", "Gruppe entfällt (Raytatouille hat kein Feld dafür)");
            }
            switch (s.material.kind) {
            case MaterialKind::Buchdahl:
                error(p + ".material", "Buchdahl-Modellglas nicht abbildbar (Raytatouille kennt "
                                       "kein Buchdahl-Material; vorher auf Katalog-Gläser "
                                       "einrasten, mikroptim snap)");
                break;
            case MaterialKind::Catalog:
                catalog = true;
                if (!options_.glass_catalog_name ||
                    options_.glass_catalog_name(s.material.name).empty()) {
                    error(p + ".material", "Katalog des Glases " + s.material.name +
                                               " unbekannt (Referenz KATALOG:NAME)");
                }
                break;
            case MaterialKind::Constant:
                constant = true;
                break;
            case MaterialKind::Air:
                break;
            }
        }
        if (sys.surfaces.back().material.kind != MaterialKind::Air) {
            error(at("surfaces", sys.surfaces.size() - 1) + ".material",
                  "Bildraum ist nicht Luft (Immersion) nicht abbildbar");
        }
        for (std::size_t g = 0; g < sys.ghost_plates.size(); ++g) {
            bool zero = true;
            for (const double a : sys.ghost_plates[g].coefficients) {
                zero = zero && a == 0.0;
            }
            if (!zero) {
                error(at("ghost_plates", g), "Geisterplatte mit Koeffizienten ≠ 0 nicht "
                                             "abbildbar (Raytatouille kennt keine Geisterplatten)");
            }
        }
        if (!sys.ghost_plates.empty() && issues_.empty()) {
            warn("ghost_plates", "Geisterplatten ohne Wirkung (alle Koeffizienten 0) entfallen");
        }
        if (sys.description) {
            warn("description", "Beschreibung entfällt (Raytatouille hat kein Feld dafür)");
        }
        if (sys.target) {
            warn("target", "Sollwerte entfallen (in Raytatouille als Operanden der "
                           "Merit-Funktion anlegen)");
        }
        if (sys.variables) {
            warn("variables", "Optimierungsvariablen entfallen");
        }
        if (sys.constraints) {
            warn("constraints", "Nebenbedingungen entfallen");
        }
        if (sys.tolerances) {
            warn("tolerances", "Toleranzsatz entfällt (Raytatouille hat ein eigenes, anders "
                               "aufgebautes Toleranzformat)");
        }
        // Umgebung: VACUUM, solange kein Katalogglas vorkommt (dann rechnen beide Programme
        // mit denselben Zahlen); mit Katalogglas Luft (Raytatouille rechnet Katalogbrechzahlen
        // relativ zur Luft der Umgebung).
        vacuum_ = !catalog;
        if (catalog && constant) {
            warn("surfaces", "Katalogglas und konstante Brechzahlen gemischt: Umgebung Luft; "
                             "Raytatouille rechnet CONST absolut, die Brechzahlen relativ zu "
                             "Luft weichen um den Faktor 1/n_Luft ≈ 0,9997 ab");
        }
    }

    Json wavelengths(const SystemDescription& sys) {
        Json list = Json::array();
        bool reference = false;
        for (std::size_t i = 0; i < sys.spectrum.wavelengths_nm.size(); ++i) {
            const double nm = sys.spectrum.wavelengths_nm[i];
            Json w;
            w["um"] = rtt_nm_to_um(nm);
            if (sys.spectrum.weights[i] != 1.0) {
                w["weight"] = sys.spectrum.weights[i];
            }
            if (!reference && nm == sys.spectrum.reference_nm) {
                w["reference"] = true;
                reference = true;
            }
            list.push_back(std::move(w));
        }
        if (!reference) {
            warn("spectrum.reference_nm",
                 "Referenzwellenlänge " + shortest(sys.spectrum.reference_nm) +
                     " nm nicht unter wavelengths_nm; als Wellenlänge mit Gewicht 0 angehängt "
                     "(Raytatouille verlangt die Referenz in der Liste)");
            list.push_back(Json{{"um", rtt_nm_to_um(sys.spectrum.reference_nm)},
                                {"weight", 0.0},
                                {"reference", true}});
        }
        return list;
    }

    static Json fields(const SystemDescription& sys) {
        Json points = Json::array();
        for (std::size_t i = 0; i < sys.source.field_angles_deg.size(); ++i) {
            Json p = Json::object();
            if (sys.source.field_angles_deg[i] != 0.0) {
                p["y"] = sys.source.field_angles_deg[i];
            }
            if (sys.source.field_weights[i] != 1.0) {
                p["weight"] = sys.source.field_weights[i];
            }
            points.push_back(std::move(p));
        }
        return Json{{"type", "angle_deg"}, {"points", std::move(points)}};
    }

    std::string material_ref(const MaterialSpec& m) {
        switch (m.kind) {
        case MaterialKind::Constant:
            return "CONST:" + shortest(m.n);
        case MaterialKind::Catalog:
            return options_.glass_catalog_name(m.name) + ":" + m.name;
        case MaterialKind::Air:
            return vacuum_ ? "VACUUM" : "AIR";
        case MaterialKind::Buchdahl:
            break;
        }
        return "";
    }

    std::string unique(std::string id) {
        if (ids_.insert(id).second) {
            return id;
        }
        for (int k = 1;; ++k) {
            const std::string c = id + "_" + std::to_string(k);
            if (ids_.insert(c).second) {
                return c;
            }
        }
    }

    Json surface_json(const SystemDescription& sys, std::size_t i, const std::string& id) {
        const Surface& s = sys.surfaces[i];
        Json j;
        j["id"] = id;
        if (s.type == SurfaceType::Sphere) {
            if (s.curvature == 0.0) {
                warn(at("surfaces", i), "Kugel mit Krümmung 0 wird als Planfläche geschrieben");
            } else {
                const double r = rtt_radius_from_curvature(s.curvature);
                if (rtt_curvature_from_radius(r) != s.curvature) {
                    warn(at("surfaces", i) + ".curvature",
                         "kein Radius R mit 1/R = c bitgleich; 1/R weicht um etwa 1 ulp "
                         "(relativ ≈ 10⁻¹⁶) ab");
                }
                j["shape"] = Json{{"base", Json{{"type", "conic"}, {"radius", r}}}};
            }
        }
        j["aperture"] = Json{{"type", "circular"}, {"radius", s.semi_diameter}};
        return j;
    }

    // Flächen-ID aus dem Kommentar, wenn er wie eine ID aussieht, sonst erzeugt.
    std::string surface_id(const SystemDescription& sys, std::size_t i, const std::string& made) {
        const auto& c = sys.surfaces[i].comment;
        if (c && id_like(*c) && ids_.count(*c) == 0) {
            return unique(*c);
        }
        if (c) {
            warn(at("surfaces", i) + ".comment",
                 "Kommentar entfällt (keine gültige oder eindeutige Flächen-ID)");
        }
        return unique(made);
    }

    static Json pose_after(double thickness) {
        return Json{{"reference", "relative_to_preceding"},
                    {"position", Json::array({0.0, 0.0, thickness})}};
    }

    // Elementname aus dem gemeinsamen Präfix der Flächen-IDs (`L1.S1`, `L1.S2` → `L1`).
    std::string element_name(const std::vector<std::size_t>& surfaces, const SystemDescription& sys,
                             const std::string& made) {
        std::string prefix;
        bool ok = true;
        for (const std::size_t i : surfaces) {
            const auto& c = sys.surfaces[i].comment;
            const std::size_t dot = c ? c->find('.') : std::string::npos;
            if (!c || !id_like(*c) || dot == std::string::npos || dot == 0) {
                ok = false;
                break;
            }
            const std::string p = c->substr(0, dot);
            if (prefix.empty()) {
                prefix = p;
            } else if (p != prefix) {
                ok = false;
                break;
            }
        }
        if (ok && !prefix.empty() && names_.insert(prefix).second) {
            return prefix;
        }
        std::string name = made;
        for (int k = 1; !names_.insert(name).second; ++k) {
            name = made + "_" + std::to_string(k);
        }
        return name;
    }

    Json tree(const SystemDescription& sys) {
        Json children = Json::array();
        names_.insert("stop");
        names_.insert("image");
        // Kommentare, die als IDs taugen, zuerst reservieren, damit erzeugte IDs ausweichen.
        Json stop;
        stop["type"] = "stop";
        stop["name"] = "stop";
        Json stop_surface;
        stop_surface["id"] = surface_id(sys, 0, "STO");
        stop_surface["aperture"] =
            Json{{"type", "circular"}, {"radius", sys.surfaces[0].semi_diameter}};
        stop["surfaces"] = Json::array({std::move(stop_surface)});
        children.push_back(std::move(stop));

        int lenses = 0;
        int dummies = 0;
        const std::size_t count = sys.surfaces.size();
        std::size_t i = 1;
        while (i < count) {
            std::vector<std::size_t> group{i};
            while (sys.surfaces[group.back()].material.kind != MaterialKind::Air) {
                group.push_back(group.back() + 1);
            }
            const bool lens = group.size() > 1;
            const std::string made =
                lens ? "L" + std::to_string(++lenses) : "D" + std::to_string(++dummies);
            const std::string name = element_name(group, sys, made);
            Json element;
            element["type"] = lens ? "lens" : "thin_element";
            element["name"] = name;
            element["pose"] = pose_after(sys.surfaces[i - 1].thickness);
            if (lens) {
                std::vector<std::string> mats;
                for (std::size_t k = 0; k + 1 < group.size(); ++k) {
                    mats.push_back(material_ref(sys.surfaces[group[k]].material));
                }
                bool same = true;
                for (const std::string& m : mats) {
                    same = same && m == mats.front();
                }
                element["material"] = same ? Json(mats.front()) : Json(mats);
            }
            Json surfaces = Json::array();
            for (std::size_t k = 0; k < group.size(); ++k) {
                const std::size_t s = group[k];
                Json sj =
                    surface_json(sys, s, surface_id(sys, s, name + ".S" + std::to_string(k + 1)));
                if (k > 0) {
                    // Lage gleich hinter "id", der Rest folgt.
                    Json ordered;
                    ordered["id"] = sj["id"];
                    ordered["pose"] = pose_after(sys.surfaces[s - 1].thickness);
                    for (auto it = sj.begin(); it != sj.end(); ++it) {
                        if (it.key() != "id") {
                            ordered[it.key()] = it.value();
                        }
                    }
                    sj = std::move(ordered);
                }
                surfaces.push_back(std::move(sj));
            }
            element["surfaces"] = std::move(surfaces);
            children.push_back(std::move(element));
            i = group.back() + 1;
        }

        Json detector;
        detector["type"] = "detector";
        detector["name"] = "image";
        detector["pose"] = pose_after(sys.surfaces.back().thickness);
        detector["surfaces"] = Json::array({Json{{"id", unique("IMG")}}});
        children.push_back(std::move(detector));
        return Json{{"type", "assembly"}, {"name", "system"}, {"children", std::move(children)}};
    }

    const RttExportOptions& options_;
    std::vector<std::string>* warnings_;
    std::vector<RttIssue> issues_;
    bool vacuum_ = false;
    std::set<std::string> ids_;
    std::set<std::string> names_;
};

std::string compose_issues(const std::vector<RttIssue>& issues) {
    std::string s = "nicht abbildbar (" + std::to_string(issues.size()) + " Stelle" +
                    (issues.size() == 1 ? "" : "n") + "):";
    for (const RttIssue& i : issues) {
        s += "\n  " + i.location + ": " + i.message;
    }
    return s;
}

} // namespace

RttConversionError::RttConversionError(std::vector<RttIssue> issues)
    : std::runtime_error(compose_issues(issues)), issues_(std::move(issues)) {}

double rtt_curvature_from_radius(double radius) {
    return 1.0 / radius;
}

double rtt_radius_from_curvature(double curvature) {
    return exact_preimage(curvature, 1.0 / curvature,
                          [](double r) { return rtt_curvature_from_radius(r); });
}

double rtt_um_to_nm(double um) {
    return shift_decimal(um, 3);
}

double rtt_nm_to_um(double nm) {
    return exact_preimage(nm, shift_decimal(nm, -3), [](double um) { return rtt_um_to_nm(um); });
}

SystemDescription system_from_rtt_json(std::string_view rtt_text, const RttImportOptions& options,
                                       std::vector<std::string>* warnings) {
    Json root;
    try {
        root = Json::parse(rtt_text.begin(), rtt_text.end());
    } catch (const Json::exception& e) {
        throw FormatError("", std::string("JSON-Syntaxfehler: ") + e.what());
    }
    try {
        return Importer(options, warnings).run(root);
    } catch (const Json::exception& e) {
        throw RttConversionError({{"/", std::string("JSON-Fehler: ") + e.what()}});
    }
}

std::string system_to_rtt_json(const SystemDescription& system, const RttExportOptions& options,
                               std::vector<std::string>* warnings) {
    try {
        return Exporter(options, warnings).run(system);
    } catch (const Json::exception& e) {
        throw FormatError("", std::string("JSON-Fehler beim Schreiben: ") + e.what());
    }
}

} // namespace mko
