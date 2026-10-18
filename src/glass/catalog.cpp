// Laden von Glaskatalogen im Format `mikroptim.glass/1` (docs/dateiformat.md).

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/dispersion.hpp"

#include "format_util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <utility>

namespace mko {

namespace {

using nlohmann::json;

constexpr std::string_view kFormatName = "mikroptim.glass";
constexpr std::string_view kFormat = "mikroptim.glass/1";
// Materialschlüsselwort der Systemdatei; darf kein Katalogname sein.
constexpr std::string_view kReservedAir = "AIR";

// Sammelt Fehler mit Herkunft (Datei) und JSON-Pfad.
class Reader {
public:
    explicit Reader(std::string origin) : origin_(std::move(origin)) {}

    [[noreturn]] void fail(const std::string& path, const std::string& message) const {
        throw GlassError(origin_ + ": " + (path.empty() ? std::string{} : path + ": ") + message);
    }

    const json& member(const json& obj, const std::string& path, const char* key) const {
        const auto it = obj.find(key);
        if (it == obj.end()) {
            fail(join(path, key), "Pflichtfeld fehlt");
        }
        return *it;
    }

    std::string string_value(const json& v, const std::string& path) const {
        if (!v.is_string()) {
            fail(path, "Zeichenkette erwartet");
        }
        return v.get<std::string>();
    }

    double number_value(const json& v, const std::string& path) const {
        if (!v.is_number()) {
            fail(path, "Zahl erwartet");
        }
        const double x = v.get<double>();
        if (!std::isfinite(x)) {
            fail(path, "endliche Zahl erwartet");
        }
        return x;
    }

    template <std::size_t N>
    std::array<double, N> number_array(const json& v, const std::string& path) const {
        if (!v.is_array() || v.size() != N) {
            fail(path, "Feld mit genau " + std::to_string(N) + " Zahlen erwartet");
        }
        std::array<double, N> out{};
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = number_value(v[i], path + "[" + std::to_string(i) + "]");
        }
        return out;
    }

    void warn_unknown(const json& obj, const std::string& path,
                      std::initializer_list<std::string_view> known,
                      std::vector<std::string>& warnings) const {
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            if (std::find(known.begin(), known.end(), it.key()) == known.end()) {
                warnings.push_back(origin_ + ": " + join(path, it.key()) +
                                   ": unbekanntes Feld ignoriert");
            }
        }
    }

    static std::string join(const std::string& path, std::string_view key) {
        return path.empty() ? std::string(key) : path + "." + std::string(key);
    }

private:
    std::string origin_;
};

void check_format(const Reader& rd, const json& root) {
    const std::string fmt = rd.string_value(rd.member(root, "", "format"), "format");
    if (fmt == kFormat) {
        return;
    }
    const std::string prefix = std::string(kFormatName) + "/";
    if (fmt.rfind(prefix, 0) == 0) {
        rd.fail("format", "unbekannte Hauptversion '" + fmt.substr(prefix.size()) +
                              "' (unterstützt: " + std::string(kFormat) + ")");
    }
    rd.fail("format",
            "'" + fmt + "' ist kein Glaskatalog (erwartet: " + std::string(kFormat) + ")");
}

Glass read_glass(const Reader& rd, const json& g, const std::string& path,
                 std::vector<std::string>& warnings) {
    if (!g.is_object()) {
        rd.fail(path, "Objekt erwartet");
    }
    Glass out;
    out.name = rd.string_value(rd.member(g, path, "name"), path + ".name");
    if (out.name.empty()) {
        rd.fail(path + ".name", "Name darf nicht leer sein");
    }
    if (out.name == kReservedAir) {
        rd.fail(path + ".name", "'AIR' ist als Materialname reserviert");
    }

    const std::string model = rd.string_value(rd.member(g, path, "model"), path + ".model");
    if (model != "sellmeier") {
        rd.fail(path + ".model", "unbekanntes Modell '" + model + "' (unterstützt: sellmeier)");
    }
    out.B = rd.number_array<3>(rd.member(g, path, "B"), path + ".B");
    out.C = rd.number_array<3>(rd.member(g, path, "C"), path + ".C");
    out.range_nm = rd.number_array<2>(rd.member(g, path, "range_nm"), path + ".range_nm");
    if (!(out.range_nm[0] > 0.0 && out.range_nm[0] < out.range_nm[1])) {
        rd.fail(path + ".range_nm", "0 < min < max verletzt");
    }

    // Kein Sellmeier-Pol (lambda^2 = C_i) im Gültigkeitsbereich.
    for (std::size_t i = 0; i < 3; ++i) {
        if (out.C[i] > 0.0) {
            const double pole_nm = std::sqrt(out.C[i]) * 1000.0;
            if (pole_nm >= out.range_nm[0] && pole_nm <= out.range_nm[1]) {
                rd.fail(path + ".C[" + std::to_string(i) + "]",
                        "Sellmeier-Pol bei " + detail::format_number(pole_nm) +
                            " nm liegt im Gültigkeitsbereich");
            }
        }
    }
    for (const double lambda_nm : out.range_nm) {
        const double n = sellmeier_index(out.B.data(), out.C.data(), nm_to_um(lambda_nm));
        if (!(std::isfinite(n) && n > 0.0)) {
            rd.fail(path, "Sellmeier-Formel liefert bei " + detail::format_number(lambda_nm) +
                              " nm keine gültige Brechzahl");
        }
    }

    if (const auto it = g.find("reference"); it != g.end()) {
        const std::string rpath = path + ".reference";
        if (!it->is_object()) {
            rd.fail(rpath, "Objekt erwartet");
        }
        if (const auto nd = it->find("n_d"); nd != it->end()) {
            out.reference_n_d = rd.number_value(*nd, rpath + ".n_d");
        }
        rd.warn_unknown(*it, rpath, {"n_d"}, warnings);
    }
    if (const auto it = g.find("source"); it != g.end()) {
        out.source = rd.string_value(*it, path + ".source");
    }
    // `transmission` und `comment` sind bekannt, werden aber (noch) nicht ausgewertet.
    rd.warn_unknown(
        g, path,
        {"name", "model", "B", "C", "range_nm", "reference", "transmission", "source", "comment"},
        warnings);
    return out;
}

std::string path_to_utf8(const std::filesystem::path& p) {
    const auto u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}

} // namespace

bool Glass::in_range(double lambda_nm) const {
    return lambda_nm >= range_nm[0] && lambda_nm <= range_nm[1];
}

double Glass::index(double lambda_nm) const {
    if (!in_range(lambda_nm)) {
        throw GlassError("Glas '" + name + "': Wellenlänge " + detail::format_number(lambda_nm) +
                         " nm liegt außerhalb des Gültigkeitsbereichs [" +
                         detail::format_number(range_nm[0]) + ", " +
                         detail::format_number(range_nm[1]) + "] nm");
    }
    return sellmeier_index(B.data(), C.data(), nm_to_um(lambda_nm));
}

GlassCatalog GlassCatalog::from_string(std::string_view json_text, std::string_view origin) {
    const Reader rd{std::string(origin)};
    json root;
    try {
        root = json::parse(json_text.begin(), json_text.end());
    } catch (const json::parse_error& e) {
        rd.fail("", std::string("JSON-Syntaxfehler: ") + e.what());
    }
    if (!root.is_object()) {
        rd.fail("", "Objekt auf oberster Ebene erwartet");
    }
    check_format(rd, root);

    GlassCatalog cat;
    const json& glasses = rd.member(root, "", "glasses");
    if (!glasses.is_array()) {
        rd.fail("glasses", "Feld (Array) erwartet");
    }
    for (std::size_t i = 0; i < glasses.size(); ++i) {
        const std::string path = "glasses[" + std::to_string(i) + "]";
        Glass g = read_glass(rd, glasses[i], path, cat.warnings_);
        const auto dup = std::find_if(cat.glasses_.begin(), cat.glasses_.end(),
                                      [&](const Glass& x) { return x.name == g.name; });
        if (dup != cat.glasses_.end()) {
            rd.fail(path + ".name", "doppelter Name '" + g.name + "' (bereits in glasses[" +
                                        std::to_string(dup - cat.glasses_.begin()) + "])");
        }
        cat.glasses_.push_back(std::move(g));
    }
    rd.warn_unknown(root, "", {"format", "description", "glasses", "source", "comment"},
                    cat.warnings_);
    return cat;
}

GlassCatalog GlassCatalog::from_file(const std::filesystem::path& path) {
    const std::string origin = path_to_utf8(path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw GlassError(origin + ": Datei kann nicht geöffnet werden");
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    if (in.bad()) {
        throw GlassError(origin + ": Lesefehler");
    }
    return from_string(buf.str(), origin);
}

void GlassCatalog::add(Glass glass) {
    if (glass.name.empty()) {
        throw GlassError("Glasname darf nicht leer sein");
    }
    if (glass.name == kReservedAir) {
        throw GlassError("'AIR' ist als Materialname reserviert");
    }
    if (contains(glass.name)) {
        throw GlassError("doppelter Glasname '" + glass.name + "'");
    }
    glasses_.push_back(std::move(glass));
}

void GlassCatalog::merge(const GlassCatalog& other) {
    for (const Glass& g : other.glasses_) {
        if (contains(g.name)) {
            throw GlassError("Kataloge enthalten doppelten Glasnamen '" + g.name + "'");
        }
    }
    glasses_.insert(glasses_.end(), other.glasses_.begin(), other.glasses_.end());
    warnings_.insert(warnings_.end(), other.warnings_.begin(), other.warnings_.end());
}

const Glass* GlassCatalog::find(std::string_view name) const {
    const auto it = std::find_if(glasses_.begin(), glasses_.end(),
                                 [&](const Glass& g) { return g.name == name; });
    return it == glasses_.end() ? nullptr : &*it;
}

const Glass& GlassCatalog::at(std::string_view name) const {
    const Glass* g = find(name);
    if (g == nullptr) {
        throw GlassError("Glas '" + std::string(name) + "' ist in keinem geladenen Katalog");
    }
    return *g;
}

} // namespace mko
