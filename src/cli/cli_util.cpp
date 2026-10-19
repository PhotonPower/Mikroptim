// Gemeinsame Bausteine der Unterbefehle: Zahlen aus Argumenten, Textformatierung, Pfade.

#include "cli/cli_util.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <system_error>

namespace mko::cli {

namespace {

template <typename T>
bool parse_number(std::string_view text, T& value) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    return !text.empty() && ec == std::errc{} && ptr == last;
}

} // namespace

int parse_int(std::string_view option, std::string_view text) {
    int value = 0;
    if (!parse_number(text, value)) {
        throw UsageError(std::string(option) + " erwartet eine ganze Zahl, nicht \"" +
                         std::string(text) + "\"");
    }
    return value;
}

std::uint64_t parse_uint64(std::string_view option, std::string_view text) {
    std::uint64_t value = 0;
    if (!parse_number(text, value)) {
        throw UsageError(std::string(option) + " erwartet eine ganze Zahl >= 0, nicht \"" +
                         std::string(text) + "\"");
    }
    return value;
}

double parse_double(std::string_view option, std::string_view text) {
    double value = 0.0;
    if (!parse_number(text, value) || !std::isfinite(value)) {
        throw UsageError(std::string(option) + " erwartet eine endliche Zahl, nicht \"" +
                         std::string(text) + "\"");
    }
    return value;
}

std::string format_number(const char* fmt, double v) {
    if (std::isnan(v)) {
        return "-";
    }
    if (std::isinf(v)) {
        return v > 0 ? "unendlich" : "-unendlich";
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, v);
    std::string s(buf);
    if (!s.empty() && s.front() == '-') {
        const std::string mantissa = s.substr(1, s.find_first_of("eE") - 1);
        if (mantissa.find_first_not_of("0.") == std::string::npos) {
            s.erase(0, 1);
        }
    }
    return s;
}

std::string fixed(double v, int decimals) {
    const std::string fmt = "%." + std::to_string(decimals) + "f";
    return format_number(fmt.c_str(), v);
}

std::string sci(double v, int decimals) {
    const std::string fmt = "%." + std::to_string(decimals) + "e";
    return format_number(fmt.c_str(), v);
}

std::string with_unit(const std::string& value, double v, const char* unit) {
    if (!std::isfinite(v) || unit[0] == '\0') {
        return value;
    }
    return value + " " + unit;
}

std::size_t display_width(std::string_view s) {
    std::size_t n = 0;
    for (const char c : s) {
        if ((static_cast<unsigned char>(c) & 0xC0U) != 0x80U) {
            ++n;
        }
    }
    return n;
}

std::string pad_right(std::string_view s, std::size_t width) {
    std::string out(s);
    const std::size_t w = display_width(s);
    if (w < width) {
        out.append(width - w, ' ');
    }
    return out;
}

std::string pad_left(std::string_view s, std::size_t width) {
    const std::size_t w = display_width(s);
    std::string out;
    if (w < width) {
        out.append(width - w, ' ');
    }
    out += s;
    return out;
}

void kv(std::ostream& out, std::string_view key, const std::string& value) {
    out << "  " << pad_right(key, 44) << value << '\n';
}

std::string jacobian_text(JacobianMethod method) {
    return method == JacobianMethod::Adjoint
               ? std::string("adjungiert (Zusatzzeilen per finiter Differenzen)")
               : std::string("finite Differenzen (zentral)");
}

std::string join_fixed(const std::vector<double>& values, int decimals) {
    std::string s;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += fixed(values[i], decimals);
    }
    return s;
}

// ---- Dateien ------------------------------------------------------------------------------

namespace fs = std::filesystem;

bool same_file(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    if (fs::exists(a, ec) && fs::exists(b, ec)) {
        const bool same = fs::equivalent(a, b, ec);
        if (!ec) {
            return same;
        }
    }
    const fs::path ca = fs::weakly_canonical(a, ec);
    const fs::path cb = fs::weakly_canonical(b, ec);
    return ca == cb;
}

std::string utf8(const fs::path& p) {
    const std::u8string s = p.generic_u8string();
    return std::string(s.begin(), s.end());
}

// Ein Eintrag bleibt, wenn er von `out_dir` aus (resolve_glass_catalogs, mit der Suche in den
// Elternverzeichnissen) auf dieselbe Datei führt wie von der Eingabedatei aus; sonst relativer
// Pfad von `out_dir` zur aufgelösten Datei, und wenn es keinen gibt (anderes Laufwerk), absolut.
std::vector<std::string> catalogs_for(const SystemDescription& system,
                                      const std::vector<fs::path>& resolved,
                                      const fs::path& out_dir) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < system.glass_catalogs.size(); ++i) {
        const std::string& entry = system.glass_catalogs[i];
        const fs::path& target = resolved.at(i);
        SystemDescription probe;
        probe.glass_catalogs = {entry};
        bool keep = false;
        try {
            const std::vector<fs::path> r = resolve_glass_catalogs(probe, out_dir);
            std::error_code ec;
            keep = fs::equivalent(r.at(0), target, ec) && !ec;
        } catch (const FormatError&) {
            keep = false;
        }
        if (keep) {
            out.push_back(entry);
            continue;
        }
        std::error_code ec;
        const fs::path abs_target = fs::absolute(target, ec).lexically_normal();
        const fs::path rel = fs::relative(abs_target, out_dir, ec);
        out.push_back(utf8(!ec && !rel.empty() ? rel : abs_target));
    }
    return out;
}

} // namespace mko::cli
