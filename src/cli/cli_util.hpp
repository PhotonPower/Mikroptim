#pragma once

// Gemeinsame Bausteine der Unterbefehle (docs/cli.md): Exit-Codes, Fehler der Kommandozeile,
// Zahlen aus Argumenten und die plattformunabhängige Textformatierung (printf-artig in der
// C-Locale, nicht endliche Werte ausgeschrieben). Interner Header der Bibliothek mikroptim_cli.

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mko::cli {

// Exit-Codes aller Unterbefehle (`mikroptim trace`, `mikroptim optimize`).
inline constexpr int kExitOk = 0;    // Ergebnis ausgegeben
inline constexpr int kExitError = 1; // Datei nicht lesbar, Formatfehler, Glasfehler, ...
inline constexpr int kExitUsage = 2; // ungültige Kommandozeile
inline constexpr int kExitNoGpu = 3; // --gpu, aber ohne CUDA gebaut oder kein CUDA-Gerät

// Fehler in der Kommandozeile (Exit-Code kExitUsage).
class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Ganze Zahl bzw. vorzeichenlose 64-Bit-Zahl bzw. endliche Gleitkommazahl aus dem Wert der
// Option `option` (std::from_chars, unabhängig von der Locale); sonst UsageError.
int parse_int(std::string_view option, std::string_view text);
std::uint64_t parse_uint64(std::string_view option, std::string_view text);
double parse_double(std::string_view option, std::string_view text);

// printf-artige Formatierung (C-Locale, unabhängig von Plattform und Compiler). NaN wird "-",
// ±∞ wird "unendlich" bzw. "-unendlich". Ein Minuszeichen vor einer zu null gerundeten Zahl
// entfällt, damit das Vorzeichen von Rundungsresten nicht in der Ausgabe erscheint.
std::string format_number(const char* fmt, double v);
std::string fixed(double v, int decimals = 6); // %.<decimals>f
std::string sci(double v, int decimals = 6);   // %.<decimals>e

// Wert mit Einheit; ohne Einheit, wenn der Wert nicht endlich ist.
std::string with_unit(const std::string& value, double v, const char* unit);

// Zahl der Zeichen eines UTF-8-Texts (Fortsetzungsbytes 10xxxxxx zählen nicht).
std::size_t display_width(std::string_view s);
// Links- bzw. rechtsbündig auf `width` Zeichen aufgefüllt.
std::string pad_right(std::string_view s, std::size_t width);
std::string pad_left(std::string_view s, std::size_t width);

// Zeile "  <Schlüssel auf 44 Zeichen><Wert>".
void kv(std::ostream& out, std::string_view key, const std::string& value);

// Werte mit fixed(·, decimals), durch ", " getrennt.
std::string join_fixed(const std::vector<double>& values, int decimals);

} // namespace mko::cli
