#pragma once

// Glaskatalog im Format `mikroptim.glass/1` (docs/dateiformat.md, Abschnitt "Glasdaten").
// Host-Code: Fehler beim Einlesen und Auswerten werden als GlassError gemeldet.

#include <array>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

// Fehler beim Laden eines Katalogs oder beim Auswerten einer Brechzahl.
class GlassError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Dispersionsmodell eines Katalogglases (docs/dateiformat.md, Glasdaten; ADR-0014).
enum class GlassModel {
    Sellmeier,  // n^2 = 1 + A + sum B_i lambda^2 / (lambda^2 - C_i)
    Polynomial, // n^2 = sum a_j lambda^{p_j}
};

// Höchstzahlen der Terme (Dateiformat; die Kataloge brauchen höchstens 5 bzw. 9).
inline constexpr std::size_t kMaxSellmeierTerms = 8;
inline constexpr std::size_t kMaxPolynomialTerms = 12;

// Ein Katalogglas (lambda in µm, C in µm^2).
struct Glass {
    std::string name;
    GlassModel model = GlassModel::Sellmeier;
    // Sellmeier: Konstante A (Standard 0) und 1 … kMaxSellmeierTerms Terme (B, C gleich lang)
    double A = 0.0;
    std::vector<double> B;
    std::vector<double> C;
    // Polynom: Koeffizienten a_j und Exponenten p_j (gleich lang, 1 … kMaxPolynomialTerms)
    std::vector<double> coefficients;
    std::vector<double> exponents;
    std::array<double, 2> range_nm{0.0, 0.0}; // Gültigkeitsbereich [min, max] in nm
    std::optional<double> reference_n_d;      // Kontrollwert bei 587,56 nm, falls angegeben
    std::string source;                       // Herkunft der Daten, falls angegeben

    // true, wenn range_nm[0] <= lambda_nm <= range_nm[1].
    bool in_range(double lambda_nm) const;

    // Brechzahl relativ zu Luft. Wirft GlassError, wenn lambda_nm außerhalb von range_nm liegt.
    double index(double lambda_nm) const;
};

// Sammlung von Gläsern mit eindeutigen Namen. Namen werden exakt (Groß-/Kleinschreibung
// beachtend) verglichen; die Kataloge verwenden Großschreibung.
class GlassCatalog {
public:
    GlassCatalog() = default;

    // Liest eine Datei im Format `mikroptim.glass/1`. Wirft GlassError bei Lese-, Syntax-
    // und Validierungsfehlern; die Meldung enthält Datei und JSON-Pfad (z. B. `glasses[2].B`).
    static GlassCatalog from_file(const std::filesystem::path& path);

    // Wie from_file, aber aus Text. `origin` erscheint in Fehlermeldungen.
    static GlassCatalog from_string(std::string_view json_text, std::string_view origin = "<Text>");

    // Fügt ein Glas hinzu. Wirft GlassError bei doppeltem oder ungültigem Namen.
    void add(Glass glass);

    // Übernimmt alle Gläser eines anderen Katalogs. Wirft GlassError bei doppelten Namen;
    // in diesem Fall bleibt dieser Katalog unverändert.
    void merge(const GlassCatalog& other);

    // Suche nach Name; nullptr, wenn nicht vorhanden.
    const Glass* find(std::string_view name) const;
    bool contains(std::string_view name) const { return find(name) != nullptr; }
    // Wie find, wirft aber GlassError, wenn das Glas fehlt.
    const Glass& at(std::string_view name) const;

    std::size_t size() const { return glasses_.size(); }
    bool empty() const { return glasses_.empty(); }
    // Gläser in Lade-Reihenfolge.
    const std::vector<Glass>& glasses() const { return glasses_; }

    // Hinweise beim Laden, z. B. ignorierte unbekannte Felder (docs/dateiformat.md).
    const std::vector<std::string>& warnings() const { return warnings_; }

private:
    std::vector<Glass> glasses_;
    std::vector<std::string> warnings_;
};

} // namespace mko
