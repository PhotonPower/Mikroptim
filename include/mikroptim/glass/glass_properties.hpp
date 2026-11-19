#pragma once

// Katalogkennzahlen eines Glases und Bereichsfilter für die Glasabfrage (M10-T5,
// docs/entwurfssteuerung.md, Abschnitt 5.3; Formeln in docs/numerik.md, Abschnitt 17).
// Host-Code, Bibliothek Mikroptim::glass.
//
//     ν_d = (n_d − 1) / (n_F − n_C),   P_g,F = (n_g − n_F) / (n_F − n_C),
//     ΔP_g,F = P_g,F − (0,6438 − 0,001682 ν_d)
//
// Normalgerade der Teildispersion nach Schott, Technical Information TIE-29 (durch die
// Normalgläser K7 und F2). Linien: d 587,5618 nm, F 486,1327 nm, C 656,2725 nm, g 435,8343 nm.

#include "mikroptim/glass/catalog.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace mko {

inline constexpr double kLineD = 587.5618;
inline constexpr double kLineF = 486.1327;
inline constexpr double kLineC = 656.2725;
inline constexpr double kLineG = 435.8343;
// Normalgerade P_g,F = a + b ν_d (Schott TIE-29)
inline constexpr double kPgfNormalA = 0.6438;
inline constexpr double kPgfNormalB = -0.001682;

struct GlassProperties {
    std::string name;
    double n_d = 0.0;
    double n_F = 0.0;
    double n_C = 0.0;
    double n_g = 0.0;
    double v_d = 0.0;   // Abbezahl ν_d
    double p_gF = 0.0;  // relative Teildispersion P_g,F
    double dp_gF = 0.0; // Abweichung von der Normalgeraden ΔP_g,F
    std::array<double, 2> range_nm{0.0, 0.0};
};

// Kennzahlen eines Glases. Wirft GlassError, wenn range_nm die Linien g … C nicht abdeckt.
GlassProperties glass_properties(const Glass& glass);

// Geschlossener Bereich [min, max]; fehlende Grenze = offen.
struct ValueRange {
    std::optional<double> min;
    std::optional<double> max;
    bool contains(double x) const { return (!min || x >= *min) && (!max || x <= *max); }
};

// Bereichsfilter auf die Kennzahlen; alle gesetzten Bereiche müssen gelten.
struct GlassPropertyFilter {
    std::optional<ValueRange> n_d;
    std::optional<ValueRange> v_d;
    std::optional<ValueRange> p_gF;
    std::optional<ValueRange> dp_gF;
    bool matches(const GlassProperties& p) const;
};

// Kennzahlen aller Gläser des Katalogs, die den Filter erfüllen (Lade-Reihenfolge). Gläser, deren
// Bereich g … C nicht abdeckt, werden übersprungen (Namen in `skipped`).
std::vector<GlassProperties> list_glass_properties(const GlassCatalog& catalog,
                                                   const GlassPropertyFilter& filter = {},
                                                   std::vector<std::string>* skipped = nullptr);

} // namespace mko
