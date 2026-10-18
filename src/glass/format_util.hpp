#pragma once

// Interne Hilfsfunktionen für Fehlermeldungen des Glasmoduls.

#include <ios>
#include <locale>
#include <sstream>
#include <string>

namespace mko::detail {

// Zahl für Meldungen, bis zu 10 signifikante Stellen, unabhängig von der globalen Locale.
inline std::string format_number(double x) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os.precision(10);
    os << x;
    return os.str();
}

} // namespace mko::detail
