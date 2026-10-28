#pragma once

// Unterbefehl `mikroptim aberrations` (docs/cli.md, M10-T1): Aberrationsbericht pro Fläche
// (paraxiale Strahlen, Seidel-Beiträge, chromatische Beiträge, Sphärochromasie) eines Systems
// mit kollimierter Quelle. Interner Header der Bibliothek mikroptim_cli.

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Hilfetext des Unterbefehls.
void print_aberrations_usage(std::ostream& out);

// Führt `mikroptim aberrations <args...>` aus; `args` sind die Argumente nach "aberrations".
int run_aberrations_command(std::span<const std::string_view> args, std::ostream& out,
                            std::ostream& err);

} // namespace mko::cli
