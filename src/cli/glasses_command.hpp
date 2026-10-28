#pragma once

// Unterbefehl `mikroptim glasses` (docs/cli.md, M11-T5): algebraische Glasvorauswahl für
// Farbkorrektur an dünnen Elementen (Paare für Achromate, Tripel für Apochromate) aus den
// Gläsern eines oder mehrerer Kataloge. Interner Header der Bibliothek mikroptim_cli.

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

void print_glasses_usage(std::ostream& out);

int run_glasses_command(std::span<const std::string_view> args, std::ostream& out,
                        std::ostream& err);

} // namespace mko::cli
