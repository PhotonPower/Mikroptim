#pragma once

// Unterbefehl `mikroptim convert` (docs/cli.md, M9-T2): Umwandlung zwischen Raytatouille
// (`*.rtt.json`) und `mikroptim.system/1` in beide Richtungen; die Richtung ergibt sich aus
// dem Inhalt der Eingabedatei. Interner Header der Bibliothek mikroptim_cli.

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

void print_convert_usage(std::ostream& out);

int run_convert_command(std::span<const std::string_view> args, std::ostream& out,
                        std::ostream& err);

} // namespace mko::cli
