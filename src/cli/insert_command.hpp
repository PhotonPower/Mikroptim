#pragma once

// Unterbefehl `mikroptim insert` (docs/cli.md, M6-T2): Einfüge-Operator A (Meniskus ohne
// Brechkraft) oder B (Aufspalten einer Linse in ein Kittglied) auf eine Systemdatei anwenden und
// das neue System schreiben; gibt die Merit vorher und nachher aus (CPU, Strahlresiduen plus
// paraxiale Nebenbedingungen und L1-Strafe der Geisterplatten). Interner Header der Bibliothek
// mikroptim_cli.

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Hilfetext des Unterbefehls.
void print_insert_usage(std::ostream& out);

// Führt `mikroptim insert <args...>` aus; `args` sind die Argumente nach "insert". Exit-Codes
// wie die übrigen Unterbefehle (kExitOk, kExitError, kExitUsage).
int run_insert_command(std::span<const std::string_view> args, std::ostream& out,
                       std::ostream& err);

} // namespace mko::cli
