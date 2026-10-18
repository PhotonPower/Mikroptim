#pragma once

// Unterbefehl `mikroptim optimize` (M3-T5, docs/cli.md): optimiert die Variablen einer
// Systemdatei mit dem gebündelten Levenberg-Marquardt (CPU oder mit --gpu auf einem
// CUDA-Gerät), optional als Multistart aus gestörten Startpunkten, schreibt das Ergebnis als
// neue Systemdatei und gibt ein Iterationsprotokoll und eine Zusammenfassung als Text oder als
// JSON `mikroptim.optimize/1` aus. Interner Header der Bibliothek mikroptim_cli.

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Formatkennung der JSON-Ausgabe.
inline constexpr std::string_view kOptimizeFormat = "mikroptim.optimize/1";

// Hilfetext des Unterbefehls.
void print_optimize_usage(std::ostream& out);

// Führt `mikroptim optimize <args...>` aus; `args` sind die Argumente nach "optimize".
// Protokoll und Zusammenfassung auf `out`, Warnungen und Fehlermeldungen auf `err`. Exit-Codes
// wie `mikroptim trace` (cli_util.hpp): kExitOk, kExitError (Laden, keine Variablen, Rechnung,
// Schreiben der Ergebnisdatei), kExitUsage (Kommandozeile, Ergebnisdatei = Eingabedatei),
// kExitNoGpu. Wirft keine Ausnahmen aus Laden, Optimierung oder Schreiben.
int run_optimize_command(std::span<const std::string_view> args, std::ostream& out,
                         std::ostream& err);

} // namespace mko::cli
