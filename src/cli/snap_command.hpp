#pragma once

// Unterbefehl `mikroptim snap` (M6-T5, docs/cli.md): rastet die Glasvariablen eines
// (kontinuierlich optimierten) Systems auf Katalog-Gläser ein, reoptimiert die übrigen
// Variablen für jede Kombination der k nächsten Gläser (gebündelter Levenberg-Marquardt, CPU
// oder mit --gpu auf einem CUDA-Gerät), schreibt das beste Ergebnis als neue Systemdatei und
// gibt einen Bericht als Text oder als JSON `mikroptim.snap/1` aus. Interner Header der
// Bibliothek mikroptim_cli.

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Formatkennung der JSON-Ausgabe.
inline constexpr std::string_view kSnapFormat = "mikroptim.snap/1";

// Hilfetext des Unterbefehls.
void print_snap_usage(std::ostream& out);

// Führt `mikroptim snap <args...>` aus; `args` sind die Argumente nach "snap". Bericht auf
// `out`, Warnungen und Fehlermeldungen auf `err`. Exit-Codes wie `mikroptim optimize`:
// kExitOk, kExitError (Laden, keine Glasvariablen, Rechnung, Schreiben der Ergebnisdatei),
// kExitUsage (Kommandozeile, Ergebnisdatei = Eingabedatei), kExitNoGpu. Wirft keine Ausnahmen
// aus Laden, Rechnung oder Schreiben.
int run_snap_command(std::span<const std::string_view> args, std::ostream& out, std::ostream& err);

} // namespace mko::cli
