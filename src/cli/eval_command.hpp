#pragma once

// Unterbefehl `mikroptim eval` (docs/cli.md, M8-T1): Auswertung eines Systems im Modus EVAL auf
// dem dichten kartesischen Pupillenraster. Kennzahlen pro Feld und Wellenlänge (RMS-OPD,
// Strehl nach Maréchal, RMS-Spot, Schwerpunkt, Verzeichnung, Brennlinien) als Text oder als
// JSON `mikroptim.eval/1`, optional eine OPD-Karte als CSV. Rechnet auf der CPU (eval_cpu) oder
// mit --gpu über mko::cuda::TraceEngine::evaluate.
// Interner Header der Bibliothek mikroptim_cli (gelinkt in das Programm und die Tests).

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Formatkennung der JSON-Ausgabe.
inline constexpr std::string_view kEvalFormat = "mikroptim.eval/1";

// Hilfetext des Unterbefehls.
void print_eval_usage(std::ostream& out);

// Führt `mikroptim eval <args...>` aus; `args` sind die Argumente nach "eval". Exit-Codes wie
// `mikroptim trace` (kExitOk, kExitError, kExitUsage, kExitNoGpu in cli/cli_util.hpp).
int run_eval_command(std::span<const std::string_view> args, std::ostream& out, std::ostream& err);

} // namespace mko::cli
