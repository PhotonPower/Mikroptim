#pragma once

// Unterbefehl `mikroptim synth` (M6-T3, docs/cli.md): Homotopie-Controller der Systemsynthese
// auf einer Startdatei (Apertur, Feld bis zum Ziel; Schrittweitensteuerung, Einfüge-Operatoren aus
// der Diagnose der Geisterplatten, Fortsetzungen), CPU oder mit --gpu auf einem CUDA-Gerät.
// Protokoll jedes Schritts als Text oder JSON `mikroptim.synth/1`, optional zusätzlich als
// JSON-Datei (--log), Ergebnis als Systemdatei (-o). Interner Header der Bibliothek mikroptim_cli.

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Hilfetext des Unterbefehls.
void print_synth_usage(std::ostream& out);

// Führt `mikroptim synth <args...>` aus; `args` sind die Argumente nach "synth". Exit-Codes wie
// `mikroptim optimize`: kExitOk (Lauf beendet, auch wenn das Ziel verfehlt wurde; der Status
// steht in der Ausgabe), kExitError (Laden, Rechnung, Schreiben), kExitUsage, kExitNoGpu.
int run_synth_command(std::span<const std::string_view> args, std::ostream& out, std::ostream& err);

} // namespace mko::cli
