#pragma once

// Unterbefehl `mikroptim tolerance` (docs/cli.md, M8-T4): Monte-Carlo-Tolerierung eines Systems
// mit dem Toleranzsatz aus dem Abschnitt `tolerances` der Systemdatei. Stichproben als Varianten
// im Batch, Kompensator Arbeitsabstand (gebündelter LM) oder keiner, Verteilung von RMS-Wellenfront
// und Strehl (Mittel, Std, Perzentile, Ausbeute) als Text oder JSON `mikroptim.tolerance/1`,
// optional Histogramm und Werte pro Stichprobe als CSV. Rechnet auf der CPU oder mit --gpu über
// mko::cuda::TraceEngine und optimize_lm_gpu (mikroptim/cuda/tolerance.hpp).
// Interner Header der Bibliothek mikroptim_cli (gelinkt in das Programm und die Tests).

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Formatkennung der JSON-Ausgabe.
inline constexpr std::string_view kToleranceFormat = "mikroptim.tolerance/1";

// Hilfetext des Unterbefehls.
void print_tolerance_usage(std::ostream& out);

// Führt `mikroptim tolerance <args...>` aus; `args` sind die Argumente nach "tolerance".
// Exit-Codes wie `mikroptim trace` (kExitOk, kExitError, kExitUsage, kExitNoGpu in
// cli/cli_util.hpp); eine Datei ohne Abschnitt `tolerances` ist ein Fehler (kExitError).
int run_tolerance_command(std::span<const std::string_view> args, std::ostream& out,
                          std::ostream& err);

} // namespace mko::cli
