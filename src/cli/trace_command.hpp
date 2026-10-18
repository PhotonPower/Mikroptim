#pragma once

// Unterbefehl `mikroptim trace` (docs/cli.md): Paraxialdaten, RMS-Wellenfront pro Feld und
// Wellenlänge und versagte Strahlen eines Systems, als Text oder als JSON `mikroptim.trace/1`.
// Die Wellenfront entsteht auf der CPU oder mit --gpu über mko::cuda::TraceEngine (nur in
// CUDA-Builds, dann linkt mikroptim_cli die Bibliothek mikroptim_cuda).
// Interner Header der Bibliothek mikroptim_cli (gelinkt in das Programm und die Tests).

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Formatkennung der JSON-Ausgabe.
inline constexpr std::string_view kTraceFormat = "mikroptim.trace/1";

// Exit-Codes: kExitOk, kExitError, kExitUsage, kExitNoGpu (cli/cli_util.hpp).

// Hilfetext des Unterbefehls.
void print_trace_usage(std::ostream& out);

// Führt `mikroptim trace <args...>` aus; `args` sind die Argumente nach "trace".
// Ergebnis auf `out`, Warnungen und Fehlermeldungen auf `err`. Wirft keine Ausnahmen aus dem
// Laden oder Auswerten: Fehler werden als Meldung auf `err` mit Exit-Code kExitError gemeldet.
int run_trace_command(std::span<const std::string_view> args, std::ostream& out, std::ostream& err);

} // namespace mko::cli
