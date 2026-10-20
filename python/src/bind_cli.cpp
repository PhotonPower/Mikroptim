// Python-Bindings: Unterbefehle der CLI im Modul (M9-T1). Dieselben Funktionen wie das Programm
// `mikroptim` (src/cli/main.cpp), Ausgabe als Text statt auf stdout/stderr. Grundlage von
// `python -m mikroptim` und der Vergleiche Python-API gegen CLI in den Tests.

#include "bindings.hpp"
#include "cli/eval_command.hpp"
#include "cli/insert_command.hpp"
#include "cli/optimize_command.hpp"
#include "cli/snap_command.hpp"
#include "cli/synth_command.hpp"
#include "cli/trace_command.hpp"
#include "mikroptim/version.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace mko::py {

namespace {

// Führt `mikroptim <args...>` aus; Rückgabe (Exit-Code, stdout, stderr).
std::tuple<int, std::string, std::string> run_cli(const std::vector<std::string>& argv) {
    std::ostringstream out;
    std::ostringstream err;
    int code = cli::kExitUsage;
    {
        nb::gil_scoped_release release;
        if (argv.empty()) {
            err << "Aufruf: mikroptim <trace|optimize|eval|snap|insert|synth> ... "
                   "(Hilfe: <befehl> --help)\n";
        } else if (argv[0] == "--version") {
            out << "mikroptim " << kVersion << '\n';
            code = cli::kExitOk;
        } else {
            const std::vector<std::string_view> args(argv.begin() + 1, argv.end());
            const std::string& command = argv[0];
            if (command == "trace") {
                code = cli::run_trace_command(args, out, err);
            } else if (command == "optimize") {
                code = cli::run_optimize_command(args, out, err);
            } else if (command == "eval") {
                code = cli::run_eval_command(args, out, err);
            } else if (command == "snap") {
                code = cli::run_snap_command(args, out, err);
            } else if (command == "insert") {
                code = cli::run_insert_command(args, out, err);
            } else if (command == "synth") {
                code = cli::run_synth_command(args, out, err);
            } else {
                err << "mikroptim: unbekannter Befehl " << command << '\n';
            }
        }
    }
    return {code, out.str(), err.str()};
}

} // namespace

void bind_cli(nb::module_& m) {
    m.def("_run_cli", &run_cli, nb::arg("args"),
          "Führt einen Unterbefehl der CLI aus (args ohne Programmnamen, z. B. [\"trace\", "
          "\"system.json\", \"--json\"]); Rückgabe (Exit-Code, stdout, stderr).");
}

} // namespace mko::py
