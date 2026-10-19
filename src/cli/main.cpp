#include "cli/eval_command.hpp"
#include "cli/insert_command.hpp"
#include "cli/optimize_command.hpp"
#include "cli/snap_command.hpp"
#include "cli/synth_command.hpp"
#include "cli/trace_command.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <cstdio>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

void print_usage() {
    std::puts(
        "Aufruf: mikroptim [--version] [--devices]\n"
        "        mikroptim trace <system.json> [--json] [--rings N] [--spokes M]\n"
        "                        [--gpu [--device N] [--precision fp64|df64]]\n"
        "        mikroptim optimize <system.json> [-o <ergebnis.json>] [--json] [--quiet]\n"
        "                        [--max-iterations N] [--runs K ...] [--gpu [--device N]]\n"
        "        mikroptim eval <system.json> [--grid N] [--json] [--gpu]\n"
        "        mikroptim snap <system.json> [-o <ergebnis.json>] [--json] [--neighbors k]\n"
        "                       [--max-combinations N] [--gpu [--device N]]\n"
        "        mikroptim insert <system.json> -o <neu.json> (--meniscus ... | --split ...)\n"
        "        mikroptim synth <start.json> --target-na NA [-o <ergebnis.json>] [--json]\n"
        "                        [--log <protokoll.json>] [--gpu [--device N]]\n"
        "  --version   Version ausgeben\n"
        "  --devices   CUDA-Geräte auflisten\n"
        "  trace       System auswerten (Hilfe: mikroptim trace --help)\n"
        "  optimize    Variablen optimieren (Hilfe: mikroptim optimize --help)\n"
        "  eval        Auswertung auf dem Pupillenraster (Hilfe: mikroptim eval --help)\n"
        "  snap        auf Katalog-Gläser einrasten (Hilfe: mikroptim snap --help)\n"
        "  insert      Einfüge-Operator A oder B anwenden (Hilfe: mikroptim insert --help)\n"
        "  synth       Systemsynthese, Homotopie (Hilfe: mikroptim synth --help)");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
    }
    const std::string_view arg{argv[1]};
    if (arg == "--version") {
        std::printf("mikroptim %s\n", mko::kVersion);
        return 0;
    }
    if (arg == "--devices") {
#ifdef MIKROPTIM_HAS_CUDA
        return mko::cuda::print_devices() > 0 ? 0 : 1;
#else
        std::puts("Ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF).");
        return 1;
#endif
    }
    if (arg == "trace") {
        const std::vector<std::string_view> args(argv + 2, argv + argc);
        return mko::cli::run_trace_command(args, std::cout, std::cerr);
    }
    if (arg == "optimize") {
        const std::vector<std::string_view> args(argv + 2, argv + argc);
        return mko::cli::run_optimize_command(args, std::cout, std::cerr);
    }
    if (arg == "snap") {
        const std::vector<std::string_view> args(argv + 2, argv + argc);
        return mko::cli::run_snap_command(args, std::cout, std::cerr);
    }
    if (arg == "eval") {
        const std::vector<std::string_view> args(argv + 2, argv + argc);
        return mko::cli::run_eval_command(args, std::cout, std::cerr);
    }
    if (arg == "synth") {
        const std::vector<std::string_view> args(argv + 2, argv + argc);
        return mko::cli::run_synth_command(args, std::cout, std::cerr);
    }
    if (arg == "insert") {
        const std::vector<std::string_view> args(argv + 2, argv + argc);
        return mko::cli::run_insert_command(args, std::cout, std::cerr);
    }
    print_usage();
    return 1;
}
