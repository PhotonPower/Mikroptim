#pragma once

// Fingerabdrücke koaxialer Ergebnisse (M8-T3, Abnahme "Transformation 0 ⇒ bitgleich zu main"):
// FNV-1a-Prüfsummen über die Bytes aller Ausgaben der CPU-Referenzen bzw. Kernel-Modi für alle
// Testsysteme unter data/systems/ (zufällig gestörte Varianten mit versagten Strahlen). Die
// Tests test_coaxial_fingerprint.cpp und test_cuda_coaxial_fingerprint.cpp verwenden nur APIs,
// die es vor M8-T3 schon gab; sie lassen sich deshalb unverändert auf main übersetzen. Gleiche
// Ausgabezeilen auf main und auf dem Zweig belegen, dass koaxiale Systeme bitgleich bleiben
// (Ablauf: docs/numerik.md, Abschnitt 2, "Dezentrierung und Verkippung").

#include "mikroptim/cpu/load_system.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mko::test {

// FNV-1a über Bytes (64 Bit).
struct Fingerprint {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    std::size_t bytes = 0;

    void add_bytes(const void* data, std::size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < n; ++i) {
            hash ^= p[i];
            hash *= 0x100000001b3ull;
        }
        bytes += n;
    }
    template <class T>
    void add(const std::vector<T>& v) {
        const std::uint64_t n = v.size();
        add_bytes(&n, sizeof(n));
        if (!v.empty()) {
            add_bytes(v.data(), v.size() * sizeof(T));
        }
    }
    template <class T>
    void add_value(const T& x) {
        add_bytes(&x, sizeof(T));
    }
};

// Ausgabezeile "[ Fingerabdruck ] <Bereich> <System> <Modus> <Prüfsumme> <Bytes>".
inline void print_fingerprint(const char* scope, const std::string& system, const std::string& mode,
                              const Fingerprint& f) {
    std::printf("[ Fingerabdruck ] %s %s %s %016llx %zu\n", scope, system.c_str(), mode.c_str(),
                static_cast<unsigned long long>(f.hash), f.bytes);
}

// Die koaxialen Systemdateien unter data/systems/ zum Stand von M8-T3 (alphabetisch; feste
// Liste, damit neue Dateien die Seeds der Störungen nicht verschieben).
inline std::vector<std::string> fingerprint_system_files() {
    return {"achromat_glasvariablen.json",
            "einzellinse_achromat_start.json",
            "einzellinse_bk7.json",
            "einzellinse_bk7_biegung.json",
            "einzellinse_bk7_nebenbedingungen.json",
            "einzellinse_bk7_variablen.json",
            "einzellinse_geisterplatte.json",
            "einzellinse_geisterplatten_exakt.json",
            "einzellinse_glasvariable.json",
            "einzellinse_homotopie_start.json",
            "einzellinse_na05_geisterplatten.json",
            "planplatte_bk7.json",
            "testsystem_mehrlinser.json",
            "weiches_versagen_start.json",
            "zweilinser_mechanik.json",
            "zweilinser_petzval.json"};
}

} // namespace mko::test
