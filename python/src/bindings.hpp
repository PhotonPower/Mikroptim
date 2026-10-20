#pragma once

// Gemeinsame Bausteine der Python-Bindings (M9-T1, Modul mikroptim._mikroptim): Systemobjekt,
// NumPy-Umwandlung (Kopien und Sichten), GPU-Prüfung. Aufbau: docs/architektur.md, Abschnitt
// "Python-Bindings"; Python-API: python/README.md.

#include "mikroptim/cpu/load_system.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::py {

namespace nb = nanobind;

// Geladenes System (Python-Klasse `System`). Unveränderlich aus Python; Änderungen entstehen
// als neues Objekt (from_json, Ergebnis der Optimierung).
struct PySystem {
    LoadedSystem loaded;
    // Verzeichnis, ab dem relative Einträge von glass_catalogs gesucht werden (das der
    // Systemdatei bzw. base_dir von from_json); absolut.
    std::filesystem::path base_dir;
    // Systemdatei wie übergeben (leer, wenn das System nicht aus einer Datei stammt).
    std::filesystem::path file;
};

// Lädt ein System aus einer Datei (load_system) bzw. bereitet eine Beschreibung vor
// (prepare_system mit base_dir).
PySystem load_py_system(const std::filesystem::path& file);
PySystem prepare_py_system(SystemDescription system, const std::filesystem::path& base_dir);

// GPU nicht verfügbar: ohne CUDA gebaut oder kein CUDA-Gerät (Python: GpuUnavailableError,
// abgeleitet von RuntimeError).
class GpuUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Prüft vor einer GPU-Rechnung, ob das Backend verfügbar ist und `device` existiert. Wirft
// GpuUnavailable (ohne CUDA, kein Gerät) bzw. std::invalid_argument (Index ungültig).
void require_gpu(int device);

// Rechentyp aus "fp64" | "df64"; wirft std::invalid_argument.
bool parse_df64(const std::string& precision);

// NumPy-Array, das die Daten eines Vektors übernimmt (Kopie aus Sicht von C++: der Vektor wird
// verschoben und gehört danach dem Array). `shape` muss zur Größe passen. Das Array entsteht
// sofort als Python-Objekt (braucht die GIL), damit Rückgaberegeln wie reference_internal der
// Eigenschaften nicht greifen.
template <class T>
auto to_numpy(std::vector<T> values, std::vector<std::size_t> shape) {
    std::size_t count = 1;
    for (const std::size_t s : shape) {
        count *= s;
    }
    if (count != values.size()) {
        throw std::logic_error("to_numpy: Form passt nicht zur Größe");
    }
    auto* owner = new std::vector<T>(std::move(values));
    nb::capsule deleter(owner, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(owner->data(), shape.size(), shape.data(), deleter).cast();
}

// Eindimensionales NumPy-Array (übernimmt die Daten).
template <class T>
auto to_numpy(std::vector<T> values) {
    const std::size_t n = values.size();
    return to_numpy(std::move(values), {n});
}

// Schreibgeschützte Sicht auf Speicher, der `owner` (ein Python-Objekt) gehört; das Array hält
// `owner` am Leben.
template <class T>
nb::ndarray<nb::numpy, const T> readonly_view(const T* data, std::vector<std::size_t> shape,
                                              nb::handle owner) {
    return nb::ndarray<nb::numpy, const T>(data, shape.size(), shape.data(), owner);
}

// Beschreibbare Sicht auf Speicher, der `owner` gehört (SystemBatch).
template <class T>
nb::ndarray<nb::numpy, T> writable_view(T* data, std::vector<std::size_t> shape, nb::handle owner) {
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
}

// Größe als size_t (Indizes und Zähler sind int in der Bibliothek).
inline std::size_t sz(long long v) {
    return static_cast<std::size_t>(v);
}

// Registrierung der Teile (je eine Quelldatei).
void bind_system(nb::module_& m);
void bind_trace(nb::module_& m);
void bind_eval(nb::module_& m);
void bind_optimize(nb::module_& m);
void bind_cli(nb::module_& m);

} // namespace mko::py
