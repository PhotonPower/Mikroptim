#pragma once

// Flächentabelle einer Systemvariante als schlanke Sicht (Zeiger und Längen), gemeinsam für
// CPU und GPU. Layout und Anforderungen: docs/architektur.md, Abschnitt "Sequenzieller Trace".

#include "mikroptim/config.hpp"

#include <cstdint>
#include <type_traits>

namespace mko {

// Größte Flächenzahl eines Systems (inklusive Blende). Begrenzt die festen Felder in
// TraceRecord und die Flächentabelle im Shared Memory; der Host prüft sie beim Aufbau.
inline constexpr int kMaxSurfaces = 64;

// Flächentyp im Tracer (1 Byte, wie `type[s]` in docs/architektur.md). Die Werte sind stabil,
// weil sie in Device-Speicher und Batches abgelegt werden.
enum class SurfaceKind : std::uint8_t {
    Stop = 0,       // Aperturblende (Ebene), nur Fläche 0
    Plane = 1,      // Ebene, Krümmung wird ignoriert
    Sphere = 2,     // Sphäre der Krümmung c = 1/R
    GhostPlate = 3, // reserviert für M4-T5 (Geisterplatte), vom Tracer noch nicht unterstützt
};

// Sicht auf die Flächendaten einer Variante. Trivial kopierbar; zeigt im CPU-Tracer auf
// Host-Vektoren (HostSurfaceTable), im Kernel auf Shared Memory.
//
// - surfaces = S, Fläche 0 ist die Blende (bzw. bei Testsystemen die Startfläche),
// - wavelengths = L,
// - type[s], curvature[s], thickness[s] für s = 0 … S−1 (thickness[S−1] = Abstand zur
//   Bildebene),
// - index[s * L + l] = Brechzahl des Mediums nach Fläche s bei Wellenlänge l (Layout wie
//   IndexTable und `index[v][s][l]`); das Medium vor Fläche 0 ist Luft.
//
// trace_ray ist über den Tabellentyp templatisiert und greift nur über diese Namen zu
// (`sys.surfaces`, `sys.wavelengths`, `sys.type[s]`, `sys.curvature[s]`, `sys.thickness[s]`,
// `sys.index[s * sys.wavelengths + l]`). Andere Tabellen (z. B. feste Felder im Shared Memory
// oder eine Batch-Sicht mit Versatz pro Variante) müssen nur diese Namen bereitstellen.
template <class R>
struct SurfaceTableView {
    int surfaces;
    int wavelengths;
    const SurfaceKind* type;
    const R* curvature;
    const R* thickness;
    const R* index;
};

static_assert(std::is_trivially_copyable_v<SurfaceTableView<double>>);
static_assert(std::is_standard_layout_v<SurfaceTableView<double>>);
static_assert(sizeof(SurfaceKind) == 1);

} // namespace mko
