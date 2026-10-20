#pragma once

// Flächentabelle einer Systemvariante als schlanke Sicht (Zeiger und Längen), gemeinsam für
// CPU und GPU. Layout und Anforderungen: docs/architektur.md, Abschnitt "Sequenzieller Trace".

#include "mikroptim/config.hpp"

#include <cstdint>
#include <type_traits>

namespace mko {

// Größte Flächenzahl eines Systems (inklusive Blende und Geisterplatten). Begrenzt die festen
// Felder in TraceRecord und die Flächentabelle im Shared Memory; der Host prüft sie beim Aufbau.
inline constexpr int kMaxSurfaces = 64;

// Größte Zahl von Koeffizienten (Ordnung K) einer Geisterplatte. Begrenzt die Geisterdaten im
// Shared Memory (höchstens kMaxSurfaces · kMaxGhostTerms Koeffizienten, 16 KiB in FP64); Datei
// und Batch prüfen sie beim Aufbau.
inline constexpr int kMaxGhostTerms = 32;

// Werte pro Fläche in den Transformationsdaten einer Tabelle (M8-T3, ADR-0011): decenter_tilt
// enthält (Δx, Δy, α, β), tilt_trig (cos α, sin α, cos β, sin β).
inline constexpr int kSurfaceTransformValues = 4;

// Flächentyp im Tracer (1 Byte, wie `type[s]` in docs/architektur.md). Die Werte sind stabil,
// weil sie in Device-Speicher und Batches abgelegt werden.
enum class SurfaceKind : std::uint8_t {
    Stop = 0,       // Aperturblende (Ebene), nur Fläche 0
    Plane = 1,      // Ebene, Krümmung wird ignoriert
    Sphere = 2,     // Sphäre der Krümmung c = 1/R
    GhostPlate = 3, // Geisterplatte: Ebene mit Phasenfunktion (docs/numerik.md, Abschnitt 6)
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
// Geisterplatten (M4-T5): Geisterplatte g ist die g-te Fläche vom Typ GhostPlate (in
// Tracerichtung gezählt, wie `ghost_surface[g]` im Batch); ihre Krümmung ist 0 und das Medium
// nach ihr ist das des Luftspalts.
// - ghost_plates = G, ghost_terms = K (K >= 1, wenn G > 0; kürzere Profile mit 0 aufgefüllt),
// - ghost_coeff[g * K + (k − 1)] = a_k der Geisterplatte g,
// - ghost_index[g * L + l] = Brechzahl n_G des virtuellen Glases bei Wellenlänge l,
// - ghost_norm_radius[g] = Normierungsradius r_g [mm].
// Tabellen ohne Geisterplatten haben G = K = 0 und Nullzeiger (Standardwerte); eine Fläche vom
// Typ GhostPlate ohne Daten (g >= G) meldet der Tracer als Unsupported.
//
// Dezentrierung und Verkippung (M8-T3, ADR-0011; docs/numerik.md, Abschnitt 2):
// - decenter_tilt[s * 4 + k] = Δx, Δy [mm], α, β [rad] der Fläche s (k = 0 … 3), Drehpunkt ist
//   der Scheitel; nullptr = koaxiale Tabelle (alle Werte 0, Standard),
// - tilt_trig[s * 4 + k] = cos α, sin α, cos β, sin β (optional, nur zusammen mit
//   decenter_tilt): vorab berechnet (Host-Tabelle, Shared Memory der Kernel); nullptr = der
//   Tracer rechnet sie mit msin/mcos aus den Winkeln (surface_transform_of).
// Flächen, deren vier Werte 0 sind, rechnet der Tracer wie in einer koaxialen Tabelle.
//
// trace_ray ist über den Tabellentyp templatisiert und greift nur über diese Namen zu
// (`sys.surfaces`, `sys.wavelengths`, `sys.type[s]`, `sys.curvature[s]`, `sys.thickness[s]`,
// `sys.index[s * sys.wavelengths + l]`, `sys.ghost_plates`, `sys.ghost_terms`,
// `sys.ghost_coeff[…]`, `sys.ghost_index[…]`, `sys.ghost_norm_radius[g]`, `sys.decenter_tilt`,
// `sys.tilt_trig`). Andere Tabellen
// (z. B. feste Felder im Shared Memory oder eine Batch-Sicht mit Versatz pro Variante) müssen
// nur diese Namen bereitstellen.
template <class R>
struct SurfaceTableView {
    int surfaces;
    int wavelengths;
    const SurfaceKind* type;
    const R* curvature;
    const R* thickness;
    const R* index;
    // Geisterplatten (Standardwerte: keine)
    int ghost_plates = 0;
    int ghost_terms = 0;
    const R* ghost_coeff = nullptr;
    const R* ghost_index = nullptr;
    const R* ghost_norm_radius = nullptr;
    // Dezentrierung und Verkippung (Standardwerte: koaxial)
    const R* decenter_tilt = nullptr;
    const R* tilt_trig = nullptr;
};

static_assert(std::is_trivially_copyable_v<SurfaceTableView<double>>);
static_assert(std::is_standard_layout_v<SurfaceTableView<double>>);

// Dieselbe Sicht, aber zur Übersetzungszeit koaxial (M8-T3): surface_transform_of liefert für
// diesen Typ ohne Laufzeitprüfung `false`, die Transformationsdaten werden nicht gelesen und der
// übersetzte Trace ist der von vor M8-T3. Die Kernel verwenden ihn für koaxiale Batches
// (eigene Kernel-Instanz ohne Transformationen, keine Kosten durch Register oder Stack).
template <class R>
struct CoaxialTableView : SurfaceTableView<R> {
    static constexpr bool kCoaxial = true;
};

static_assert(std::is_trivially_copyable_v<CoaxialTableView<double>>);
static_assert(sizeof(SurfaceKind) == 1);

} // namespace mko
