#pragma once

// Zentrale Toleranzen für Tests (siehe docs/numerik.md, Abschnitt "Genauigkeit").
// Änderungen nur mit Begründung im PR.

namespace mko::test {

// Längen [mm] in FP64 gegen analytische Lösungen
inline constexpr double kLengthFp64 = 1e-11;
// Dimensionslose Größen (Richtungskosinus, Normierung) in FP64
inline constexpr double kUnitFp64 = 1e-13;
// CPU gegen GPU, beide FP64 (Unterschiede nur durch FMA-Kontraktion)
inline constexpr double kCpuGpuFp64 = 1e-12;
// Brechzahl aus Katalogkoeffizienten gegen gedruckte Kontrollwerte (n_d mit 5 Nachkommastellen,
// Rundung der Datenblätter und Koeffizienten). Kein Maß für die Rechengenauigkeit.
inline constexpr double kIndexCatalog = 1e-5;
// df64 gegen FP64 (ab M7)
inline constexpr double kDf64VsFp64 = 1e-9;

} // namespace mko::test
