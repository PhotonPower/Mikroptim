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
// Buchdahl-Fit mit 3 Dispersionstermen an Sellmeier über 400-700 nm: maximaler Fehler der
// Brechzahl (Abnahmekriterium M1-T3). Modellfehler, kein Maß für die Rechengenauigkeit.
inline constexpr double kBuchdahlFit3 = 1e-5;
// Buchdahl-Fit an ein exaktes Buchdahl-Polynom: Rückgewinnung der Koeffizienten und
// Restfehler. Konditionszahl der Vandermonde-Matrix in omega etwa 1,4e3 (3 Terme, 400-700 nm),
// Householder-QR in FP64; beobachtet: nu3 bis 2,5e-13 (kleinste Spaltennorm w^3), n0, nu1
// und Restfehler um 1e-15.
inline constexpr double kBuchdahlFitExact = 1e-12;
// Pupillenquadratur: Fehler exakt integrierbarer Mittelwerte (Zernike-Mittelwerte und
// Orthogonalität, Summe der Gewichte). Abnahmekriterium M1-T4; Rundungsfehler der
// Summation über bis zu 256 Stützstellen und der Polynomauswertung liegen bei 1e-15.
inline constexpr double kQuadrature = 1e-14;
// Gauß-Legendre-Knoten und -Gewichte gegen unabhängige FP64-Rechnungen (geschlossene
// Formeln, Newton-Iteration): wenige ulp; Knoten absolut, Gewichte relativ.
inline constexpr double kQuadratureNode = 1e-15;
// Paraxialdaten gegen geschlossene Formeln (Linsenmacherformel, dünne Linse), relativ.
// Abnahmekriterium M1-T6; der y-nu-Trace über wenige Flächen hat Rundungsfehler ~1e-15.
inline constexpr double kParaxialRelative = 1e-12;
// Gerundete Zahlenwerte aus Plan und Doku (zwei Nachkommastellen, z. B. f ≈ 101,05 mm) [mm].
// Kein Maß für die Rechengenauigkeit; die exakte Prüfung läuft gegen die Formel.
inline constexpr double kDocumentedValue = 1e-2;
// Grenzwert realer Strahlen h -> 0 gegen die paraxiale Schnittweite [mm] (M1-T6): bei
// h = 1e-4 mm ist die sphärische Längsaberration der Testlinse ~1.6e-10 mm.
inline constexpr double kParaxialLimit = 1e-8;
// Relative Abweichung vom Skalierungsgesetz Längsaberration ∝ h² (Verhältnis bei h und h/2
// gleich 4) für h ≤ 0,5 mm; der Term höherer Ordnung (∝ h⁴) liefert dort ~1e-4.
inline constexpr double kAberrationScaling = 1e-3;
// Abstand eines gebrochenen Strahls (Gerade) vom analytischen (virtuellen) Bildpunkt [mm] in
// FP64, aplanatische Punkte einer Kugelfläche (Abnahmekriterium M1-T5). Gemessen bis ca.
// 5e-13 am Aperturrand (sin θ = 0,95), wo der Austritt nahezu streifend ist und
// Rundungsfehler des Schnittpunkts (ca. 1e-14) in der Richtung verstärkt werden.
inline constexpr double kRayPointFp64 = 1e-12;
// Strahlumkehr: Vorwärts- und Rückwärtstrace durch ein mehrlinsiges System kehren zum Start
// zurück [mm], auch für die OPL (Abnahmekriterium M1-T5).
inline constexpr double kRayReversalFp64 = 1e-10;
// df64 gegen FP64 (ab M7)
inline constexpr double kDf64VsFp64 = 1e-9;

} // namespace mko::test
