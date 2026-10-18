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
// CPU gegen GPU für OPD und Residuen zufällig gestörter Varianten (M2-T2), relativ zum Betrag
// des CPU-Werts. Schranke pro Strahl: max(kCpuGpuFp64, kCpuGpuFp64Relative · |x|), also rein
// absolut kCpuGpuFp64 bis |x| = 0,1 mm (ca. 180 Wellen). Schlecht konditionierte Strahlen (nahe
// Totalreflexion oder streifend, z. B. k = 2e-3 an der letzten Fläche und danach d_z < 0)
// verstärken die Rundungsunterschiede aus FMA-Kontraktion und sin/cos der CUDA-Bibliothek; sie
// haben |OPD| von einigen mm. Beobachtet: bis 2,4e-12 mm bei |OPD| = 1,8 … 6 mm, relativ
// <= 1,3e-12 (auch mit --fmad=false noch 1,3e-12 mm, also nicht nur FMA).
inline constexpr double kCpuGpuFp64Relative = 1e-11;
// Merit des Kernels MERIT gegen die Summe der Quadrate der RESIDUALS-Ausgabe, relativ
// (Abnahmekriterium M2-T3). Beide Kernel bilden bitgleiche Residuen; der Unterschied ist nur
// die Summationsreihenfolge der Quadratsumme (Baumreduktion gegen kompensierte Summe auf dem
// Host), Fehlerschranke ~ (⌈n/128⌉ + 7 + F·L) · ε ≈ 1e-14 bei n <= 80 Stützstellen pro Tupel.
inline constexpr double kMeritVsResidualSum = 1e-13;
// Dasselbe mit Rechentyp df64 (M7-T1): Die Baumsumme im Block läuft in df64 (Summanden >= 0,
// Tiefe ⌈n/128⌉ + 5 + 4 <= 10 Additionen mit je <= 3u² + 13u³ ≈ 1,07e-14), danach FP64 wie
// oben; dazu die Rundung der RESIDUALS-Ausgabe nach double (2 · 2⁻⁵³ pro r²). Schranke
// ≈ 1,1e-13 bei n <= 128 Stützstellen pro Tupel.
inline constexpr double kMeritVsResidualSumDf64 = 2e-13;
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
// Referenzsphäre gegen exakte Kugelwellen [mm] (Abnahmekriterium M1-T7): OPL bis zur
// verschobenen Referenzsphäre und OPD nach Kolbenabzug gegen den Kosinussatz. OPL bis ca.
// 100 mm, also einige ulp (ca. 1e-14) pro Rechenschritt.
inline constexpr double kReferenceSphereFp64 = 1e-12;
// RMS-OPD im aplanatischen Fall (Punktquelle im aplanatischen Punkt, Referenz im exakten
// virtuellen Bildpunkt) [mm], Abnahmekriterium M1-T7, bis sin θ = 0,95.
inline constexpr double kAplanaticRms = 1e-10;
// Sphärische Aberration der Einzellinse bei NA <= 0,05 gegen die Seidel-Näherung W040,
// relativ (Abnahmekriterium M1-T7). Modellfehler (Terme höherer Ordnung), kein Maß für die
// Rechengenauigkeit.
inline constexpr double kSeidelRelative = 1e-2;
// Gradient des Dickenprofils einer Geisterplatte gegen zentrale finite Differenzen mit
// Richardson-Extrapolation (5 Stufen, Abbruchfehler O(h¹⁰)), relativ zu |∇t| (Abnahmekriterium
// M4-T5), K bis 16. Rest vor allem aus Rundung (ε |t| / h), am größten nahe Extrema des Profils
// (kleines |∇t|); beobachtet ca. 1e-11.
inline constexpr double kGhostGradientFd = 1e-10;
// RMS-OPD einer Geisterplatte mit gefitteter idealer Fokussierphase bei NA 0,3 [mm]
// (Abnahmekriterium M4-T5). Fit- und Rundungsfehler liegen um Größenordnungen darunter.
inline constexpr double kGhostFocusRms = 1e-6;
// Finite-Differenzen-Jacobi (M3-T2) gegen Richardson-Extrapolation, relativ pro Spalte
// ‖J_p − D_p‖₂/‖D_p‖₂ und in der Frobeniusnorm (Abnahmekriterium M3-T2). Gilt für die
// Schablonen zweiter Ordnung (zentral, einseitig an Schranken). Erwartet: Abbruch h²/6·r‴ mit
// h/x ≈ 6·10⁻⁶ und Rundung ≈ ε·|OPL|/h, zusammen ≈ 10⁻¹⁰ … 10⁻⁸.
inline constexpr double kFdJacobianRichardson = 1e-6;
// Dasselbe für die Schablonen erster Ordnung (vorwärts, gekürzt bei sehr engen Schranken):
// Fehler ≈ (d_a + d_b)/2·r″/r′ ≈ h·|r″/r′|, also ≈ 6·10⁻⁶·|x·r″/r′|. Modellfehler des
// Verfahrens, kein Maß für die Rechengenauigkeit.
inline constexpr double kFdJacobianFirstOrder = 1e-4;
// Konvergenz der Richardson-Referenz selbst: R₁(H/2) (Fehler O(H⁴)) gegen R₂ (O(H⁶)) mit
// H = 10⁻³·max(|x|, x_typ), relativ pro Spalte; muss deutlich unter kFdJacobianRichardson
// liegen, damit die Referenz die Abnahme nicht verfälscht.
inline constexpr double kFdRichardsonReference = 1e-8;
// Zentrale Differenz analytischer Zusatzresiduen (Polynome bis Grad 2 und 1/w) gegen die
// exakte Ableitung, relativ: Abbruch (h/w)² ≈ 4·10⁻¹¹, Rundung ε·|e|/h ≲ 10⁻⁹.
inline constexpr double kFdExtraAnalytic = 1e-8;
// df64 gegen FP64 (Abnahmekriterium M7-T1) [mm]: OPL, OPD und Residuen pro Strahl, Strahlhöhen
// der Max-Reduktionen. Ziel aus docs/numerik.md, Abschnitt 12 (df64: typisch 1e-9).
inline constexpr double kDf64VsFp64 = 1e-9;
// df64 gegen FP64 relativ zum Betrag des FP64-Werts (M7-T1), wie kCpuGpuFp64Relative: Schranke pro
// Strahl max(kDf64VsFp64, kDf64VsFp64Relative · |x|), also rein absolut bis |x| = 10 mm (ca.
// 18.000 Wellen). Schlecht konditionierte Strahlen extremer Varianten (Geisterplatte weit
// außerhalb des Normierungsradius, Profilwerte bis 1e7, |OPD| von 20 mm) verstärken den
// relativen Rundungsfehler von df64 (≈ u² = 3,6e-15) wie in FP64; beobachtet 9,0e-10 mm bei
// |OPD| = 21 mm (relativ 4,3e-11), alle übrigen Strahlen <= 1,6e-11 mm.
inline constexpr double kDf64VsFp64Relative = 1e-10;
// df64-Grundoperationen gegen eine FP64-Referenz, relativ (M7-T1). Schranken aus der Literatur
// (include/mikroptim/math/df64.hpp, docs/numerik.md, Abschnitt 12; u = 2⁻²⁴, u² = 2⁻⁴⁸):
// Addition/Subtraktion 3u² + 13u³, Multiplikation 4u², Division 15u² + 56u³, Wurzel 25/8 u²;
// dazu je u²/32 (= 2⁻⁵³) für die Rundung der FP64-Referenz und für die Umwandlung nach double.
inline constexpr double kDf64AddRelative = (3.0 + 13.0 * 0x1p-24 + 0.0625) * 0x1p-48;
inline constexpr double kDf64MulRelative = (4.0 + 0.0625) * 0x1p-48;
inline constexpr double kDf64DivRelative = (15.0 + 56.0 * 0x1p-24 + 0.0625) * 0x1p-48;
inline constexpr double kDf64SqrtRelative = (3.125 + 0.0625) * 0x1p-48;
// sin und cos in df64 gegen FP64 für |x| <= 100 (absolut). Reduktion mit dreiteiligem π/2 und
// Taylor-Polynome mit Abbruchfehler < 7e-17; die Rundungsfehler der etwa 20 df64-Operationen
// liegen bei wenigen u² (u² ≈ 3,6e-15), dazu ≤ 1 ulp der FP64-Referenz.
inline constexpr double kDf64Trig = 1e-14;

} // namespace mko::test
