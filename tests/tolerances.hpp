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
// CPU gegen GPU für die Strafgrößen π des weichen Versagens (M4-T2, dimensionslos), relativ:
// Schranke max(kCpuGpuFp64, kCpuGpuPenaltyRelative · |π|). Tief im Versagensbereich entstehen
// die Strafen aus fortgesetzten Strahlen weit außerhalb des Systems (Höhen bis 10⁴ mm,
// Geisterplatten weit außerhalb von r_g mit Profilwerten bis 1e26), oft nach mehreren weichen
// Ereignissen; dort verstärken sich die Rundungsunterschiede der FMA-Kontraktion. Beobachtet
// (je 1.000 bis 10.000 gestörte Varianten): relativ bis 7e-8 (ein Strahl, π = 1,86, weich und
// danach hart), sonst ≤ 5,6e-10; Strahlen nahe an einer Grenze (π klein) einige 1e-14. Gegen
// df64 statt FP64 auf der GPU beobachtet bis 2,7e-8.
inline constexpr double kCpuGpuPenaltyRelative = 1e-6;
// CPU gegen GPU, OPD und Wellenfrontresiduen in Tupeln mit weich versagten Strahlen (M4-T2)
// [mm], ersetzt dort kCpuGpuFp64 als absoluten Anteil. Weich versagte Strahlen sind fortgesetzt
// (verfehlte Fläche, Totalreflexion) und laufen weit außerhalb des Systems; ihre OPL ist so
// schlecht konditioniert wie die streifender gültiger Strahlen, und über den Kolben wirkt das
// auf alle Strahlen des Tupels. Beobachtet bis 1,35e-10 mm (|OPD| = 2,5 mm).
inline constexpr double kCpuGpuSoftTuple = 1e-9;
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
// Gewichtete Merit (M4-T1) gegen Σ_f Σ_λ ŵ_f ŵ_λ (RMS_fλ/λ_ref)² mit dem RMS aus M1-T7 bzw. aus
// der OPD desselben Laufs, relativ (Abnahmekriterium M4-T1). Dieselbe OPD auf beiden Seiten;
// Unterschiede nur aus der Rundung von s_{f,l} und r_k (wenige ε) und der Summationsreihenfolge
// (Σ W_k OPD_k² gegen Σ r_k², Baumreduktion im Kernel), ≈ 1e-14.
inline constexpr double kWeightedMerit = 1e-12;
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
// FP32-Vorfilterung (M7-T2): Grenzabstand (Einheit eines Kosinus) in float gegen dieselbe
// Funktion in FP64 an der Versagensgrenze der Totalreflexion. Nahe der Grenze ist der
// Grenzabstand √|k|, die Rundung von k in float (einige u ≈ 6e-8) wirkt dort als
// √(δk) ≈ 5e-4; weit weg einige u. Beobachtet 4,6e-5.
inline constexpr double kPrefilterMarginFp32VsFp64 = 1e-3;
// FP32-Vorfilterung: größte optische Länge Λ_v (Maximum über Strahlen in float) GPU gegen
// CPU-Referenz, relativ; Unterschiede nur aus FMA-Kontraktion auf dem Gerät (einige u).
inline constexpr double kPrefilterGpuCpuLengthRelative = 1e-4;

// Modus EVAL (M8-T1): Brennlinien aus den Feldstrahlen (Nachbarstrahlen im Abstand δ = 10⁻³ der
// Pupille) bei einer perfekten Kugelwelle (aplanatische Punkte) [mm]. Die Strahlen schneiden
// sich exakt im Bildpunkt; Rundungsfehler der Lage (~10⁻¹⁴ mm) werden durch den kleinen
// Winkelunterschied der Nachbarstrahlen (~2δ·sin θ) um ~10³ verstärkt.
inline constexpr double kEvalFocusFp64 = 1e-9;
// EVAL: Brennlinien gegen die Coddington-Gleichungen, relativ zu max(1 mm, |z|). Abbruchfehler
// der symmetrischen Nachbarstrahlen O(δ²) (Koma hebt sich im Paar ±δ auf) und Rundung ~ε/δ.
inline constexpr double kEvalFocusRelative = 1e-6;
// EVAL: Bildschale gegen die Petzval-Näherung z = κ_P h²/2 bei θ = 1°, relativ. Modellfehler
// der Näherung O(θ²) ≈ 3·10⁻⁴, kein Maß für die Rechengenauigkeit.
inline constexpr double kEvalPetzvalModel = 1e-3;
// EVAL: RMS der OPD auf dem Raster gegen den RMS des stetigen Modells (Legendre-Projektion der
// Gauß-Quadratur) auf denselben Rasterpunkten, relativ. Nur Rundung und Modellabbruch.
inline constexpr double kEvalGridModel = 1e-8;
// EVAL: Konvergenz des Raster-RMS gegen die Gauß-Quadratur, Faktor c der Schranke c·N^(−4/3).
// Der Randanteil dominiert den inneren Mittelpunktfehler O(N⁻²): Die Zahl der Gitterpunkte im
// Kreis weicht um E ~ R^(2/3) (R = N/2, Gaußsches Kreisproblem; beobachtet |E| <= 1,5 R^(2/3))
// von π N²/4 ab, relativ also ≈ 0,8·1,5·N^(−4/3), und der RMS einer zum Rand wachsenden OPD
// (sphärische Aberration) reagiert darauf etwa doppelt so stark. Gemessen c <= 2,4.
inline constexpr double kEvalGridConvergence = 4.0;
// EVAL: Verhältnis der Abweichung des Spot-Schwerpunkts von der paraxialen Bildhöhe bei
// halbierter Apertur (Koma der dritten Ordnung ∝ NA², Verhältnis 4); Terme höherer Ordnung
// ändern es relativ um O(NA²) und die Verzeichnung (∝ θ²) um ~10⁻³. Modellfehler.
inline constexpr double kEvalComaScaling = 0.05;
// EVAL: Schwerpunkt des Spots bei kleinem Feld (0,05°) gegen die paraxiale Bildhöhe f_λ tan θ,
// relativ (Einzellinse, NA <= 0,025). Modellfehler: Koma der dritten Ordnung (∝ NA²) und
// Verzeichnung (∝ θ²), kein Maß für die Rechengenauigkeit.
inline constexpr double kEvalCentroidParaxial = 1e-3;
// EVAL: Summe der Rastergewichte gegen 1 (bis zu 5·10⁴ Summanden 1/n, Rundung <= n·ε).
inline constexpr double kEvalWeightSum = 1e-11;
// EVAL: halbe gegen volle Pupille [mm] (Kennzahlen, Karten): dieselben Strahlen, nur andere
// Summationsreihenfolge im Kolben und in den Momenten (mit Spiegelung), Werte bis ~10⁻³ mm.
inline constexpr double kEvalHalfPupil = 1e-14;
// EVAL: Änderung des Defokus-Koeffizienten bei verschobener Bildebene gegen n δz (NA²/2 + NA⁴/8),
// relativ. Modellfehler: sin U ist wegen der sphärischen Aberration nicht streng ∝ ρ
// (Längsaberration/Brennweite ≈ 1 % bei NA 0,05; gemessen 2,8·10⁻³), kein Maß für die
// Rechengenauigkeit.
inline constexpr double kEvalDefocusModel = 1e-2;
// EVAL: Brennlinien GPU gegen CPU, relativ zu max(1 mm, |z_t| + |z_s|). Die Lagen der
// Feldstrahlen weichen um kCpuGpuFp64 ab; die Brennlinie teilt diese Differenz durch den
// Winkelunterschied der Nachbarstrahlen (~2δ·NA ≈ 10⁻⁴ … 10⁻³).
inline constexpr double kEvalFocusGpuCpu = 1e-6;

// PSF (M8-T2): erste Nullstelle der PSF des perfekten Systems gegen 0,61 λ/NA bzw.
// j₁,₁/(2π) λ/NA, relativ (Abnahmekriterium M8-T2).
inline constexpr double kPsfFirstZero = 1e-2;
// PSF: Identitäten der FFT, die nur durch Rundung verletzt werden (Strehl aus der FFT gegen die
// direkte Summe, Parseval, MTF gegen die gezählte Autokorrelation, Strehl 1 des perfekten
// Systems). Fehler der FFT ~ ε·log₂(M²) relativ zur Norm, M <= 1024: ≈ 2·10⁻¹⁵.
inline constexpr double kPsfFftIdentity = 1e-12;
// PSF: perfekte Pupille auf dem Raster N = 128 gegen [2 J₁(v)/v]² (Werte normiert auf das
// Maximum) bis zum dritten dunklen Ring. Modellfehler des Rasters (treppenförmiger Rand,
// Flächenfehler ~N^(−4/3)), kein Maß für die Rechengenauigkeit; beobachtet 7,5·10⁻⁴.
inline constexpr double kPsfAiry = 2e-3;
// PSF: MTF der perfekten Pupille auf dem Raster N = 128 gegen (2/π)(acos ν − ν√(1 − ν²)),
// absolut. Modellfehler des Rasters (gezählte Überlappung zweier Treppenkreise); beobachtet
// 7,2·10⁻⁴.
inline constexpr double kPsfMtfGrid = 2e-3;
// PSF: Strehl gegen die Maréchal-Näherung exp(−(2πσ)²), relativ, für σ <= 0,07 Wellen
// (Abnahmekriterium M8-T2). Der Fehler der Näherung ist s⁴ (κ/12 − 1/4) + O(s⁶) mit s = 2πσ
// und der Kurtosis κ der Phase, für Defokus (κ = 1,8) und sphärische Aberration (κ = 2,14)
// höchstens 0,5 % bei σ = 0,07.
inline constexpr double kPsfMarechal = 1e-2;
// PSF: Koeffizient von s⁴ in Strehl − Maréchal gegen die Entwicklung bis s⁶ aus den Momenten
// der Phase, Schranke kPsfMarechalOrder · s⁴ (Rest O(s⁸)/s⁴); beobachtet höchstens 0,03 · s⁴.
inline constexpr double kPsfMarechalOrder = 0.1;
// PSF: Strehl der Pupille auf dem Raster N = 256 gegen die stetige Kreispupille (Defokus,
// sphärische Aberration, σ <= 0,07 Wellen). Modellfehler des Rasters; beobachtet 5,1·10⁻⁵.
inline constexpr double kPsfGridStrehl = 2e-4;
// PSF: NA der Pupillenabbildung aus den Feldstrahlen (Differenzen im Abstand δ = 10⁻³) gegen
// die exakte NA des aplanatischen Punkts (lineare Abbildung, nur Rundung ~ε/δ).
inline constexpr double kPsfNaFieldRays = 1e-10;
// PSF: GPU (cuFFT) gegen CPU (eigene Radix-2-FFT), absolut in Einheiten des Maximums der
// perfekten PSF bzw. MTF(0) = 1. Andere Zerlegung der FFT und sin/cos der CUDA-Bibliothek
// (<= 2 ulp); erwartet einige ε·log₂(M²) ≈ 10⁻¹⁵, beobachtet 5,6·10⁻¹⁶ (RTX 3090).
inline constexpr double kPsfCpuGpu = 1e-13;

// Gebündelter LM (M3-T4), Abnahme 1: Formfaktor der Einzellinse bei minimaler sphärischer
// Aberration gegen den Planwert 0,74 (Dünnlinsentheorie 2(n² − 1)/(n + 2) = 0,7397). Modellfehler
// der Dünnlinsentheorie (Mittendicke 6 mm bei f = 100 mm verschiebt das Optimum auf q ≈ 0,72),
// kein Maß für die Rechengenauigkeit.
inline constexpr double kLmShapeFactor = 0.05;
// Abnahme 2: Merit nach Rückkehr aus 1 % Störung relativ über dem Optimum. Erreicht wird
// typisch |ΔF|/F ≈ 1e-11 (Rauschgrenze der Merit aus der Rundung der OPD-Residuen ~1e-9).
inline constexpr double kLmReturnRelative = 1e-6;
// Formfaktor nach GPU- gegen CPU-Lauf derselben Aufgabe: Rundungsunterschiede (FMA) ändern den
// Weg; am flachen Optimum (F ändert sich um ~1e-9 relativ, wenn q um ~1e-4 wandert) endet jeder
// Lauf irgendwo in der Rauschzone der Merit.
inline constexpr double kLmShapeAgreement = 1e-3;

// Petzval-Nebenbedingung (M4-T4): Restverletzung (|κ_P| − κ_max)/κ_max nach der
// Penalty-Fortsetzung. Eine Strafe √w·max(0, g) erreicht die Grenze nur asymptotisch
// (Überschreitung ∝ 1/w, Lagrange-Multiplikator λ = 2 w g); beim Gewicht der Datei (1.000 mm² ·
// 1/(F·L·λ_ref²)) beobachtet ~1e-8.
inline constexpr double kPetzvalPenaltyRelative = 1e-6;
// Brennweite nach der Optimierung mit Petzval-Bedingung, |f − f_s| in mm: Gleichgewicht der
// Brennweitenstrafe (Gewicht 0,01/mm² · 1/(F·L·λ_ref²)) gegen Wellenfront und Petzval-Strafe,
// beobachtet ~5e-7.
inline constexpr double kPetzvalFocalLength = 1e-5;

// L1-Strafe der Geisterkoeffizienten (M4-T6), Abnahme Exaktheit: größter Betrag der
// Geisterkoeffizienten am Ende [mm] (Abnahmekriterium). Mit der Glättung bleiben die
// Koeffizienten bei a ≈ −ε g_k/(μ w_k) (docs/optimierung.md, Abschnitt 3); im Abnahmesystem
// (ε = 1e-7 mm, μ etwa 40-fach über der Schwelle) beobachtet ~3e-9.
inline constexpr double kGhostExactCoefficient = 1e-8;
// Stationarität der geglätteten Strafe am Ende, in Einheiten von ε: |a_k − a_k,Vorhersage| / ε
// mit a_Vorhersage = −ε s/√(1 − s²), s = g_k/(μ w_k) und g_k = ∂F_rest/∂a_k aus der FD-Jacobi
// am Endpunkt. Fehler aus dem Abbruch des LM (Schritt ≤ 1e-14 mm) und der FD-Jacobi der
// Wellenfrontspalten (h = 6e-10 mm, relativ ~1e-4 des kleinen Gradienten); beobachtet ≤ 5e-7.
inline constexpr double kGhostStationarity = 1e-4;
// Gewinn der geglätteten Strafe gegenüber dem realen System allein, F_allein − F, gegen die
// Vorhersage Σ_k ε g_k²/(2 μ w_k) mit g_k am Endpunkt, relativ. Die Vorhersage vernachlässigt
// die Kopplung über die übrigen Variablen (die Restkoeffizienten korrigieren z. B. zusätzlich
// die Brennweite); beobachtet 0,3 %.
inline constexpr double kGhostSmoothingGain = 0.05;

// Weiches Versagen (M4-T2), Stetigkeit der Merit entlang einer Geraden durch eine
// Versagensgrenze (10⁴ Punkte): größtes Verhältnis eines Nachbarsprungs |F_{i+1} − F_i| zum
// größeren seiner beiden Nachbarn. Eine stetige Merit mit Knick oder Wurzelsingularität
// F ≈ a + b√(t − t*) (gültige Seite einer Totalreflexion oder eines Verfehlens) ergibt höchstens
// 1/(√2 − 1) ≈ 2,4; ein Sprung ragt um Größenordnungen heraus (beobachtet > 1.000).
inline constexpr double kContinuityNeighborRatio = 10.0;
// Dasselbe: Sprung am Übergang nach Verfeinerung des Intervalls um 10⁴ relativ zum Sprung davor.
// Stetig mindestens wie √Δt (Faktor 1e-2), beobachtet ≤ 3e-3; ein Sprung bleibt (Faktor ≈ 1).
inline constexpr double kContinuityShrink = 0.05;
// Konvexe Hülle (M6-T1): Geometrie der Facetten (Normalen, Abstände der Ecken zur Ebene,
// Volumen) bei Koordinaten der Größe 1; Rundung weniger Operationen (~10 ε).
inline constexpr double kHullGeometry = 1e-12;
// Abstand zur Hülle gegen den analytischen Würfelabstand bzw. die unabhängige Projektion auf alle
// Facetten, Kanten und Ecken, Koordinaten bis ~5: Rundung der Projektion (~10 ε · 5).
inline constexpr double kHullDistance = 1e-12;
// Gradient des Abstands gegen die zentrale Differenz mit h = 1e-6 bei Abstand ≥ 0,05:
// Abbruchfehler h²/(6 d²) ≈ 7e-11, Rundung ε d/h ≈ 1e-9.
inline constexpr double kHullGradientFd = 1e-7;
// Gradient des Strafbeitrags w max(0, φ)² nach den Rohkoordinaten gegen die zentrale Differenz
// (Schritt 1e-6 der Skala, Abstand ≥ 0,05 bzw. im Band des Sicherheitsabstands), relativ zu
// |Ableitung| + 1/s: Der Strafbeitrag ist stückweise quadratisch, die Differenz bis auf Rundung
// (ε·w φ²/h ≈ 1e-10) exakt, außer wenn das Muster die Grenze zweier Stücke überspannt.
inline constexpr double kGlassPenaltyGradientFd = 1e-7;
// Glas-Hülle der Testgläser gegen die unabhängige Python-Referenz (numpy-Fit, scipy/Qhull):
// Volumen und Oberfläche relativ. Die Fits beider Seiten stimmen auf ~1e-13 überein; Volumen und
// Fläche aus ~30 Dreiecken.
inline constexpr double kGlassHullReference = 1e-9;
// Glas-Hülle (M6-T1), Abnahme Optimierung: Abstand des Glases zur Hülle nach der
// Penalty-Fortsetzung (normierte Koordinaten, eine Einheit = Spannweite des Katalogs). Die Strafe
// erreicht den Rand nur asymptotisch (Überschreitung ∝ 1/w, w·d ≈ 0,1 … 0,3); beim Gewicht der
// Datei (10⁸) beobachtet ~2e-9. 1e-6 normiert entspricht ~4e-7 in n₀ und ~2e-7 in ν₁.
inline constexpr double kGlassHullPenaltyDistance = 1e-6;

} // namespace mko::test
