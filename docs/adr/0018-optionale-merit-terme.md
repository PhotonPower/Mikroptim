# ADR-0018: Optionale Merit-Terme – Augmented Lagrangian und toleranzbewusste Merit

- **Status:** angenommen (Entscheidung des Maintainers)
- **Datum:** 2026-11-19

## Kontext

[optimierung-erweiterungen.md](../optimierung-erweiterungen.md) verlangt für neue Merit-Terme eine eigene ADR. Zwei solche Terme sind ohne ADR gemergt worden, weil sie optional und standardmäßig aus sind:

- **Augmented Lagrangian für einseitige Bedingungen** (M11-T2, PR #88; [optimierung.md, Abschnitt 2.6](../optimierung.md#26-augmented-lagrangian-für-einseitige-bedingungen-m11-t2)). Die Strafe w · max(0, g)² erreicht die Grenze nur asymptotisch (g* ≈ λ/(2w)); große w verschlechtern die Kondition und legen den Punkt an den Knick von max(0, ·) (Befund F10).
- **Toleranzbewusste Merit F_tol** (M11-T6, PR #92; [optimierung.md, Abschnitt 14](../optimierung.md#14-toleranzbewusste-optimierung-m11-t6), [optimierung-erweiterungen.md, Abschnitt 6](../optimierung-erweiterungen.md#6-toleranzbewusste-optimierung-m11-t6)). Die Merit bewertet sonst nur das nominale System; von nominal gleichwertigen Lösungen soll die unempfindlichere gewählt werden.

Diese ADR hält die Grundsatzentscheidung für beide nachträglich fest.

## Entscheidung

1. **Augmented Lagrangian.** Jedes einseitige Residuum (paraxiale Bedingungen mit Modus `min`/`max`, Petzval, mechanische Bedingungen) bekommt eine Verschiebung θ ≥ 0 in der Einheit von g:
   r = √w · max(0, g + θ), Fortschreibung nach jedem LM-Lauf θ ← min(max(0, θ + g*), θ_max/√w), Multiplikator λ = 2wθ (Hestenes/Powell, Form mit Verschiebung). Abbruch der äußeren Schleife bei max √w |Δθ| ≤ T oder nach K Läufen. Formeln, Schranke und Wahl von w: optimierung.md §2.6. Gleichungen, L1-Strafe und Glas-Hülle werden nicht verschoben.
2. **Toleranzbewusste Merit.** F_tol(x) = (1 − α) F(x) + (α/S) Σ_s F(x ⊕ δ_s), 0 < α ≤ 1, über feste, deterministische Störungen des Toleranzsatzes (Schema `plus_minus` mit S = 2D oder `samples` mit festem Startwert). Die gestörten Systeme sind gewöhnliche Varianten im Batch (ADR-0004); `ToleranceAwareBackend` setzt A = Σ w_s D_s A_s D_s, g = Σ w_s D_s g_s, F = Σ w_s F_s pro Lauf zusammen. Formeln, Störungen von Variablen und Batch: optimierung.md §14.
3. **Optional, Standard aus.** Schalter: `optimize --augmented-lagrangian K[,T]`, `synth --augmented-lagrangian K[,T[,S]]` (Strahlsuche nicht unterstützt); `optimize --tolerance-aware α`. Ohne Option laufen alle Pfade **bitgleich** wie bisher (θ = 0 addiert nichts; ohne `--tolerance-aware` kein zusätzliches Backend). Ein Wechsel des Standards braucht eine Messung an den Abnahmesystemen und eine neue Entscheidung.
4. **Determinismus** nach ADR-0007: keine `atomicAdd` auf Gleitkommazahlen. Die θ-Fortschreibung ist elementweise ohne Reduktion; g* kommt aus denselben Rechnungen wie die Residuen. Die Zusammensetzung von F_tol hat pro Lauf eine feste Reihenfolge s = 0 … S. Zwei Läufe mit gleichen Eingaben sind bitgleich (CPU, und auf derselben GPU), auch gemeinsam gegen einzeln.
5. **Zustand.** θ ist Zustand des Optimierers bzw. eines Kandidaten (`ConstraintSpec::shift`), **nicht** Teil des Dateiformats `mikroptim.system/1`. Die Störungen von F_tol hängen nur von Toleranzsatz, Schema und Startwert ab und brauchen keinen eigenen Zustand.

## Konsequenzen

- **θ-Persistenz:** Damit ein Lauf mit Augmented Lagrangian fortgesetzt oder reproduziert werden kann, muss θ im Ergebnis- bzw. Neustartzustand stehen. Das setzt ein eigener Folge-PR um; das Systemformat bleibt unverändert.
- **Kosten F_tol** pro LM-Zyklus (optimierung.md §14, Kosten; Zweilinser, Standardsatz, S = 36): GPU (RTX 3090) etwa **2,7-fach**, CPU etwa **71-fach**. Mit S Stichproben skaliert der Faktor etwa mit (1 + S)/(1 + 2D). Die Option ist deshalb für späte Stufen gedacht.
- **Kosten Augmented Lagrangian** (optimierung.md §2.6, Messungen): Petzval 391 statt 546 LM-Zyklen, Mechanik 56 statt 104 bei moderatem festem Gewicht (Datei · 10⁻⁴ bzw. · 10⁻¹; das kleinste Gewicht der Fortsetzung ist zu klein). Am System von M6-T3 allein kein Gewinn (28.637 statt 21.038 Zyklen), mit `--stall 20,1e-3` 1.785 statt 4.931 Zyklen bei kleinerem RMS.
- **Mit F_tol** ist die nominelle Teilmerit F₀ (nicht koaxialer Batch, volle Pupille) bis auf Rundung, aber nicht bitgleich gleich der gewöhnlichen Merit. Augmented Lagrangian und äußeres Strafverfahren laufen unverändert über `ToleranceAwareBackend`.
- **Bekannte Grenzen:**
  - F_tol bewertet gestörte Systeme mit der vollen Merit bei fester Bildlage. Reine Bildverschiebung und Kippung (etwa durch Dezentrierung einer Linse mit Brechkraft) zählen als Verschlechterung, während Tolerierung und Empfindlichkeiten (optimierung.md §11, §12) die Kippung abziehen. Folgeaufgabe M11-T6b: Kippungsabzug bzw. Bildlage-Kompensator, Anbindung an `synth` und die Rangfolge der Strahlsuche.
- **Nachtrag M11-T6b (Umsetzung der Folgeaufgabe):**
  - **Kippungsabzug als Teiloption, Standard unverändert.** `ToleranceAwareSettings::tilt` = `keep` (Standard, bitgleich zu M11-T6) oder `remove` (`--tolerance-tilt remove`): gestörte Teilsysteme s ≥ 1 nach Abzug von Kolben und Kippung pro Tupel mit derselben Anpassung wie die Tolerierung (`PistonTiltFit`, `TiltProjection`, optimierung.md §14 „Kippungsabzug“); das nominelle Teilsystem behält die volle Merit. Nach Punkt 3 dieser ADR wird der Standard nicht ohne Messung an den Abnahmesystemen und neue Entscheidung gewechselt; `remove` ist für Toleranzsätze mit Dezentrierungen und Verkippungen empfohlen.
  - **Kippungsabzug statt Bildlage-Kompensator:** Die Abnahme verlangt Übereinstimmung mit der Bewertung der Monte-Carlo-Tolerierung; ein Kompensator wäre in linearer Näherung dieselbe Projektion, bräuchte aber Variablen pro Störung. Ein Arbeitsabstand-Kompensator (Fokus) pro Störung bleibt offen; F_tol mit `remove` entspricht der Tolerierung ohne Kompensator.
  - **Jacobi-Matrix:** Die Projektion wird auf r_s und jede Spalte von J_s angewandt (Maske am Punkt fest), A_s, g_s und F_s entstehen auf dem Host aus J, das die inneren Backends über die neuen optionalen Methoden `LmBackend::host_jacobians`/`host_residuals` liefern (CPU und GPU). Deterministisch, keine Gleitkomma-Atomics, Läufe gemeinsam und einzeln bitgleich.
  - **Synthese und Strahlsuche:** `synth --tolerance-aware α` (Standard aus) über `OpticalHomotopyOptions::system_optimizer`; Toleranzsatz an jedem Punkt der Mikroptim-Standardsatz der realisierten Beschreibung, weil Einfügungen die Flächen umnummerieren. Die Merit eines Schritts und damit die Rangfolge der Strahlsuche ist F_tol; Bewertung F_B und Ziel bleiben nominal. `--global` (M11-T7) mit `--tolerance-aware` bleibt gesperrt.
  - Augmented Lagrangian: Glas-Hülle nicht verschoben (Vektorresiduum, Richtung innen undefiniert); Fortschreibung pro Kandidat in der Strahlsuche fehlt; Lauf auf der Zielspezifikation (M6-T6) steht aus.

## Verworfene Alternativen

- **Penalty-Fortsetzung mit immer größerem w allein** (bisheriger Standard, bleibt Standard): Restverletzung λ/(2w), schlechte Kondition bei großem w.
- **F_tol als eigener Kernel-Modus statt Varianten im Batch:** Sonderfall in Jacobi und Reduktion ohne Nutzen; über Varianten entsteht die Jacobi-Matrix ohne Sonderfall (FD wie adjungiert).
- **Kompensatoren pro Störung schon in der ersten Version:** eigene Variablen pro Störung; für das Orakel der Abnahme nicht nötig und als Folgeaufgabe abgegrenzt.
- **θ im Dateiformat:** θ ist Optimiererzustand, keine Systembeschreibung; eine Formatänderung von `mikroptim.system/1` wäre unnötig.
