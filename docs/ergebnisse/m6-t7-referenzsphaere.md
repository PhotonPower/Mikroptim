# M6-T7: Referenzsphäre unabhängig von der letzten Fläche

Bericht zur Abnahme von M6-T7 ([Meilensteinplan](../plan/meilensteine.md#m6-t7--referenzsphäre-unabhängig-von-der-letzten-fläche), Entscheidung: [ADR-0017](../adr/0017-referenzsphaere-bildpunkt.md), Formeln: [numerik.md, Abschnitt 7](../numerik.md#referenzsphäre-um-den-bildpunkt-m6-t7-adr-0017)). Läufe auf dem Rechner des Maintainers (RTX 3090, geteilt mit anderen Agenten), Release-Build mit CUDA 12.9.

**Kurzfassung.**

- Die Merit kann die OPD jetzt auf einer Sphäre um den idealen Bildpunkt messen, deren Radius nicht an der letzten Fläche hängt: bis zur paraxialen Austrittspupille (`image_exit_pupil`) oder unendlich (`image_infinite`). Dateien ohne das neue Feld `reference_sphere` rechnen bitgleich wie bisher (568 von 568 Fingerabdrücken gleich).
- Mit der neuen Konvention erreicht der Versuch „Apertur zuerst auf der Achse“ (M11-T4) **NA 0,95 ohne die Hilfsbedingung gegen F12** und ohne die Falle: am Endpunkt 0,06–0,17 Wellen RMS auf der Achse (Bewertung durch die Austrittspupille), paraxialer Fokus 1–4 µm neben der Bildebene. Bisher: NA 0,37 mit Hilfsbedingung, NA 0,92 nur über die Falle (76 Wellen auf der Achse).
- Die Hilfsbedingung `paraxial_focus` (F12-Schutz) ist mit der neuen Konvention **nicht mehr nötig**; mit ihr endet der Lauf ebenfalls bei NA 0,95, aber mit schlechterem Achswert (0,19 statt 0,095 Wellen nach der Nachoptimierung).
- Der Vergleichszweig von M6-T6 („Apertur zuerst“ bei 10 % Feld), der damals NA 0,94 nur in der Falle erreichte (reales System 21 Wellen), bleibt mit `image_infinite` ehrlich bei NA 0,47 stehen, dort aber beugungsbegrenzt (0,048/0,061/0,068 Wellen bei 0/7/10 % Feld).
- Das Feld bleibt das Problem der Achs-Synthese: bei NA 0,95 nach Nachoptimierung nur auf der Achse 1,4–2,6 Wellen bei 7 % und 3,7–13 Wellen bei 10 % Feld.

## 1. Gewählte Definition

| Größe | Wahl | Begründung |
|---|---|---|
| Mittelpunkt | idealer Bildpunkt **M**_f = (0, h_f, w) in der Bildebene | Hauptstrahl-Durchstoßpunkt und Schwerpunkt unterscheiden sich nur um eine Verschiebung in der Bildebene, die als Kippung der OPD erscheint; die Merit soll diese Kippung (Verzeichnung, Brennweitenfehler) weiter bestrafen. Kein Bezug auf einen einzelnen, eventuell versagenden Strahl, keine zusätzliche Ableitung. |
| Radius | `image_exit_pupil`: \|**M**_f − **E**\| bis zur paraxialen Austrittspupille bei λ_ref; `image_infinite`: ∞ | Vorschlag des Maintainers (Austrittspupille) und die robuste Variante: Jede endliche Sphäre hat ihre Falle im Punkt **E** (eine Kugelwelle um **E** trifft die Sphäre nur dort). Bei Geisterplatten liegt die paraxiale Pupille oft nahe am Bild (M6-T6, Befund F11: z_AP = 0,73 mm bei w ≈ 0,5 mm). Bei κ = 0 gibt es weder Falle noch Verfehlen. |
| κ | beim Aufbau des Batches eingefroren (`update_reference_curvature`), ∂κ/∂x = 0 | wie das Feldziel h_f; FD- und adjungierte Jacobi bleiben exakt konsistent, der Unterschied zur mitlaufenden Pupille ist von zweiter Ordnung in der Queraberration. |
| Bildlage | w = `thickness[S−1]`, wie bisher als Variable möglich | Modus „paraxialer Fokus“ zurückgestellt: nähme dem Optimierer den Defokus zum Ausgleich der Restsphärik und bräuchte die Ableitung der Schnittweite in der adjungierten Jacobi; `paraxial_focus` deckt den Bedarf weich ab. |
| Kompatibilität | Datei ohne Feld = `last_surface` | keine stillen Ergebnisänderungen bestehender Dateien; neue Läufe setzen das Feld (oder `--reference-sphere`). |

## 2. Orakel und Tests

| Kriterium | Test | Ergebnis |
|---|---|---|
| Schritt gleich dem EVAL-Schritt durch die Austrittspupille | `ImageReference.StepEqualsExitPupilStepOfEval` | 20.000 Strahlen, OPL bitgleich, Status gleich (4.272 verfehlt); Fußpunkt \|(**p** − **M**)·**d**\| ≤ 1,2·10⁻¹⁴ mm |
| Reine Defokussierung (Punktquelle in Luft, analytisch OPL = n(**M** − **q**)·**d** bzw. mit κ) | `ImageReference.PureDefocusOfPointSourceIsAnalytic` | 729 Strahlen, drei Ebenenfolgen, κ = 0, 1/40, −1/55: max \|OPL − analytisch\| = 1,4·10⁻¹⁴ mm (`kReferenceSphereFp64` = 10⁻¹²) |
| Falle F12: ideal fokussierende Geisterplatte (NA 0,3), Hilfsebene im Brennpunkt, Bildebene 2,7 mm dahinter | `ImageReference.FocusInAuxiliaryPlaneIsNoLongerHidden` | alt: RMS 2,0·10⁻¹⁴ mm (Defokus versteckt); neu ∞: RMS 3,589·10⁻² mm = analytischer Defokus, max \|ΔOPD\| 1,6·10⁻¹⁴ mm; neu AP: 2,843·10⁻² mm, max \|ΔOPD\| 1,8·10⁻¹⁵ mm |
| Hilfsebene in Luft hinter dem System (Bildebene im Raum fest), 20 % und 75 % des Arbeitsabstands | `ImageReference.AuxiliaryPlaneBehindSystemDoesNotChangeOpd` | Einzellinse / Mehrlinser / Einzellinse mit Geisterplatten: neu max \|ΔOPD\| ≤ 1,3·10⁻¹³ mm, κ relativ ≤ 2,3·10⁻¹⁶; alt 1,3·10⁻³ / 2,5·10⁻⁴ / 8,0·10⁻⁴ mm |
| κ aus der Austrittspupille wie EVAL, eingefroren | `ImageReference.CurvatureFromParaxialExitPupilIsFrozen` | bitgleich zu `eval_reference_geometry`; `VariableSet::apply` ändert κ nicht |
| Merit = Σ ŵ ŵ (RMS/λ)² aus `evaluate_wavefront` | `ImageReference.MeritEqualsWeightedSumOfWavefrontRms` | 6 Fälle × 2 Radien: ≤ 5,6·10⁻¹⁶ relativ |
| Verfehlte Sphäre hart/weich, Dateiformat, EVAL `infinite` | `ImageReference.MissedSphereHardAndSoft`, `.FileFormatAndCompatibility`, `.EvalInfiniteReference`, `CliEval.InfiniteReferenceAndDefaultFromFile` | bestanden |
| Adjungierter Schritt gegen Richardson (κ = 0, κ > 0, κ < 0, verfehlt) | `AdjointLocal.ImageStepMatchesRichardson` | je 200 Fälle, max 1,1·10⁻¹⁰ relativ (`kAdjointLocalRichardson` = 10⁻⁷) |
| Adjungierte Jacobi gegen Richardson und FD (Geisterplatten, Punktquelle, Mehrlinser mit paraxialen Bedingungen, weiches Versagen, verfehlte Sphäre; Arbeitsabstand als Variable) | `AdjointJacobian.ImageReferenceAgainstRichardsonAndFd` | Trace-Zeilen gegen Richardson pro Spalte ≤ 1,0·10⁻⁷, skaliert ≤ 1,9·10⁻⁸; gegen FD-Jacobi gleich deren eigenem Fehler (≤ 1,6·10⁻⁶ bei weichem Versagen); r₀ bitgleich |
| GPU gegen CPU: RESIDUALS (weich/hart, auch mit kleinem Radius und verfehlter Sphäre) | `CudaResiduals.MatchCpuWithImageReference` | 6 Fälle, 1,4·10⁶ Strahlen, max \|ΔOPD\| 2,7·10⁻¹¹ mm (Fälle mit weich fortgesetzter Sphäre, `kCpuGpuSoftTuple`), sonst ≤ 7,5·10⁻¹² mm |
| GPU gegen CPU: MERIT | `CudaMerit.MatchesCpuWithImageReference` | max \|√F_GPU − √F_CPU\| 1,5·10⁻¹⁰ Wellen |
| df64 gegen FP64 | `CudaDf64.MatchesFp64ForPerturbedVariants` | max \|ΔOPD\| 1,4·10⁻¹¹ mm, 0 Statuswechsel |
| FP32-Vorfilter | `Prefilter.*`, `CudaPrefilter.MatchesCpuReference` | 0 falsch aussortiert; Klassen GPU/CPU gleich in 300 von 300 Varianten |
| EVAL `infinite` GPU gegen CPU | `CudaEval.MatchesCpu` | max \|ΔOPD\| 8,2·10⁻¹³ mm |
| Adjungierte Jacobi GPU | `CudaAdjointJacobian.ImageReferenceMatchesCpuFdAndDf64` | gegen CPU ≤ 2,1·10⁻¹¹, df64 ≤ 1,8·10⁻¹⁰, gegen FD-GPU gleich dem FD-Fehler |
| Alte Konvention bitgleich zu main | `CoaxialFingerprint.*`, `CudaCoaxialFingerprint.*` auf main und auf dem Zweig | 568 von 568 Prüfsummen gleich |

## 3. Messung: „Apertur zuerst auf der Achse“ (`--osc-versuch`)

Ablauf wie M11-T4 ([Bericht](m11-t4-sinusbedingung.md)): Start `zielspezifikation_start.json`, nur Achsfeld, 546 nm, NA 0,3 → 0,95, höchstens 12 Runden mit Nachoptimierung und zwei erzwungenen Einfügungen; Bewertung bei 0/7/10 % Feld mit `mikroptim eval --grid 256 --reference exit-pupil` am Endpunkt (mit Geisterplatten) und nach Entfernen der Geisterplatten und Hilfsebenen und 2000 LM-Zyklen auf der Achse. Neu: `--referenzsphaere K` setzt `reference_sphere` in der Startdatei, `--ohne-f12-schutz` lässt `paraxial_focus` weg.

```
python tools/m6t6_zielspezifikation.py --osc-versuch --gpu --referenzsphaere image_infinite --ohne-f12-schutz [--osc-gewicht 0]
```

RMS in Wellen bei 546 nm (Referenz Austrittspupille), OSC in Zone 1/√2 und am Rand:

| Lauf | Referenzsphäre | F12-Schutz | Residuum OSC | erreichte NA | Elemente | Endpunkt 0/7/10 % | nachoptimiert 0/7/10 % | OSC nachoptimiert | Laufzeit |
|---|---|---|---|---|---|---|---|---|---|
| M11-T4 | letzte Fläche | ja | ohne | 0,375 | 4 | 1,08 / 14,6 / 20,9 | 0,053 / 0,064 / 0,073 | −0,007 / −0,014 | 72 s |
| M11-T4 | letzte Fläche | ja | mit | 0,365 | 6 | 0,79 / 0,80 / 0,80 | 0,085 / 0,090 / 0,094 | −5·10⁻⁴ / 3·10⁻⁴ | 120 s |
| M11-T4 | letzte Fläche | nein | mit | 0,9225 (Falle F12) | 9 | 76,3 / 78,6 / 81,0 | 0,155 / 0,443 / 0,916 | – | – |
| **A** | Bildpunkt, ∞ | nein | mit | **0,95** | 8 | 0,086 / 19,9 / 58,7 | **0,095** / 1,99 / 13,2 | 0,013 / 0,009 | 40 min |
| **B** | Bildpunkt, ∞ | nein | ohne | **0,95** | 10 | 0,174 / 2,57 / 3,68 | 0,168 / 2,57 / 3,67 (ohne Kippung 0,95 / 1,35) | 0,11 / 0,24 | 41 min |
| **C** | Bildpunkt, AP | nein | mit | **0,95** | 9 | 0,059 / 29,4 / 34,1 | **0,060** / 1,45 / 8,79 | 0,008 / 0,009 | 26 min |
| **E** | Bildpunkt, ∞ | ja | mit | 0,95 | 10 | 0,301 / 9,95 / 21,1 | 0,193 / 14,5 / 17,9 | 0,010 / −7·10⁻⁴ | 28 min |

**Keine Falle.** Der paraxiale Fokus der Endpunkte liegt 1,7 µm (A) bzw. 4,0 µm (B) hinter der Bildebene, C 0,09 µm; nach der Nachoptimierung 1,3 µm (A), 4,2 µm (B), 0,07 µm (C) – bisher 2,69 mm in der Falle. Der Achswert der Bewertung durch die Austrittspupille stimmt mit der Merit überein (A: Merit-RMS 0,055 Wellen auf der Achse mit `mikroptim trace`, Bewertung 0,086 auf dem 256er-Raster; bisher 0,076 gegen 151,7 Wellen).

**F12-Schutz entbehrlich.** Mit `paraxial_focus` (E) erreicht der Lauf ebenfalls NA 0,95, braucht aber zwei Elemente mehr und ist auf der Achse schlechter (0,19 statt 0,095 Wellen nach der Nachoptimierung); die Bedingung hält die Bildebene im paraxialen Fokus, obwohl bei NA 0,95 der beste Fokus wegen der Restsphärik davon abweicht. Das Skript setzt sie nur noch ohne `--ohne-f12-schutz`, Standard bleibt das alte Verhalten (Nachrechenbarkeit von M11-T4).

**Zwei Radien.** `image_exit_pupil` (C) und `image_infinite` (A) kommen beide zu NA 0,95; C endet mit dem besten Achswert (0,060 Wellen, beugungsbegrenzt) und besseren Feldwerten. Bei C stand der Controller ab Runde 9 bei NA 0,95 mit „Ziel nicht erreicht“ (Merit über der Annahmeschwelle), das Ergebnis ist die letzte Nachoptimierung. Eine Falle in **E** trat nicht auf, obwohl die paraxiale Austrittspupille des Endpunkts (mit Geisterplatten) nur 0,58 mm hinter der letzten Fläche liegt, also 0,43 mm hinter dem Bild (R ≈ 0,43 mm, σ = −1; F11): Der Fokus liegt 0,09 µm neben der Bildebene. Dass die Sphäre so klein werden kann, ist der Grund, `image_infinite` für die Synthese zu empfehlen.

**Feld.** Die Läufe optimieren nur die Achse; das Feld wird nur bewertet. Mit OSC-Residuum (A, C) bleibt die Koma klein (OSC ≤ 0,013), trotzdem wachsen die Fehler bei 7–10 % Feld auf einige bis über zehn Wellen (Astigmatismus und Bildfeldwölbung der Geisterplatten-Lösung; ohne Kippung kaum kleiner). Ohne Residuum (B) ist der Feldfehler kleiner, die OSC aber 0,1–0,24. Das Feld gehört in die Synthese (M6-T6, Stufen mit Feld); diese Läufe zeigen nur, dass die Apertur jetzt nicht mehr an der Referenzsphäre scheitert.

## 4. M6-T6-Teillauf: Vergleichszweig „Apertur zuerst“ bei 10 % Feld

`python tools/m6t6_zielspezifikation.py --apertur-zweig --gpu --referenzsphaere image_infinite` (Strahlsuche B = 2, zwei Varianten, ohne erzwungenes Einfügen wie in M6-T6):

| Lauf | erreichte NA (Merit) | Elemente | reales System nachoptimiert, RMS 0/7/10 % [Wellen] |
|---|---|---|---|
| M6-T6 (letzte Fläche) | 0,94 (Falle F12) | 4 | 21 auf der Achse (bis 120 bei vollem Feld) |
| M6-T7, `image_infinite` | 0,4725 | 4 | 0,048 / 0,061 / 0,068 |

Der Zweig bleibt nach vier Runden stehen („kein Fortschritt, Abbruch“, das Skript fügt in diesem Zweig nichts erzwungen ein), aber das Ergebnis ist ehrlich: Merit und reales System stimmen überein. Die Bewertung bei vollem Feld (3–4°) hat versagte Strahlen und keine PSF (Feldstrahl versagt); sie ist für einen bei 10 % Feld entworfenen Zweig nicht aussagekräftig. Ein vollständiger M6-T6-Hauptlauf (mehrere Stunden) mit der neuen Konvention steht aus; er ist die nächste Messung.

## 5. Folgerungen

- Neue Läufe und Skripte sollten `reference_sphere` setzen; für die Synthese mit Geisterplatten `image_infinite` (keine Falle, kein Verfehlen), für die Endbewertung bleibt EVAL durch die Austrittspupille (`exit-pupil`) bzw. `infinite` für denselben Bezug wie die Merit.
- Der F12-Schutz im Skript kann mit der neuen Konvention entfallen (`--ohne-f12-schutz`); er ist für die alte Konvention als Standard belassen, damit M11-T4 nachrechenbar bleibt.
- Offen: Standard der Datei für neu erzeugte Startsysteme (heute nur ausdrücklich), Hauptlauf M6-T6 mit neuer Konvention, Feldstufen bei hoher NA.

## Dateien

| Datei | Inhalt |
|---|---|
| [`tools/m6t6_zielspezifikation.py`](../../tools/m6t6_zielspezifikation.py) | `--referenzsphaere K`, `--ohne-f12-schutz` (OSC-Versuch und Vergleichszweig) |
| `docs/ergebnisse/m6-t7/{A,B,C,E}_*/m6t7_osc.json` | Zusammenfassung der Läufe A, B, C, E (Runden, Punkt, RMS und OSC am Endpunkt und nachoptimiert) |
| `docs/ergebnisse/m6-t7/{A,B,C,E}_*/o0_start.json`, `o1_final.json`, `o2.json` | Start, Ergebnis der Synthese, nachoptimiertes System |
| `docs/ergebnisse/m6-t7/D_apertur_zweig_image_infinite/` | Start, nachoptimiertes System und Bewertung des Vergleichszweigs |
