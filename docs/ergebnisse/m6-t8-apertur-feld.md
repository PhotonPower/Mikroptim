# M6-T8: Gemeinsame Apertur-/Feld-Homotopie und Feldregel (Befunde F14/F15)

Aufgabe: [Meilensteinplan](../plan/meilensteine.md#m6-t8--gemeinsame-apertur-feld-homotopie-und-feldregel). Ausgangslage: [Bericht M6-T6b](m6-t6b-zielspezifikation-adr0017.md), Befund **F14** (ab der Einzellinse endet die Feldstufe immer bei 27 % Feld, „keine Geisterplatte aktiv“) und **F15** (Aperturstufe nach den Farbstufen ohne Fortschritt). Alle Zahlen aus Läufen auf dem Rechner des Maintainers (RTX 3090, geteilt mit anderen Agenten, FP64, Release mit CUDA 12.9), Referenzsphäre `image_infinite`, 16 Ringe × 8 Speichen, FD-Jacobi. Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json` – die Systeme sind nicht herstellbar; es werden keine Linsendaten realer Objektive verwendet.

**Kurzfassung.**

- **Ursache von F14** (Abschnitt 1): Die Feldfehler (Astigmatismus mit Koma, F_B ∝ φ⁴) sind für die Geisterplatten unsichtbar – vorne liegen sie blendennah (|ȳ/y| = 0,18), hinten ist ihr Normierungsradius 1,42 mm sechsmal größer als das Bündel (r_g wird nur mit der NA skaliert, das System ist zum Objekt hin geschrumpft). Selbst mit μ/100 senken sie F nicht. Ohne Bedarf wächst das System in der Feldstufe nie; die „27 %“ sind der letzte Rasterpunkt der Schrittweitensteuerung vor F_B = F_max. Ein einzelnes zusätzliches Element in irgendeinem Spalt reicht nicht (bestenfalls −18 % in F bei 35 % Feld).
- **Umgesetzt** (alle Standard aus, ohne Option bitgleich): (a) gemeinsamer Homotopie-Parameter für Apertur und Feld (`synth --joint-aperture-field`, Bahn NA(t), φ(t) = φ₀ + t^p (φ₁ − φ₀)); (b) **Feldregel** (`--field-rule`): Operator A in der Feldlinsenlage (größtes |ȳ/y| aus dem Aberrationsbericht), wenn die Platten keinen Bedarf melden und Feldfehler dominieren; dazu (c) **Einfügen nur mit Verbesserung** (`--insertion-improve N`, aus den ersten Messungen, F16).
- **Ergebnis:** Mit der Feldregel erreicht die Feldstufe ab der Einzellinse bei NA 0,5 erstmals **100 % Feld** (F1); nachoptimiert ohne Geisterplatten, monochromatisch, RMS **0,039/0,022/0,059 Wellen** bei 0/70/100 % Feld, Strehl 0,94/0,98/0,88 – beugungsbegrenzt, aber mit **25 Elementen** (F16; der 9-Linser aus M6-T6 hat 0,015/0,024/0,033 mit 9). Ab dem NA-0,855-System aus M6-T6b kommt die Feldstufe mit gemeinsamer Homotopie, Feldregel und Verbesserungsprüfung von 10 % auf **23 % Feld bei NA 0,869** (25 El., 0,022/0,054/0,087 Wellen am Teilfeld) – mehr als vorher (10 %), aber nicht über 27 %. Die gemeinsame Homotopie allein öffnet ab der Einzellinse das Feld auf 37 % (NA 0,36, 4 El.) und bleibt dann stehen.
- **F15** (Apertur nach den Farbstufen) ist nicht bearbeitet (Abschnitt 4).

## 1. Diagnose von F14

Untersucht am Zwischenstand von V1 aus M6-T6b am Ende der Feldstufe (`docs/ergebnisse/m6-t6b/V1_basis/p2_final.json`: 6 Glaselemente, NA 0,5, erreichter Feldanteil 0,2719), jeweils mit `mikroptim optimize --gpu --max-iterations 1000 --rings 16` am angegebenen Feldanteil und `mikroptim aberrations` (Seidel, paraxial).

**(1) Die 27 % sind ein Raster der Schrittweitensteuerung, kein harter Rand.** Die Feldphase beginnt mit Δ₀ = 1/8 ab 10 %: 0,225 angenommen, 0,35 abgelehnt, 0,2875 abgelehnt, 0,256 angenommen, 0,287 abgelehnt, 0,272 angenommen, 0,287 abgelehnt → Δ = 1/128 < Δ_min = 1/64 → Diagnose ohne Bedarf → `step_too_small`. Darum endet die Stufe in V1, V3, V5 und V9 am selben Punkt 0,2719: Die eigentliche Grenze (F_B = F_max) liegt in allen Läufen zwischen 0,272 und 0,287, unabhängig von NA (0,4/0,5) und Elementzahl (3–10).

**(2) F_B wächst wie φ⁴ (Astigmatismus).** Optimiert: F = 4,74·10⁻³ bei 0,272, 1,25·10⁻² bei 0,35 (Schwelle F_max = 5,10·10⁻³). (0,35/0,272)⁴ = 2,7, gemessen 2,6. Seidel-Summen bei 0,272 (Wellen bei 546 nm): W040 = 0,34, W131 = −0,22, **W222 = 0,31**, W220 = 0,004, W311 = 0,38 (Verzeichnung, von höheren Ordnungen kompensiert: Kippung am Teilfeldrand vernachlässigbar). Der Petzvalradius liegt auf der Grenze 130 mm; die Bildfeldwölbung ist korrigiert, übrig bleibt Astigmatismus mit Koma. Bei vollem Feld wären das W222 ≈ 4 Wellen.

**(3) Die Geisterplatten sehen den Feldfehler nicht.** Das Wörterbuch liegt an zwei Lagen: (a) 0,1 mm hinter der Hilfsebene vor der ersten Linse, |ȳ/y| = 0,18 (blendennah: eine r⁴-Asphäre dort erzeugt W222/W040 = 4(ȳ/y)² ≈ 0,13 – Astigmatismus nur zusammen mit viel sphärischer Aberration); (b) 0,1 mm hinter der letzten Linse vor dem Objekt, |ȳ/y| ≈ 0,65, aber mit Normierungsradius r_g = 1,42 mm bei einem Bündel von Randstrahl 0,15 mm + Hauptstrahl 0,10 mm: r_g wird nur mit der NA skaliert, das System ist seit dem Start zum Objekt hin geschrumpft. Mit r/r_g ≈ 0,18 trägt die Ordnung 2 nur (0,18)⁴ ≈ 10⁻³ ihres Koeffizienten bei; die Platten sind tot (Gewinn F(a = 0) − F = 5·10⁻⁶ Wellen²).

| Versuch bei Feldanteil 0,35 | F [Wellen²] | RMS 0/70/100 % des Teilfelds [Wellen] | aktive Platten |
|---|---|---|---|
| Ausgangslage (0,272 zum Vergleich) | 4,74·10⁻³ | 0,039/0,070/0,087 | keine |
| Ausgangslage | 1,25·10⁻² | 0,067/0,113/0,139 | keine |
| μ aller Platten × 0,01 | 1,28·10⁻² | 0,065/0,110/0,143 | vorne 1, hinten 3 (dominant Ordnungen 4–8), Empfehlung A hinten |
| hintere Platten mit r_g = 0,3 mm (Bündel) | 1,48·10⁻² | 0,073/0,100/0,162 | keine |
| ohne Petzval-Bedingung | 6,1·10⁻³ (`max_iterations`) | 0,047/0,090/0,088 | keine |

Selbst mit hundertfach billigeren Platten sinkt F nicht: Die Platten werden aktiv, tauschen aber nur Korrektur mit den Linsen; an diesen beiden Lagen kann das rotationssymmetrische Profil den Astigmatismus nicht darstellen. Der Bedarf nach F_B ist also echt (F_B > F_max), die Diagnose der Platten kann ihn aber grundsätzlich nicht melden – das ist F3 in der Form von F4. Die Petzval-Bedingung kostet bei 35 % Feld die Hälfte von F; ohne sie läge die Grenze nur wenig höher (6,1·10⁻³ > F_max).

**(4) Ein zusätzliches Element in einem beliebigen Spalt reicht nicht.** Meniskus (SYN-36, 0,3 mm, Krümmungen und Dicke frei) in jedem Luftspalt bei 35 % Feld, danach `optimize`: bestenfalls F = 1,03·10⁻² (Spalt 9), Blendenspalt 1,10·10⁻², Spalt 11 1,14·10⁻², Spalt 1 1,24·10⁻², Spalt 7 1,46·10⁻²; der Spalt vor der Hilfsebene am Objekt brach mit dieser groben Startdicke (F = 165). Das erklärt V5 (fünf erzwungene Planplatten in den mittleren Spalten, kein Fortschritt): Jedes Element bringt wenig, und die Feldphase misst Fortschritt erst ab Δ_min = 1/64 – ein Gewinn von wenigen Prozent im Feld wird nach dem Einfügen mit Δ₀ = 1/8 gar nicht sichtbar.

**(5) Der 9-Linser aus M6-T6 ist anders gebaut.** V6-Z2 (volles Feld, 0,033 Wellen) hat dicke Elemente (2,1; 4,0; 4,5 mm) und weite Luftspalte (2,2–2,5 mm); V1 ist kompakt mit 0,5-mm-Menisken. Der alte Weg führte über viele Einfügungen an der bildnahen Lage (M6-T6, p2: vier Menisken hinter den Platten 3–5) unter der alten Merit, die den Fehler um den Faktor 1,3–2 unterschätzte (F11) – die Schwelle war dort effektiv 1,7- bis 4-mal weiter.

**Folgerung für die Umsetzung.** Die Ursache ist der Weg des Controllers: Im Feld wächst das System nur, wenn die Geisterplatten Bedarf melden, und das tun sie für Feldfehler nicht. Daraus:

- **(b) Feldregel** (Bedarfskriterium für Feldfehler): Melden die Platten keinen Bedarf und dominieren Feldfehler (RMS am Rand ≥ 1,5 × Achse), fügt der Controller Operator A in der Feldlinsenlage ein (Spalt mit größtem |ȳ/y|, hier der Spalt vor dem Objekt) – dort, wo die Platten (b) wirken müssten, aber wegen r_g nicht können. Ein Vorversuch am V1-Zwischenstand bestätigt die Wahl: Der erste Meniskus der Regel hinter der Frontlinse senkte F_B bei 27,2 % Feld von 4,7·10⁻³ auf 2,5·10⁻³, die Feldstufe kam auf 31,9 % (vorher 27,2 %); der zweite Meniskus im 0,1-mm-Spalt dahinter kollidierte mit den Rändern (Randdicken-Strafe, F = 25 Wellen²) – daher die Mindestdicke 2·`min_air_gap` des Spalts.
- **(a) Gemeinsame Homotopie**: Weil die Feldgrenze ab der Einzellinse von NA und Elementzahl kaum abhängt (V9), allein nicht ausreichend; zusammen mit der Feldregel aber sinnvoll, damit die Feldfehler wachsen, während das System noch wächst und die Elemente der Apertur schon für das Feld mitgewählt werden (F5: keine Reserve, wenn das Feld erst bei NA 0,5 dazukommt).

## 2. Umsetzung

Festlegungen in [optimierung.md, „Gemeinsame Apertur-/Feld-Homotopie und Feldregel (M6-T8)“](../optimierung.md#gemeinsame-apertur-feld-homotopie-und-feldregel-m6-t8), Kommandozeile in [cli.md](../cli.md#mikroptim-synth). Kurz:

| Option | Controller / Modell | Wirkung |
|---|---|---|
| `--joint-aperture-field [--joint-field-exponent P] [--joint-step D0,DMIN,DMAX]` | `HomotopyOptions::joint_aperture_field` | eine Phase `aperture_field` über t ∈ [0, 1], NA linear, Feld mit t^P |
| `--field-rule [--field-ratio F] [--field-min-rms W] [--field-glass G]` | `OpticalHomotopyOptions::field_rule` | ohne Geisterbedarf und bei RMS(Rand) ≥ 1,5 · RMS(Achse): Operator A im Spalt mit größtem \|ȳ/y\| (Spalt ≥ 2·`min_air_gap`) |
| `--insertion-improve N` | `HomotopyOptions::insertion_must_improve` | Einfügen in den stetigen Phasen nur, wenn F_B danach kleiner ist; sonst bis zu N Varianten (Feldregel: nächste Spalte; Geisterbedarf: andere Lagen/Gläser) |

Skript [`tools/m6t6_zielspezifikation.py`](../../tools/m6t6_zielspezifikation.py): `--gemeinsam [--gemeinsam-exponent P]` (Stufen 1 und 2 als eine gemeinsame Phase vom Start nach (`--na-feld`, 100 %), Runden und erzwungenes Einfügen wie p1 + p2), `--feldregel`, `--einfuegen-verbessern N`, `--ab-p2` (Startdatei ist das Ergebnis der Stufe 1, z. B. `m6-t6b/V2_apertur_zuerst/p1_final.json`), `--bis p2` (nach Feldstufe und Zwischenkandidat Z2 enden; Z2 bei Teilfeld zusätzlich bei vollem Feld bewertet); `--archiv` liest `lauf.log` auch in der ANSI-Codepage.

**Bitgleichheit ohne Option:** Unit-Tests (Abschnitt 5) und ein Lauf am echten System: `synth` wie `V1_basis/p1_r0` aus M6-T6b (Start `p0_start.json`, gleiche Optionen, GPU) mit dem neuen Programm – alle 51 Zeilen des Schrittprotokolls (Punkt, Δ, Status, Zyklen, F, F_B, RMS, Σ|a|, Elemente, Entscheidung) gleich dem archivierten `p1_r0.txt`.

## 3. Messung

Alle Läufe `python tools/m6t6_zielspezifikation.py --gpu --bis p2 …` (Ausgabe `docs/ergebnisse/m6-t8/<Lauf>`), FD-Jacobi, 16 Ringe, `--stall 0`, höchstens 24 Elemente pro synth-Lauf. Z2 = Ende der Feldstufe ohne Geisterplatten und Hilfsebenen monochromatisch nachoptimiert (2.000 Zyklen) und mit `mikroptim eval --grid 256 --reference exit-pupil` bewertet; RMS in Wellen bei 546 nm am **erreichten** Feldanteil (Felder 0/70/100 % davon), Strehl aus der FFT. Die Basis-Läufe sind V1 und V2 aus M6-T6b (deterministisch, siehe Bitgleichheit). Zeit = Summe der Programmaufrufe (GPU geteilt mit bis zu vier Läufen und anderen Agenten).

| Lauf | Start | Optionen | Ende Feldstufe: NA / Feld | El. | Z2: RMS 0/70/100 % des Teilfelds [Wellen] | Strehl (FFT) | Z2 bei vollem Feld: RMS 70/100 % | Zeit [s] |
|---|---|---|---|---|---|---|---|---|
| **V1** (M6-T6b, Basis) | Einzellinse | – | 0,50 / **0,272** | 6 | 0,028/0,050/0,044 | 0,97/0,91/0,93 | – | 2.732 (ganzer Lauf) |
| **F1** | Einzellinse | Feldregel | 0,50 / **1,00** | 25 | **0,039/0,022/0,059** | 0,94/0,98/0,88 | (= Teilfeld) | 4.426 |
| FV1 | Einzellinse | Feldregel, Verbesserung 4 | 0,50 / 0,631 | 24 | 0,049/0,061/0,102 | 0,91/0,87/0,66 | 0,16/0,69 | 5.831 |
| J1 | Einzellinse | gemeinsam (Diagonale, Ziel NA 0,5) | 0,359 / 0,367 | 4 | 0,050/0,068/0,057 | 0,91/0,83/0,88 | 0,67/2,5 | 1.069 |
| JFV1 | Einzellinse | gemeinsam, Feldregel, Verbesserung 4 | 0,376 / 0,442 | 2 | 0,102/0,109/0,111 | 0,66/0,62/0,62 | 0,47/1,7 | 485 |
| JF1 | Einzellinse | gemeinsam, Feldregel | 0,359 / 0,367 (Runde 0) | 11 | – (abgebrochen) | – | – | – |
| **V2** (M6-T6b, Basis) | Apertur zuerst, NA 0,855, 5 El. | – | 0,855 / **0,10** | 18 | (V2-M1: 0,048/0,047/0,072) | 0,91/0,92/0,83 | 4,2/9,7 | 9.204 (ganzer Lauf) |
| F855 | V2 `p1_final` | Feldregel | 0,855 / 0,10 | 26 | 0,26/0,37/0,60 | 0,10/0,10/0,05 | 28/36 | 1.868 |
| FV855 | V2 `p1_final` | Feldregel, Verbesserung 4 | 0,855 / 0,10 | 5 | 0,048/0,046/0,072 | 0,91/0,92/0,82 | 4,3/9,8 | 1.110 |
| **JFV855** | V2 `p1_final` | gemeinsam bis (0,95; 100 %), Feldregel, Verbesserung 4 | **0,869 / 0,231** | 25 | 0,022/0,054/0,087 | 0,98/0,89/0,78 | 6,0/6,7 | 5.996 |
| JF855 | V2 `p1_final` | gemeinsam, Feldregel | – (abgebrochen in Runde 0, Einfüge-Serie) | – | – | – | – | – |

JF1 und JF855 (ohne Verbesserungsprüfung) wurden abgebrochen, nachdem sich die Einfüge-Serien abzeichneten (JF1: Runde 0 endete wie J1 bei (0,359; 0,367), aber mit 11 statt 4 Elementen); sie waren der Anlass für `--insertion-improve`. Nur die Textprotokolle `p2_r0.txt` liegen bei.

**Einfügungen** (Versuche in den synth-Läufen der Feldstufe, „verworfen“ = von der Verbesserungsprüfung zurückgenommen):

| Lauf | Feldregel | Geisterbedarf | verworfen | Bemerkung |
|---|---|---|---|---|
| F1 | 13 | 6 | – | Runde 0: 0,27 → 0,71 mit 24 El. (Höchstzahl), danach 0,71 → 1,00 in sieben Runden nur über das Freigeben der Spalte (Skript), am Ende ein erzwungenes Einfügen des Skripts (Ziel am realen System verfehlt, 25 El.) |
| FV1 | 67 | 0 | 50 | 17 übernommen; zweimal erzwungenes Einfügen des Skripts |
| JFV1 | 0 | 7 | 6 | Feldregel nie ausgelöst (RMS Rand/Achse < 1,5 entlang der Diagonale) |
| FV855 | 6 | 2 | 8 | alle verworfen: kein Element verbessert das NA-0,855-System bei 11,6 % Feld |
| JFV855 | 54 | 1 | 36 | 19 übernommen, 24 Runden |
| F855 | 7 | 12 | – | Einfüge-Serie (F_B nach dem Einfügen 0,1–2,5 Wellen²), 26 El. ohne Fortschritt |

**Bewertung.**

1. **F14 ist behoben, wo die Feldregel greift:** Bei NA 0,5 führt sie die Feldstufe ab der Einzellinse bis 100 % (F1), das nachoptimierte reale System ist bei vollem Feld beugungsbegrenzt (≤ 0,059 Wellen, Strehl ≥ 0,88; Basis V1: 27 %). Der Weg baut wie der alte M6-T6-Lauf eine Folge dünner Menisken aus dem hochbrechenden Testglas (Feldlinsenlage zwischen den Gruppen bzw. vor dem Objekt) – mit 25 Elementen fast dreimal so viele wie V6-Z2. Das Ergebnis ist ein Nachweis, dass der Controller das Feld mit Bedarf aus den Feldfehlern öffnen kann, kein brauchbarer Entwurf.
2. **Bei hoher NA nur ein Teilerfolg:** Ab dem 5-Linser bei NA 0,855 verbessert kein einzelnes Element die Feldstufe (FV855: alle acht Versuche verworfen; F855 ohne Prüfung: 26 El., schlechter). Erst die gemeinsame Homotopie (Apertur und Feld zugleich in Richtung (0,95; 100 %)) mit Feldregel und Prüfung kommt auf 23 % Feld bei NA 0,869 (Z2 am Rand 0,087 Wellen, knapp über dem Ziel). Ziel „Feld über 27 % bei hoher NA“: **nicht erreicht**.
3. **Gemeinsame Homotopie allein** öffnet ab der Einzellinse mehr Feld als die Stufenfolge (37 % statt 27 %), aber bei kleinerer NA (0,36): Die Feldgrenze dieses kompakten Aufbaus wandert entlang der Diagonale nur mit (NA·φ ≈ 0,13 wie bei V1 mit 0,5·0,27), ohne Feldregel wächst das System nicht. Mit Feldregel und Prüfung (JFV1) bleibt es bei 2 Elementen, weil die Diagonale das Verhältnis Rand/Achse unter 1,5 hält und die Geisterbedarfe sechsmal verworfen werden.
4. **Einfügen nur mit Verbesserung** hält die Systeme schlank, wo sonst Serien entstehen (FV855: 5 statt 26 El.; JFV1: 2 El.), bremst aber die Feldstufe bei NA 0,5 (FV1 63 % statt F1 100 %): Viele Menisken der Feldregel verbessern F_B erst zusammen mit den späteren Spaltfreigaben des Skripts.
5. „Farbe früh“ mit der Apertur zuerst (Empfehlung 2 aus M6-T6b) und die Stufen nach p2 (Apertur, Farbe) sind aus Zeitgründen nicht gerechnet; die Läufe enden mit `--bis p2`.

## 4. Befunde

| Nr. | Stand nach M6-T8 |
|---|---|
| F14 | **Ursache geklärt** (Abschnitt 1); mit `--field-rule` bei NA 0,5 behoben (100 % Feld, beugungsbegrenzt, aber 25 El.), bei NA 0,855 nur bis 23 % (mit `--joint-aperture-field --insertion-improve`) |
| F15 | **nicht bearbeitet**: Die Läufe enden nach der Feldstufe. Die neue Prüfung „Einfügen nur mit Verbesserung“ wirkt in allen stetigen Phasen, also auch in der Aperturstufe nach den Farbstufen; gemessen ist das nicht |
| F16 | **neu adressiert** durch `--insertion-improve`: verhindert Serien ohne Gewinn (FV855, JFV1), kostet aber Fortschritt, wo Elemente erst gemeinsam wirken (FV1) |
| F3/F4 | für Feldfehler durch die Feldregel umgangen; der Normierungsradius der hinteren Platten (nur mit der NA skaliert, sechsmal größer als das Bündel) bleibt ein Mangel des Wörterbuchs – eine Anpassung an die paraxiale Bündelgröße am Ort der Platte wäre eine eigene Aufgabe (im Versuch mit r_g = 0,3 mm wurden die Platten aber auch nicht aktiv) |

**Empfehlungen.** (1) Feldregel in der Feldstufe bei mittlerer NA verwenden und danach ein Elementbudget durchsetzen bzw. Elemente zurückbauen (`remove_element`, M10-T4) – der 25-Linser zeigt, dass das Feld erreichbar ist, nicht wie schlank. (2) Für hohe NA ein Modul statt einzelner Menisken einfügen (Petzval-Gruppe, dicker Meniskus aus der Wissensbasis M10-T5): Die Diagnose zeigt, dass einzelne dünne Elemente den Astigmatismus des NA-0,855-Systems nicht lösen. (3) Normierungsradien der Geisterplatten aus der Bündelgröße am Ort der Platte (eigene Aufgabe, ändert die Merit). (4) Feldregel + Farbe früh + Apertur danach als nächster vollständiger Lauf.

## 5. Tests

`tests/test_homotopy.cpp`: `HomotopyControl.JointApertureFieldDiagonal`, `…Exponent`, `…InsertsOnPath`, `…OffOrSingleParameterUnchanged`, `…InvalidOptions`, `HomotopyControl.InsertionMustImprove` (synthetisches Modell `tests/homotopy_mock.hpp`, neu `bad_options`), `OpticalHomotopy.FieldRuleRecommendsMeniscusAtFieldPosition` (Lage gegen den Aberrationsbericht, Glas, Varianten, Merit nach dem Einfügen innerhalb `kInsertionMeritRelative`); `tests/test_cli_synth.cpp`: `CliSynth.JointApertureFieldAndFieldRule` (JSON-Optionen nur mit Option, Schrittart, Text, Fehler der Kommandozeile); `python/tests/test_tool_m6t6.py::test_synth_options_field_rule`.

## 6. Dateien

```
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --feldregel --out docs/ergebnisse/m6-t8/F1_feldregel
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --gemeinsam --out docs/ergebnisse/m6-t8/J1_gemeinsam
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --start docs/ergebnisse/m6-t6b/V2_apertur_zuerst/p1_final.json --na-feld 0.95 --gemeinsam --feldregel --einfuegen-verbessern 4 --out docs/ergebnisse/m6-t8/JFV855_gemeinsam_feldregel_verbessern
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --start docs/ergebnisse/m6-t6b/V2_apertur_zuerst/p1_final.json --ab-p2 --feldregel --einfuegen-verbessern 4 --out docs/ergebnisse/m6-t8/FV855_feldregel_verbessern
```

Die übrigen Läufe mit den Optionen der Tabelle. Je Lauf: `m6t6_lauf.json`, `m6t6_zeiten.json`, `lauf.log`, Textprotokolle `p*_r*.txt`, Stufenergebnisse, Z2 (`z2_s0.json` und Bewertungen `z2_*.json`, bei Teilfeld auch `z2_vollfeld_pupille.json`/`_unendlich.json`). Mit `--archiv` verkleinert, ohne JSON-Protokolle der synth-Läufe und ohne PSF-Dateien bei vollem Feld (2,4 MB). Glasdaten: nur `basis.json` und die frei erfundenen Testgläser – die Systeme sind nicht herstellbar.
