# M6-T11: Farbstufe der Synthese

Aufgabe: [Meilensteinplan](../plan/meilensteine.md#m6-t11--farbstufe-der-synthese-gezielt-verbessern), freigegeben vom Maintainer am 2026-11-22. Ausgangslage: [Bericht M6-T10](m6-t10-zyklisch.md) – alle Farbstufen scheiterten (Startpunkt nicht annehmbar, die Geisterplatten empfehlen Operator A, die Farbregel greift kaum, Rückbau 0/195). Alle Zahlen aus Läufen auf dem Rechner des Maintainers (RTX 3090, bis zu sieben Läufe gleichzeitig – Laufzeiten grob), FP64, Release mit CUDA 12.9, Referenzsphäre `image_infinite`, 16 Ringe × 8 Speichen, FD-Jacobi. Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json`; keine Linsendaten realer Objektive.

**Kurzfassung.**

- **Umgesetzt** (alles Option, ohne sie bitgleich; [optimierung.md, „Farbstufe“](../optimierung.md#farbstufe-m6-t11)): Operator B und Farbregel mit Partnerglas nach dem spektralen Rest und einem Zweig „Startradien des Glieds“ (`thin_member_shape` mit Randstrahlneigung); Farbbedarf aus den chromatischen Seidel-Termen (Kriterium d); Null- bzw. starkes Kittglied aus der Wissensbasis statt Meniskus, Gläser aus der Vorauswahl (`anomalous_pgf`, `merte_pair`); stetige Farbstufe ab λ_ref im Skript; Rückbau bei Q × Parameter der Stufe (`--rueckbau-punkt`).
- **An kleinen Systemen wirkt Operator B neu deutlich:** Einzellinse → Kittglied, der Zweig mit den Startradien gewinnt: RMS 0,0150 → **0,0006** Wellen (480/644 nm) bzw. 0,0158 → **0,0053** Wellen (436–656 nm), gleiche Elementzahl, eine Optimierung mehr (Abschnitt 2). Längsfarbe des dünnen Achromats schon beim Einfügen 0,778 → 0,018 mm (Test).
- **Diagnose der Farbstufe:** Die monochromatischen Endstände aus M6-T9/M6-T10 haben bei 436–656 nm 16–24 Wellen chromatischen RMS (σ_col), eine Fokusspanne g…C von 160–300 µm (Schärfentiefe ±0,3 µm) und ΔW020 = 70–113 Wellen; ihre Gläser sind nur CaF₂, SYN-30, SYN-36 (das Geisterplatten-Wörterbuch). F_B wächst über der Spektrumbreite w wie F₀ + c w² mit c = 130–280 Wellen², also schon bei w ≈ 0,005 über F_max. Mit allen Gläsern frei optimiert bleibt der 9-Linser bei ≈ 15 Wellen (ohne Glasfreiheit ebenso). Die „Farbe früh“-Systeme aus M6-T6b haben dagegen 1,5–1,9 µm Fokusspanne und σ_col 0,14–0,19 Wellen. **Farbe lässt sich aus diesen monochromatischen Ständen lokal nicht mehr herstellen** (F13 quantifiziert).
- **Messung ab den monochromatischen Ständen:** Die stetige Farbstufe beginnt annehmbar (F_B des Endstands), der Controller fügt jetzt nur noch Kittglieder ein (Operator B bzw. Modul, kein Meniskus mehr), kommt aber nur bis zur Spektrumbreite 0,001–0,15; nach Z2 am vollen Spektrum 0,46–18,7 Wellen am Rand – nicht besser als M6-T10 (0,5–1,8).
- **Farbe früh weiter (die wirksamen Läufe):** ab V3 (M6-T6b, 12 El., NA 0,535, 27 % Feld) Feld 27 → 38 %, dann **NA 0,535 → 0,6075** polychromatisch (16 El.): 436–656 nm auf der Achse ≤ 0,028, bei 70/100 % ≤ 0,111/0,130 Wellen, Strehl ≥ 0,64 – höchste NA eines polychromatischen Systems bisher, auf der Achse beugungsbegrenzt. Ab V7 (11 El., NA 0,5, volles Feld, ≤ 0,26 Wellen) ≤ 0,091/0,113/0,173 Wellen (19 El.). Polychromatisch ab der Einzellinse (P): 8 El., NA 0,363, 37 % Feld, ≤ 0,071/0,151/0,253 Wellen.
- **Rückbau bei 85 %:** in keiner Stufe ein Element entfernt (0 von 114 Versuchen): Kein polychromatisches Stufenergebnis liegt unter F_max, F_B ≤ 0,5 F_max ist auch bei 85 % nicht erreichbar.
- Ziel (NA 0,95, volles Feld, 436–656 nm beugungsbegrenzt) nicht erreicht.

## 1. Umsetzung

| Punkt der Aufgabe | Umsetzung | Option |
|---|---|---|
| 1. Operator B/Farbregel auf spektralen Rang und Startradien | `SplitInitOptions::preselect_wavelengths_nm` (Partner nach dem spektralen Rest über das Zielspektrum), `member_start` + `apply_member_start` (alle drei Krümmungen aus `thin_member_shape`, Biegung S_I = 0, Randstrahlneigung u/y vor der Linse aus dem paraxialen Strahl – neu `incoming_slope`, numerik.md §15), weiterer Zweig, der Controller wählt | `synth --split-spectrum`, `--split-member-start`; Skript `--farb-spektrum-rang`, `--farb-startradien` |
| 2. Farbbedarf erkennen, Modul statt Meniskus | Kriterium (d) der Farbregel: σ_col aus Fokusverlauf, Querfarbe, Sphärochromasie ≥ S × größter RMS; Kittglied-Modul (Null- bzw. starkes Kittglied, Wissensbasis M10-T5) im Luftspalt größter Randstrahlhöhe, wenn keine Linse teilbar ist (bzw. immer), vor der Geisterdiagnose; Modul und Teilen gegenseitig als Varianten für `--insertion-improve`; die bestehende Farbregel wird erweitert, nicht dupliziert | `--color-chromatic S`, `--color-module doublet\|strong`, `--color-module-always`, `--color-module-anomalous`, `--color-module-incidence W` |
| 3. Startpunkt der Farbstufe annehmbar | aus der Diagnose (Abschnitt 3): stetige Spektrum-Homotopie ab λ_ref (Start = F_B des monochromatischen Endstands), direkt zum vollen Spektrum; beim Neustart einer Runde wieder das Zielspektrum (vorher schrieb synth das Spektrum des Endpunkts zurück). Eine an F_B gekoppelte Annahmeschwelle ist **nicht** umgesetzt: Sie würde mehrere Wellen Farbe annehmen, statt sie zu korrigieren | Skript `--farb-stetig --farb-ziel voll` |
| 4. Rückbau bei 80–90 % | Rückbau im Zyklus bei Q × NA, Feldanteil bzw. Spektrumbreite, danach synth ohne Einfügen zurück zum erreichten Punkt (Farbe: über Z2) | Skript `--rueckbau-punkt Q` |

Behoben während der Messung: Das Modul konnte in einen Spalt gesetzt werden, dessen Teilstücke unter der Dickenschranke liegen (Abbruch des Laders, Läufe F7/P); solche Spalte entfallen jetzt. Die anfangs freigegebene Glasvariable des Moduls hatte bei Dicke 0 keinen Gradienten und lief im Lauf P bis n ≤ 0 (Abbruch); die Gläser des Moduls bleiben jetzt fest. Alle Läufe wurden danach mit demselben Programmstand neu gerechnet. Zuletzt brach F3 in `c1_apertur_r3` ab, weil im Zweig „Startradien des Glieds“ die freie Glasvariable von Operator B auf n ≤ 0 lief (Paraxialtrace wirft); der Controller verwirft jetzt bei mehreren Zweigen nur den werfenden Zweig (Vermerk „Zweig verworfen“, `HomotopyControl.ThrowingBranchIsDiscarded`), werfen alle, geht die Ausnahme weiter wie bisher. F3 lief ab dort mit diesem Stand weiter (die übrigen Ergebnisdateien sind unverändert); der zweite Durchlauf von F3 wurde aus Zeitgründen abgebrochen.

## 2. Operator B alt gegen neu (Standardwechsel?)

`synth --color-rule` an frei erfundenen Testsystemen (CPU bzw. GPU), höchstens 3 bzw. 6 Elemente:

| System | Optionen | Status | El. | RMS max [Wellen] | Optimierungen | gewählter Zweig |
|---|---|---|---|---|---|---|
| Einzellinse N-BK7, 480/644 nm | alt | Ziel | 2 | 0,0150 | 8 | – (Partner CaF₂) |
| | `--split-spectrum` | Ziel | 2 | 0,0115 | 8 | – (Partner SYN-23) |
| | `+ --split-member-start` | Ziel | 2 | **0,00060** | 9 | Startradien |
| Einzellinse N-BK7, 436–656 nm | alt | Ziel | 2 | 0,0158 | 8 | – (CaF₂) |
| | `--split-spectrum` | Ziel | 2 | 0,0158 | 8 | – (CaF₂) |
| | `+ --split-member-start` | Ziel | 2 | **0,0053** | 9 | Startradien |
| Einzellinse der Zielspezifikation (NA 0,3, Achse, 436–656 nm, Geisterplatten) | alt / spektral / neu | `max_elements` (6) | 6 | – | 41 / 41 / 43 | 3 × B, 2 × A |
| | neu + Modul + Kriterium d | `max_elements` (6) | 6 | – | 26 | 5 × B, kein A |

Bewertung: An dünnen Gliedern bei kleiner NA ist der neue Zweig klar besser und kostet eine Optimierung. In Systemen hoher NA (Zielspezifikation, alle Messläufe unten) verliert er fast immer gegen „wie bisher“ (Startmerit 10¹–10⁴⁵ Wellen², weil die dünne Biegung die dicke Linse im konvergenten Bündel stark verbiegt); der Controller nimmt dann Zweig 0, das Ergebnis ist dasselbe wie ohne Option, nur langsamer. **Empfehlung:** `--split-spectrum` und `--split-member-start` könnten Standard werden (nie schlechter, an Achromaten deutlich besser); ein Standardwechsel ändert aber alle Referenzläufe – **Frage an den Maintainer** im PR, hier nicht umgestellt. **Nachtrag M6-T11b:** vom Maintainer am 2026-11-22 freigegeben; beide sind seit M6-T11b Standard von `synth` (wirksam, wo vorausgewählt wird: Farbregel, `--split-preselect`, Kittglied-Modul), abschaltbar mit `--no-split-spectrum` und `--no-split-member-start`.

## 3. Diagnose: warum die Farbstufe scheitert

Farbkennzahlen der Startsysteme bei 436–656 nm ohne Optimierung (Aberrationsbericht ohne Platten, `mikroptim eval`):

| System | El. | NA | Gläser | ΔW020 | ΔW111 | Fokusspanne g…C | σ_col | RMS max |
|---|---|---|---|---|---|---|---|---|
| 9-Linser M6-T9 (volles Feld) | 9 | 0,50 | CaF₂, SYN-30, SYN-36 | 70,6 | 76,3 | 298 µm | 21,5 | 45,9 |
| 16-Linser M6-T10 E | 16 | 0,6725 | ebenso | 102,5 | 78,8 | 241 µm | 24,4 | 59,5 |
| 5-Linser M6-T9 (23 % Feld) | 5 | 0,869 | ebenso | 111,1 | 15,8 | 158 µm | 16,3 | 37,3 |
| 12-Linser M6-T10 A (65 % Feld) | 12 | 0,869 | ebenso | 113,4 | 38,8 | 160 µm | 18,7 | 54,5 |
| V7 M6-T6b (Farbe früh, volles Feld) | 11 | 0,50 | + SYN-29 | −0,12 | 0,72 | 1,9 µm | 0,19 | 0,26 |
| V3 M6-T6b (Farbe früh, 27 % Feld) | 12 | 0,535 | 6 Gläser | 0,34 | 0,53 | 1,5 µm | 0,14 | 0,11 |

(Wellen bei 546 nm; σ_col nach Kriterium d.) Die monochromatische Synthese baut stark brechende Elemente aus den drei Gläsern des Geisterplatten-Wörterbuchs, deren Farbbeiträge sich addieren; eine Einzellinse f = 3,29 mm hätte etwa 55 µm Längsfarbe F…C, diese Systeme haben das 3- bis 5-Fache und dazu große Querfarbe.

Verlauf der Bewertungsmerit am Anfang der stetigen Farbstufe (erste Schritte von `c0_farbe_voll_r0`, mit den zurückgekehrten Geisterplatten, die Farbe teilweise tragen):

| Start | F₀ [Wellen²] | F_B bei w = 1/256, 1/128, 1/64, 1/32 | c = (F_B − F₀)/w² |
|---|---|---|---|
| 9-Linser | 7,7·10⁻⁴ | 0,0030 / 0,0096 / 0,035 / 0,131 | 133–147 |
| 16-Linser | 3,8·10⁻³ | 0,0081 / 0,021 / 0,071 / 0,255 | 257–280 |
| 5-Linser | 9,2·10⁻⁵ | 0,0026 / 0,0097 / 0,036 / 0,115 | 117–163 |
| 12-Linser | 4,6·10⁻³ | 0,0086 / 0,020 / 0,066 / 0,237 | 238–259 |

Mit F_max = 5,1·10⁻³ endet die erste stetige Phase bei w ≈ 0,002–0,006 (Spektrum ±0,5 nm um 546 nm). Jedes eingefügte Kittglied verschiebt das nur wenig, weil es bei dieser Breite optimiert wird, wo Farbe kaum zählt. Gegenprobe ohne Synthese: der 9-Linser mit 436–656 nm, 3.000 LM-Zyklen, mit und ohne alle Gläser als Variablen (Glas-Hülle): RMS am Rand 15,5 bzw. 15,4 Wellen. Daraus folgen die Entscheidungen zu Punkt 3: Der Start der Farbstufe ist mit der stetigen Homotopie annehmbar; das Problem ist nicht der Start, sondern der Abstand des Systems von jedem farbkorrigierten Entwurf. Eine an F_B gekoppelte Schwelle ändert daran nichts.

Warum die Farbregel in M6-T10 nicht griff: An den Startpunkten der Farbstufe tragen die Geisterplatten (Gläser mit Dispersion) die Farbe teilweise; das RMS-Verhältnis (a) liegt dann bei ≈ 1, und |W040| ist bei hoher NA größer als |ΔW020| (b). Wo (b) doch zutraf, ließ sich keine Linse teilen (Linsen an der Dickenschranke 0,5 mm, Operator B braucht 1,0 mm), und die Regel gab an die Geisterdiagnose ab → Operator A. Mit Kriterium (d) und dem Modul fügt der Controller in allen Messläufen nur noch Kittglieder ein.

## 4. Messung

Optionen aller Läufe: `--farb-stetig --farb-ziel voll --farb-modul doublet --farb-spektrum-rang --farb-startradien --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --rueckbau-punkt 0.85` (P ohne Zyklus). Z2 = Stufenergebnis ohne Geisterplatten nachoptimiert (2.000 Zyklen) am vollen Spektrum der Stufe; bewertet nach dem Rückbau (`mikroptim eval --grid 256`), RMS in Wellen bei λ des Tupels, größter Wert über 436–656 nm (bei V3/V7 mit 405/707 nm schwach gewichtet in der Optimierung, aber nicht in der Tabelle) bei 0/70/100 % des erreichten Feldanteils; Strehl = kleinster FFT-Strehl über 436–656 nm.

| Lauf | Start | Stufe | NA | Feld | Spektrum | El. | erreichte Spektrumbreite | RMS 0/70/100 % | Strehl min | Zeit [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| M9 | 9-Linser M6-T9 | farbe | 0,50 | 1,00 | 436–656 | 9 → 15 | 0,011 | 0,615/1,266/1,691 | 0,00 | 6.427 |
| M9 | M9 c0 | apertur | 0,50 | 1,00 | 436–656 | 15 → 18 | – | 0,227/0,383/0,576 | 0,02 | 3.381 |
| M16 | 16-Linser M6-T10 E | farbe | 0,6725 | 1,00 | 436–656 | 16 → 19 | 0,002 | 0,458/0,899/1,585 | 0,00 | 7.839 |
| M5 | 5-Linser M6-T9 | farbe | 0,869 | 0,23 | 436–656 | 5 → 12 | 0,148 | 0,632/0,733/0,841 | 0,00 | 6.790 |
| M12 | 12-Linser M6-T10 A | farbe | 0,869 | 0,65 | 436–656 | 12 → 13 | 0,001 | 15,7/17,0/18,7 | 0,00 | 6.787 |
| F7 | V7 M6-T6b | apertur | 0,50 | 1,00 | 405–707 | 11 → 19 | – | 0,091/0,113/0,173 | 0,35 | 5.480 |
| F3 | V3 M6-T6b | feld | 0,535 | 0,27 → 0,38 | 405–707 | 12 → 15 | – | 0,032/0,120/0,152 | 0,41 | 6.129 |
| F3 | F3 c0 | apertur | **0,535 → 0,6075** | 0,38 | 405–707 | 15 → 16 | – | **0,028/0,111/0,130** | **0,64** | 3.421 |
| P | Einzellinse NA 0,3, 10 % Feld, 436–656 nm von Anfang (Hauptablauf, `--gemeinsam`) | p2 gemeinsam, p3 Apertur, p6 Z2, p7 eingerastet | 0,363 | 0,37 | 405–707 | 1 → 8 | – | 0,071/0,151/0,253 | 0,46 | 8.599 + Feinoptimierung |

Laufzeiten = Summe der Programmaufrufe der Stufe bei bis zu sieben gleichzeitigen Läufen.

- **Ab den monochromatischen Ständen** (M9, M16, M5, M12): Die stetige Farbstufe beginnt annehmbar, öffnet aber nur bis w = 0,001–0,011 (M5, der kleinste Start: 0,148). Eingefügt werden nur Kittglieder: Operator B an den wenigen dicken Linsen (oft dieselbe Linse mehrmals), bei M5 zusätzlich drei Null-Kittglieder; der Zweig „Startradien“ verliert bei diesen NA immer (Startmerit um Größenordnungen höher). Nach Z2 am vollen Spektrum bleibt der Farbfehler im Bereich von M6-T10 (M9 0,6–1,7 gegen B 0,65–1,76 Wellen). M12 (65 % Feld, NA 0,869) bricht in Z2 auf 16–19 Wellen ein. Die anschließende Aperturstufe von M9 startet über F_max und füllt das Budget (18 El.) ohne NA-Fortschritt, verbessert dabei aber die Farbe auf 0,23–0,58 Wellen.
- **F7** (ab V7): Startpunkt mit 405–707 nm nicht annehmbar (F_B 0,011 > F_max); die Startphase fügt zwei Kittglieder (B) und drei Null-Kittglieder ein, bis das Budget erreicht ist. Die Startphase verlangt keine Verbesserung von F_B (F_B stieg bis 0,046; Z2 holt es wieder herunter; seit M6-T11b prüft `--insertion-improve` auch die Startphase, mit Spielraum bis F_max). Ergebnis bei NA 0,5 und vollem Feld 0,17 statt 0,26 Wellen am Rand, mit 19 statt 11 Elementen, nicht beugungsbegrenzt.
- **F3** (ab V3): Feldstufe 27 → 38 % (15 El.), Aperturstufe NA 0,535 → 0,6075 in vier Runden (16 El.). Bei NA 0,6075 und 38 % Feld (Feldwinkel 0/1,16/1,66°) auf der Achse RMS 0,022–0,028 bei 436–656 nm (Strehl ≥ 0,97), bei 70 % 0,043–0,111, am Rand 0,046–0,130; die Grenze setzen 436 und 486 nm am Feldrand.
- **P** (polychromatisch ab der Einzellinse): Die gemeinsame Apertur-/Feldstufe bleibt nach fünf Runden bei NA 0,361 und 37 % Feld mit 8 Elementen stehen, die Aperturstufe kommt auf 0,363. Auf der Achse beugungsbegrenzt (≤ 0,071 Wellen über 436–656 nm), am Feldrand 0,25 Wellen bei 436 nm.

## 5. Vergleich

| Größe | M6-T6b (#105) | M6-T9 (#110) | M6-T10 (#111) | M6-T11 |
|---|---|---|---|---|
| polychromatisch, volles Feld | V7: 11 El., NA 0,50, ≤ 0,26 | – | B/C: 12–18 El., NA 0,50, 0,53–1,8 | F7: 19 El., NA 0,50, ≤ 0,17; ab monochromatisch M9 ≤ 0,58 (18 El.), M16 ≤ 1,6 (19 El., NA 0,6725) |
| polychromatisch, Teilfeld | V3: 12 El., NA 0,535, 27 %, ≤ 0,108 | – | – | **F3: 16 El., NA 0,6075, 38 %, ≤ 0,130** (Achse ≤ 0,028); P: 8 El., NA 0,363, 37 %, ≤ 0,253 |
| Farbe ab NA 0,869 | – | – | A: 14 El., 65 %, 0,48–0,81 | M5 (23 %): 0,63–0,84; M12 (65 %): 16–19 |
| Rückbau | – | 25 → 9 bzw. 5 | 0/195 | 0/114 (bei 85 % der Stufe) |

## 6. Befunde

| Nr. | Stand nach M6-T11 |
|---|---|
| F9 (Diagnose empfiehlt nie B) | **erledigt im Controller**: Farbregel mit Kriterium (d) und Modul fügt in den Farbstufen nur Kittglieder ein (B oder Null-Kittglied), auch wenn die Platten A empfehlen |
| F13 (Farbe nach dem monochromatischen Aufbau) | **quantifiziert, nicht lösbar mit lokalen Mitteln**: σ_col 16–24 Wellen, Fokusspanne 160–300 µm, c = 130–280 Wellen²; auch freie Gläser helfen nicht (≈ 15 Wellen). Farbe muss vor den monochromatischen Stufen ins System (F7, F3, P) |
| F15 (Apertur nach den Farbstufen) | **teilweise erledigt**: ab einem polychromatischen System nahe der Schwelle öffnet die Aperturstufe mit Kittglied-Einfügungen (F3: NA 0,535 → 0,6075 bei 38 % Feld); ab Systemen deutlich über F_max (F7, M9) bleibt sie stehen und füllt das Budget, verbessert dabei aber die Farbe |
| F16 (Einfüge-Serien) | bestätigt: in den Farbstufen werden viele Kittglieder wieder verworfen („Einfügen nur mit Verbesserung“), dieselbe Linse wird wiederholt geteilt (M12: Linse 7 viermal) |
| F17 (Rückbau braucht Abstand) | verallgemeinert: Rückbau bei 85 % entfernt nichts, solange das Stufenergebnis selbst über F_max liegt; der Vorschlag aus M6-T10 hilft nur bei beugungsbegrenzten Stufen |

## 7. Tests

`tests/test_color_stage.cpp` (sieben Tests, Orakel: Glasvorauswahl direkt, `thin_member_shape`, paraxiale Längsfarbe 0,778 → 0,018 mm, Schwelle von Kriterium d aus Bericht und Wellenfront, Merit-Invarianz des Null-Kittglieds 3,7·10⁻¹⁴ relativ, Einfallswinkel sin I = |y c|, Glasregel `anomalous_pgf`, ohne Modul-Option bitgleiche Empfehlung), `GlassPreselection.MemberShapeWithIncomingSlope` (Seidel-Summen gegen den Aberrationsbericht), `HomotopyControl.ThrowingBranchIsDiscarded` (Mock: werfender Zweig verworfen, werfen alle, Ausnahme wie bisher), `CliSynth.ColorStageOptions`, Python `test_tool_m6t6.py::test_synth_options_color_stage`, `::test_prune_point`, `::test_reopen_after_prune`.

## 8. Dateien

```
python tools/m6t6_zielspezifikation.py --gpu --zyklus farbe,apertur --start docs/ergebnisse/m6-t9/F1R_z2_rueckbau/z2_s0.json --elementbudget 16 --farb-budget 6 OPT --out docs/ergebnisse/m6-t11/M9_9linser
python tools/m6t6_zielspezifikation.py --gpu --zyklus farbe --start docs/ergebnisse/m6-t10/E_9linser_apertur/c0_apertur_z2r_s0.json --elementbudget 22 --farb-budget 6 OPT --out docs/ergebnisse/m6-t11/M16_16linser
python tools/m6t6_zielspezifikation.py --gpu --zyklus farbe --start docs/ergebnisse/m6-t9/JFV855R_z2_rueckbau/z2_s0.json --elementbudget 12 --farb-budget 6 OPT --out docs/ergebnisse/m6-t11/M5_5linser
python tools/m6t6_zielspezifikation.py --gpu --zyklus farbe --start docs/ergebnisse/m6-t10/A_5linser/c0_feld_z2r_s0.json --elementbudget 18 --farb-budget 6 OPT --out docs/ergebnisse/m6-t11/M12_12linser
python tools/m6t6_zielspezifikation.py --gpu --zyklus apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t6b/V7_start_z2alt_farbe_frueh/p7.json --elementbudget 18 OPT --out docs/ergebnisse/m6-t11/F7_v7
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t6b/V3_farbe_frueh/p7.json --elementbudget 18 OPT --out docs/ergebnisse/m6-t11/F3_v3
python tools/m6t6_zielspezifikation.py --gpu --farbe-von-anfang voll --gemeinsam --farb-modul doublet --farb-spektrum-rang --farb-startradien --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --elementbudget 16 --out docs/ergebnisse/m6-t11/P_farbe_von_anfang
```

mit `OPT` = `--farb-stetig --farb-ziel voll --farb-modul doublet --farb-spektrum-rang --farb-startradien --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --rueckbau-punkt 0.85`. Die Verzeichnisse sind mit `--archiv` verkleinert.
