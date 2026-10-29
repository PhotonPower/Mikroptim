# M6-T6: Lauf auf der Zielspezifikation (Plan-Apochromat 50x/0,95)

Bericht zur Forschungsaufgabe M6-T6 ([Meilensteinplan](../plan/meilensteine.md#m6-t6--lauf-auf-der-zielspezifikation)): stufenweiser Lauf nach [optimierung.md, Abschnitt 7](../optimierung.md#7-homotopie-controller) bis zur Spezifikation aus [spezifikation.md](../spezifikation.md), ausgehend von einer frei erfundenen Einzellinse. Alle Zahlen stammen aus Läufen auf dem Rechner des Maintainers (RTX 3090, 24 Kerne) mit dem Release-Build dieses Zweigs; die GPU wurde zeitgleich von anderen Agenten genutzt, Laufzeiten sind daher nur Orientierung.

**Kurzfassung.** Das Ziel (NA 0,95, volles Feld, 436–656 nm beugungsbegrenzt) wird **nicht** erreicht.

- **Hauptlauf** (Stufenfolge „Feld vor Apertur“, Abschnitt 3): NA 0,3 → 0,5 bei 10 % Feld mit 4 Elementen, dann Feld 10 → 100 % bei NA 0,5 mit 9 Elementen (Ziel der Stufe erreicht), dann Apertur nur bis **NA 0,57** (12 Elemente). Die Farbstufen (zwei Wellenlängen, dann 436–656 nm) scheitern: Der Controller fügt nur Menisken ein (nie ein Kittglied), auch erzwungene Kittglieder helfen kaum; nach der Feinoptimierung und dem Glas-Einrasten hat das Endsystem 22 Elemente und 0,43–5,6 Wellen RMS (Referenz Scheitel) bzw. 1,4–59 Wellen (Referenz Austrittspupille), Strehl ≈ 0.
- **Bester beugungsbegrenzter Zwischenstand** (Kandidat Z2, Ende der Feldstufe, ohne Geisterplatten nachoptimiert): 9 Elemente, NA 0,5, volles Feld (±4,345°), **monochromatisch 546 nm**, RMS 0,034/0,043/0,057 Wellen bei 0/70/100 % Feld, Strehl (FFT, skalar und vektoriell) 0,96/0,94/0,88, Petzvalradius 130 mm, Verzeichnung ≤ 0,07 %; Arbeitsabstand 0,25 mm (unter der angenommenen Grenze 0,3 mm).
- **Vergleichszweig** „Apertur zuerst“ (Standardfolge von optimierung.md §7): Merit-RMS bei NA 0,94 und 10 % Feld mit nur 4 Elementen – ein Artefakt der Referenzsphäre (Befund F12), das reale System hat > 20 Wellen RMS.
- **Laufzeit:** Hauptlauf 30 min, Vergleichszweig 6,5 min auf der RTX 3090 (geteilt); die Erkundung davor (rund 40 Läufe) dauerte etwa 10 Stunden.
- **Befunde:** 13 Grenzen des Controllers und der Merit (Abschnitt 6), davon ein Fehler im Code behoben (F1), die übrigen mit begründeten Gegenmaßnahmen im Skript umgangen oder als Empfehlung an den Maintainer formuliert. Ein Vergleich mit den Startsystemen des Maintainers ist offen (Abschnitt 8).

## 1. Annahmen

Die Spezifikation lässt Punkte offen (spezifikation.md, „Offene Punkte“). Für den Lauf gelten die folgenden Annahmen; keine davon ist eine Entscheidung des Maintainers.

| Punkt | Annahme im Lauf | Begründung |
|---|---|---|
| Arbeitsabstand | ≥ 0,25 mm während der Synthese (0,1 mm bis zur Hilfsebene der Geisterplatten plus ≥ 0,15 mm), ≥ 0,3 mm in der Feinoptimierung, sofern die Synthese ihn nicht schon unterschritten hat (Z2: 0,25 mm) | übliche Größenordnung trockener 50x/0,95-Objektive (Hinweis der Aufgabe) |
| größter Linsendurchmesser | 20 mm (`max_semi_diameter` 10 mm bei der freien Öffnung plus 0,2 mm) | Hinweis der Aufgabe; kein BD-Ringkanal berücksichtigt |
| Anlagefläche / Abgleichlänge | Blende bis Objekt ≤ 45 mm (`parfocal_length` mit Modus `max`, Referenz Blende, offset 0): die Anlagefläche liegt in der Blendenebene oder davor, die genaue Abgleichlänge stellt die Fassung ein | Eine Gleichung „Blende bis Objekt = 45 mm“ legt die Blende weit hinter die Optik; die Austrittspupille (in Gebrauchsrichtung) liegt dann nahe dem Objekt, und das Feld ist nicht korrigierbar (Abschnitt 6, Befund F7). Die Lage der Blende ist deshalb Variable. |
| Lage der Blende | variabel (Blendenspalt 0,5 … 40 mm), Start bei objektseitiger Telezentrie | ADR-0006 legt nur fest, dass die Blende die erste Fläche ist |
| Spektrale Gewichte | 435,83; 486,13; 546,07; 587,56; 656,27 nm gleich (1), 404,66 und 706,52 nm mit 0,1; λ_ref = 546,07 nm; zwei Wellenlängen der Achromasie-Stufe 480 und 643,85 nm | Hinweis der Aufgabe (flach, 405/700 schwach); 700 nm durch die r-Linie 706,52 nm ersetzt |
| Beugungsziel | RMS ≤ λ/14 pro Feld und Wellenlänge 436–656 nm (Wellen bei der Wellenlänge des Tupels); 405/700 nm nur bewertet | spezifikation.md; Zielprüfung des Controllers |
| Felder | 0; 3,045; 4,345° (0, 70, 100 % von FZ 25 mit f_T = 164,5 mm), gleich gewichtet | spezifikation.md |
| Petzvalradius | ≥ 130 mm (`petzval`, κ_max = 1/130 mm, w = 10⁴ mm²) | spezifikation.md |
| Verzeichnung | keine eigene Grenze; die Merit enthält die Kippung (Bildpunkte f·tan θ), berichtet wird der Wert des Hauptstrahls | spezifikation.md, offener Punkt 6 |
| DIC (spannungsarme Gläser) | nicht berücksichtigt; die Glaskataloge haben kein passendes Datenfeld | optimierung.md, Abschnitt 9 |
| Mechanik | Glasrand ≥ 0,3 mm und Luftrand ≥ 0,1 mm bei der freien Öffnung plus 0,2 mm, Luftspalte ≥ 0,1 mm, Glas-Mittendicke ≥ 0,4 mm, Gewichte 10⁴/mm² | wie die Startsysteme von M6-T3, Glasdicke ergänzt |
| Gläser | nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json` (Hülle und Einrasten); keine Herstellerkataloge | AGENTS.md §8; das Ergebnis ist mit realen Gläsern **nicht** herstellbar |
| Startsystem | frei erfundene Einzellinse (Abschnitt 2) | AGENTS.md §8: keine realen Objektivdaten |

## 2. Startsystem und Werkzeuge

**Startsystem** [`data/systems/zielspezifikation_start.json`](../../data/systems/zielspezifikation_start.json) (frei erfunden): Einzellinse aus einem hochbrechenden Glas (Buchdahl-Fit des Testglases SYN-36, n = 1,807, ν_d = 48,6), f = 3,29 mm, NA 0,3, 546,07 nm, Blendenspalt variabel (Start telezentrisch). Zwei Hilfsebenen tragen das Geisterplatten-Wörterbuch (je drei Gläser: hochbrechend wie SYN-36, Fluorit CAF2, Schwerflint wie SYN-30; K = 10, μ = 47,2 Wellen²/mm, Ordnungsfaktor 1 wie in M6-T3): Fläche 1 im kollimierten Bündel vor der Linse (blendennah), Fläche 4 im konvergenten Bündel vor dem Objekt (bildnah, Hauptstrahlhöhe relativ groß). Die Spalte mit Geisterplatten haben keine Dickenvariable, weil der Operator A genau sie teilt (Befund F2). Nebenbedingungen wie in Abschnitt 1, alle von Anfang an.

**Hinweis (ab M11-T1c):** `mikroptim synth` bricht seitdem standardmäßig bei Stagnation ab (`--stall 20,1e-3`); das Skript übergibt `--stall 0`, damit der dokumentierte Lauf nachrechenbar bleibt. Mit `--aktuell` rechnet es mit den aktuellen Standards, mit `--farbregel` zusätzlich mit der Farbregel (M11-T5c), z. B. `python tools/m6t6_zielspezifikation.py --gpu --aktuell --farbregel --out ergebnisse_lokal/m6t6`.

**Reproduzierbarer Lauf:** [`tools/m6t6_zielspezifikation.py`](../../tools/m6t6_zielspezifikation.py) (Python 3, nur Standardbibliothek) verkettet `mikroptim synth`, `optimize`, `insert`, `snap` und `eval` und ändert zwischen den Stufen die Systemdateien:

```
cmake --preset release-cuda && cmake --build --preset release-cuda
python tools/m6t6_zielspezifikation.py --gpu            # Lauf, Ausgabe in docs/ergebnisse/m6-t6/
python tools/m6t6_zielspezifikation.py --tabellen       # Tabellen dieses Berichts
```

Eine Stufe, deren Ergebnisdatei existiert, wird übersprungen (Fortsetzung nach Abbruch). Der Lauf schreibt alle Zwischenstände nach [`docs/ergebnisse/m6-t6/`](m6-t6/) (pro synth-Lauf Protokoll als Text und JSON `mikroptim.synth/1` und Ergebnisdatei, pro Nachoptimierung Ein- und Ausgabe, `m6t6_lauf.json` als Zusammenfassung); im Repository liegt die mit `--archiv` verkleinerte Fassung (Abschnitt 9).

**Bausteine des Skripts** (Gegenmaßnahmen zu den Befunden in Abschnitt 6):

- `synth_loop`: synth → (Luftspalte freigeben und nachoptimieren → synth)* bis zum Ziel der Stufe. Der Controller hält Luftspalte ohne Dickenvariable fest, und die Spalte hinter eingefügten Elementen haben keine (F2). Bleibt er stehen (Schrittweite unter Δ_min ohne aktive Geisterplatte), werden alle Spalte ohne Geisterplatten und alle Glasdicken Variablen (`add_gap_variables`), `mikroptim optimize` rechnet am erreichten Punkt nach (1000 Zyklen), synth startet dort neu. Ohne Fortschritt wird μ einmal halbiert (F3); weiter ohne Fortschritt folgt ein **erzwungenes Einfügen** (`forced_insertion`: Planplatte 0,4 mm aus dem hochbrechenden Glas mitten in jeden Luftspalt ≥ 0,7 mm, alle Kandidaten nachoptimiert, der beste gewinnt; in den Spektrumstufen `forced_split`: jede Glaslinse ≥ 1 mm mit `mikroptim insert --split --free-glass` geteilt).
- Stufenfolge „Feld vor Apertur“ (F4, F5): kleines Feld (10 %) und Petzval-Bedingung von Anfang an, Feld auf 100 % bei NA 0,5, danach die Apertur.
- Merit mit 16 Ringen statt 10 (F6), Rechnung auf der GPU (`--gpu`, F8).
- Endsystem: Geisterplatten und Hilfsebenen entfernt, Buchdahl-Fits der Testgläser durch die Katalognamen ersetzt, Feinoptimierung mit 405/700 nm schwach (`polish`), Glas-Einrasten (`snap`, k = 3), Bewertung mit `eval` (Raster 256 × 256, Referenz durch die Austrittspupille und durch den Scheitel, skalare und vektorielle PSF).

## 3. Ablauf und Protokoll des Laufs

Stufen des Hauptlaufs (`python tools/m6t6_zielspezifikation.py --gpu`; Merit mit 16 Ringen × 8 Speichen, LM höchstens 500 Zyklen pro synth-Schritt und 1000 pro Nachoptimierung, FD-Jacobi, FP64). Eine „Runde“ ist ein synth-Lauf mit anschließender Nachoptimierung der Spalte; „erzwungen“ zählt die Runden mit erzwungenem Einfügen. F_B und RMS (mit Geisterplatten, größter Wert über die Tupel, Wellen) beziehen sich auf den letzten angenommenen Schritt der Stufe.

| Stufe | Inhalt | Runden | Status der letzten Runde | Punkt (NA, Feldanteil) | Glaselemente | Einfügungen (Controller) | erzwungen | F_B [Wellen²] | RMS [Wellen] | Laufzeit [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| p1 | Apertur 0,3 → 0,5, 546 nm, 10 % Feld, Petzval aktiv | 2 | `goal_reached` | 0,50; 0,10 | 1 → 4 | 3 | 0 | 1,9·10⁻³ | 0,054 | 84 |
| p2 | Feld 10 → 100 % bei NA 0,5 | 6 | `goal_reached` | 0,50; 1,00 | 9 | 4 | 1 | 3,1·10⁻³ | 0,068 | 326 |
| p3 | Apertur → 0,95 bei vollem Feld | 16 | `step_too_small` | **0,57**; 1,00 | 12 | 0 | 3 | 4,6·10⁻³ | 0,084 | 295 |
| p4 | 480 und 643,85 nm | 4 | `max_elements` (Budget 1 pro Runde) | 0,57; 1,00 | 17 | 4 | 1 | 0,43 | 0,52 | 154 |
| p5 | 435,83 … 656,27 nm | 4 | `max_elements` | 0,57; 1,00 | 22 | 4 | 1 | 1,17 | 0,97 | 477 |
| p6 | Feinoptimierung ohne Geisterplatten, 405/700 nm mit Gewicht 0,1, Glas-Hülle in drei Stufen | – | 3 × 2000 Zyklen | 0,57; 1,00 | 22 | – | – | F = 11,7 | – | 36 |
| p7 | Glas-Einrasten (k = 3, 9 Kombinationen, 500 Zyklen) | – | `max_iterations` | 0,57; 1,00 | 22 | – | – | F = 3,63 | – | 26 |
| Z2 | Zwischenkandidat: Ende von p2 ohne Geisterplatten nachoptimiert (546 nm) | – | 2000 Zyklen | 0,50; 1,00 | 9 | – | – | F = 2,1·10⁻³ | 0,057 | 14 |

Summe der Programmaufrufe des Hauptlaufs: 1.412 s (davon synth 1.053 s, 75 %), Vergleichszweig 386 s. Der Lauf ist deterministisch: Ein früherer Lauf mit demselben Skript (vor einer Korrektur am erzwungenen Einfügen: `mikroptim insert` hatte die Standardauswahl der mechanischen Bedingungen in Listen der alten Flächen umgeschrieben, sodass neue Flächen nicht mehr eingeschränkt waren; das Skript stellt die Standardauswahl jetzt wieder her) und ein Erkundungslauf mit denselben Einstellungen trafen in p1 und p2 dieselben Entscheidungen mit denselben Zahlen im Protokoll.

**Protokoll (gekürzt).** Vollständig: `docs/ergebnisse/m6-t6/p*_r*_log.json` (Format `mikroptim.synth/1`) und die Textprotokolle `p*_r*.txt`.

- *p1:* Bedarf bei NA 0,3075 (Schrittweite unter Δ_min, F_B > F_max) → Operator A, Meniskus aus dem hochbrechenden Glas hinter der Blende (Platte 0); bei NA 0,37 zwei weitere Menisken (Schwerflint, Fluorit) an derselben Lage; danach Stillstand (keine aktive Platte), Spalte frei, synth erreicht NA 0,5 (`goal_reached`).
- *p2:* Feld 0,1 → 0,225 → 0,49, Stillstand; nach einer μ-Halbierung bis Feldanteil 0,881 mit vier Menisken an der bildnahen Lage (Platten 3, 4, 5) und einem blendennahen (Platte 0); dann Start abgelehnt ohne Bedarf → erzwungene Planplatte in Spalt 5 (bester von zwei Kandidaten, F = 5,8·10⁻³) → volles Feld erreicht.
- *p3:* Apertur in Schritten von 0,0025 bis 0,57 in 16 Runden, ohne Einfügen durch den Controller; drei erzwungene Planplatten (Spalt 5, 5, 11) bringen je wenige Tausendstel NA.
- *p4/p5:* Bei jedem Spektrumwechsel wird der Start abgelehnt, die Diagnose empfiehlt immer Operator A (Menisken an der bildnahen und an der blendennahen Lage), nie B; erzwungene Kittglieder helfen wenig (p4: Linse 14 senkt F der Nachoptimierung von 182 auf 102 Wellen², die übrigen drei Kandidaten gar nicht; p5: Linse 39 von 1,201 auf 1,193 Wellen²). Die Elementzahl ist durch das Budget von einem Einfügen pro Runde begrenzt; ein früherer Lauf mit zwei Einfügungen pro Runde stieß bei 59 Flächen und 6 Geisterplatten an die Grenze von 64 Tabellenflächen (`kMaxSurfaces`).

**Fokuslage** (Prüfung auf die Falle F12; `--tabellen`): In allen Stufen des Hauptlaufs liegt der paraxiale Fokus in der Bildebene (hintere Schnittweite 0,147 … 0,509 mm gegen den Abstand der letzten Fläche zum Objekt 0,150 … 0,498 mm), Petzvalradius −99 … −130 mm (p6: −99,5 mm, die Bedingung ist dort leicht verletzt; p7: −126,8 mm).

## 4. Kandidaten und Endsystem

**Stammbaum.** Der Hauptlauf verfolgt einen Zweig (einzweigiger Controller; die Strahlsuche brachte in der Erkundung bei diesen Stufen keinen Vorteil, F8): Start (1 Element) → p1 (4) → p2 (9) → **Z2** (9, nachoptimiert, monochromatisch) bzw. p3 (12) → p4 (17) → p5 (22) → p6/p7 (22, Endsystem). Jede Einfügung (Abschnitt 3) erzeugt zwei Sattelzweige, der bessere wird weitergeführt; die erzwungenen Einfügungen wählen den besten von 2–8 Kandidaten (Lagen bzw. Linsen). Der **Vergleichszweig** „Apertur zuerst“ (`--apertur-zweig`, Strahlsuche B = 2 mit zwei Varianten pro Einfügen, 12 Runden) baut in der ersten Runde einen Stammbaum aus 9 Kandidaten (4 expandiert, 3 verworfen: Duplikat bzw. Rang, 2 behalten mit 4 Elementen, `a1_r0.txt`); weil das Skript zwischen den Runden nur das Ergebnis weitergibt, beginnt jede weitere Runde mit einem Kandidaten. Er erreicht in der Merit NA 0,94 bei 10 % Feld mit 4 Elementen, das ist aber die Falle F12: Nach Entfernen der Hilfsebene und Nachoptimierung hat das System auf der Achse 21 Wellen RMS (bei 100 % Feld bis 120 Wellen).

| Kandidat | Elemente | NA | Feld | Spektrum | größter RMS [Wellen], Referenz Austrittspupille | Strehl (FFT) am Rand des Feldes | beugungsbegrenzt |
|---|---|---|---|---|---|---|---|
| Z2 (Ende p2, nachoptimiert) | 9 | 0,50 | 100 % | 546 nm | 0,057 | 0,88 | ja (monochromatisch) |
| Endsystem p7 | 22 | 0,57 | 100 % | 405–707 nm | 59 (Scheitel: 5,6) | 0,000 | nein |
| Vergleichszweig a2 | 4 | 0,94 | 10 % | 546 nm | 522 (Scheitel: 22) | 0,000 | nein (F12) |

**Systemdaten von Z2** (`docs/ergebnisse/m6-t6/z2_s0.json`; Syntheseergebnis aus frei erfundenen Testgläsern, nicht herstellbar; Tracerichtung vom Unendlichraum zum Objekt, Fläche 0 = Blende, Radius > 0: Krümmungsmittelpunkt in Tracerichtung; die letzte Dicke ist der Arbeitsabstand; „Strahlhöhe + 0,2 mm“ = größte Strahlhöhe über alle Felder plus Fassungszugabe): Brennweite 3,289 mm, Blende bis Objekt 21,55 mm, größter Halbdurchmesser 3,81 mm (Linsendurchmesser ≈ 7,6 mm), Petzvalradius −130,0 mm, Arbeitsabstand 0,250 mm.

| Fläche | Typ | Radius [mm] | Dicke [mm] | Material danach | Strahlhöhe + 0,2 mm |
|---|---|---|---|---|---|
| 0 | stop | ∞ | 0,5918 | AIR | – |
| 1 | sphere | 209,092 | 0,5000 | SYN-36 | 1,885 |
| 2 | sphere | -26,307 | 0,1000 | AIR | 1,902 |
| 3 | sphere | 45,681 | 0,5000 | CAF2 | 1,905 |
| 4 | sphere | 3,068 | 2,5014 | AIR | 1,904 |
| 5 | sphere | 326,802 | 0,4006 | SYN-36 | 2,594 |
| 6 | sphere | -373,379 | 2,5081 | AIR | 2,663 |
| 7 | sphere | 29,793 | 3,2302 | SYN-30 | 3,563 |
| 8 | sphere | -6,596 | 2,2788 | AIR | 3,808 |
| 9 | sphere | -3,834 | 1,8465 | SYN-36 | 3,129 |
| 10 | sphere | -4,922 | 0,3034 | AIR | 3,718 |
| 11 | sphere | 4,810 | 4,4810 | SYN-36 | 3,263 |
| 12 | sphere | 11,952 | 0,1000 | AIR | 1,771 |
| 13 | sphere | 10,391 | 0,5000 | SYN-30 | 1,693 |
| 14 | sphere | 10,087 | 0,1000 | AIR | 1,485 |
| 15 | sphere | 4,276 | 0,6236 | CAF2 | 1,350 |
| 16 | sphere | -8,745 | 0,2359 | AIR | 1,101 |
| 17 | sphere | -3,052 | 0,5000 | SYN-36 | 0,807 |
| 18 | sphere | -51,598 | 0,2500 | AIR | 0,620 |

**Systemdaten des Endsystems p7** (`docs/ergebnisse/m6-t6/p7.json`, 22 Elemente, nach dem Glas-Einrasten): Brennweite 3,285 mm, Blende bis Objekt 35,52 mm, größter Halbdurchmesser 3,58 mm, Petzvalradius −126,8 mm (Bedingung um 3 % verletzt), Arbeitsabstand 0,300 mm. Das eingerastete Kittglied (Flächen 40–42) besteht aus zweimal demselben Glas.

| Fläche | Typ | Radius [mm] | Dicke [mm] | Material danach | Strahlhöhe + 0,2 mm |
|---|---|---|---|---|---|
| 0 | stop | ∞ | 0,5000 | AIR | – |
| 1 | sphere | 134,302 | 0,5000 | CAF2 | 2,109 |
| 2 | sphere | -120,914 | 0,1000 | AIR | 2,131 |
| 3 | sphere | 161,888 | 0,5000 | SYN-30 | 2,139 |
| 4 | sphere | -29,825 | 0,1000 | AIR | 2,151 |
| 5 | sphere | 26,727 | 0,5322 | SYN-36 | 2,152 |
| 6 | sphere | -15,989 | 0,2334 | AIR | 2,141 |
| 7 | sphere | -14,486 | 0,5000 | CAF2 | 2,106 |
| 8 | sphere | 3,349 | 0,9324 | AIR | 2,027 |
| 9 | sphere | -21,409 | 0,4000 | SYN-36 | 2,088 |
| 10 | sphere | 54,221 | 0,2226 | AIR | 2,165 |
| 11 | sphere | -31,240 | 1,1662 | SYN-36 | 2,203 |
| 12 | sphere | 2358,445 | 0,3346 | AIR | 2,446 |
| 13 | sphere | 48,156 | 0,6630 | SYN-36 | 2,592 |
| 14 | sphere | -58,690 | 0,1000 | AIR | 2,687 |
| 15 | sphere | 82,543 | 0,4095 | SYN-36 | 2,741 |
| 16 | sphere | -62,064 | 0,1517 | AIR | 2,785 |
| 17 | sphere | -34,271 | 0,9692 | SYN-30 | 2,809 |
| 18 | sphere | 36,908 | 2,8775 | SYN-36 | 2,992 |
| 19 | sphere | -5,854 | 2,3015 | AIR | 3,281 |
| 20 | sphere | -3,771 | 2,9573 | SYN-36 | 2,784 |
| 21 | sphere | -5,481 | 1,7274 | AIR | 3,576 |
| 22 | sphere | 19,568 | 5,7321 | SYN-36 | 3,301 |
| 23 | sphere | -7,227 | 0,1331 | AIR | 2,697 |
| 24 | sphere | -6,825 | 0,5000 | CAF2 | 2,599 |
| 25 | sphere | -7,520 | 0,1163 | AIR | 2,448 |
| 26 | sphere | -7,247 | 0,5000 | CAF2 | 2,370 |
| 27 | sphere | -7,697 | 0,1516 | AIR | 2,226 |
| 28 | sphere | -6,700 | 0,5000 | SYN-36 | 2,150 |
| 29 | sphere | 7,350 | 0,4337 | AIR | 2,026 |
| 30 | sphere | -41,805 | 0,5000 | SYN-36 | 2,023 |
| 31 | sphere | -27,637 | 0,1000 | AIR | 2,027 |
| 32 | sphere | -46,092 | 0,5715 | SYN-36 | 2,023 |
| 33 | sphere | -6,634 | 0,1000 | AIR | 2,022 |
| 34 | sphere | -7,831 | 0,5000 | SYN-30 | 1,985 |
| 35 | sphere | 3,747 | 0,7427 | AIR | 1,952 |
| 36 | sphere | -180,523 | 2,9329 | SYN-30 | 2,081 |
| 37 | sphere | -4,275 | 0,1000 | AIR | 2,600 |
| 38 | sphere | 3,944 | 1,6328 | CAF2 | 2,465 |
| 39 | sphere | -6,735 | 0,1000 | AIR | 2,315 |
| 40 | sphere | 1,753 | 0,8653 | SYN-36 | 1,592 |
| 41 | sphere | 3,012 | 0,8297 | SYN-36 | 1,262 |
| 42 | sphere | 1,612 | 0,3000 | AIR | 0,667 |

## 5. Bewertung

**Z2** (NA 0,50, 546,07 nm; Raster 256 × 256, Referenz durch die Austrittspupille):

| Feld [°] | λ [nm] | RMS [Wellen bei λ] | RMS ohne Kippung | Strehl (Maréchal) | Strehl (FFT, skalar) | Strehl (vektoriell, unpolarisiert) | Verzeichnung [%] | versagt |
|---|---|---|---|---|---|---|---|---|
| 0,000 | 546,07 | 0,0340 | 0,0340 | 0,955 | 0,956 | 0,956 | – | 0 |
| 3,045 | 546,07 | 0,0430 | 0,0410 | 0,929 | 0,936 | 0,936 | 0,069 | 0 |
| 4,345 | 546,07 | 0,0571 | 0,0569 | 0,879 | 0,882 | 0,882 | 0,001 | 0 |

MTF bei f_c/2 (skalar, sagittal/tangential) und Halbwertsbreiten der vektoriellen PSF (unpolarisiert):

| Feld [°] | MTF_s | MTF_t | FWHM x [µm] | FWHM y [µm] | Anteil E_z |
|---|---|---|---|---|---|
| 0,000 | 0,376 | 0,376 | 0,569 | 0,569 | 0,064 |
| 3,045 | 0,375 | 0,370 | 0,572 | 0,583 | 0,066 |
| 4,345 | 0,349 | 0,352 | 0,578 | 0,592 | 0,068 |

**Endsystem p7** (NA 0,57, Referenz durch die Austrittspupille):

| Feld [°] | λ [nm] | RMS [Wellen bei λ] | RMS ohne Kippung | Strehl (Maréchal) | Strehl (FFT, skalar) | Strehl (vektoriell, unpolarisiert) | Verzeichnung [%] | versagt |
|---|---|---|---|---|---|---|---|---|
| 0,000 | 404,66 | 15,9083 | 15,9083 | 0,000 | 0,000 | 0,000 | – | 0 |
| 0,000 | 435,83 | 12,2305 | 12,2305 | 0,000 | 0,000 | 0,000 | – | 0 |
| 0,000 | 486,13 | 8,0929 | 8,0929 | 0,000 | 0,001 | 0,001 | – | 0 |
| 0,000 | 546,07 | 5,0180 | 5,0180 | 0,000 | 0,019 | 0,019 | – | 0 |
| 0,000 | 587,56 | 3,6195 | 3,6195 | 0,000 | 0,036 | 0,035 | – | 0 |
| 0,000 | 656,27 | 2,1110 | 2,1110 | 0,000 | 0,041 | 0,041 | – | 0 |
| 0,000 | 706,52 | 1,4418 | 1,4418 | 0,000 | 0,033 | 0,033 | – | 0 |
| 3,045 | 404,66 | 39,4773 | 22,9551 | 0,000 | 0,000 | 0,000 | -13,519 | 0 |
| 3,045 | 435,83 | 30,9689 | 19,0197 | 0,000 | 0,000 | 0,000 | -9,098 | 0 |
| 3,045 | 486,13 | 21,7080 | 14,4245 | 0,000 | 0,000 | 0,000 | -3,906 | 0 |
| 3,045 | 546,07 | 14,9084 | 10,8156 | 0,000 | 0,000 | 0,000 | 0,370 | 0 |
| 3,045 | 587,56 | 11,7974 | 9,0737 | 0,000 | 0,000 | 0,000 | 2,565 | 0 |
| 3,045 | 656,27 | 8,3554 | 7,0362 | 0,000 | 0,000 | 0,000 | 5,322 | 0 |
| 3,045 | 706,52 | 6,7070 | 5,9876 | 0,000 | 0,000 | 0,000 | 6,863 | 0 |
| 4,345 | 404,66 | 58,8439 | 30,1144 | 0,000 | 0,000 | 0,000 | -16,873 | 0 |
| 4,345 | 435,83 | 47,3624 | 26,0630 | 0,000 | 0,000 | 0,000 | -11,975 | 0 |
| 4,345 | 486,13 | 34,7949 | 21,1582 | 0,000 | 0,000 | 0,000 | -6,274 | 0 |
| 4,345 | 546,07 | 25,4329 | 17,1115 | 0,000 | 0,000 | 0,000 | -1,610 | 0 |
| 4,345 | 587,56 | 21,0637 | 15,0639 | 0,000 | 0,000 | 0,000 | 0,777 | 0 |
| 4,345 | 656,27 | 16,0867 | 12,5463 | 0,000 | 0,000 | 0,000 | 3,768 | 0 |
| 4,345 | 706,52 | 13,5907 | 11,1743 | 0,000 | 0,000 | 0,000 | 5,437 | 0 |

**Endsystem p7**, Referenz durch den Scheitel der letzten Fläche (wie die Merit):

| Feld [°] | λ [nm] | RMS [Wellen bei λ] | RMS ohne Kippung | Strehl (Maréchal) | Strehl (FFT, skalar) | Strehl (vektoriell, unpolarisiert) | Verzeichnung [%] | versagt |
|---|---|---|---|---|---|---|---|---|
| 0,000 | 404,66 | 2,2198 | 2,2198 | 0,000 | – | – | – | 0 |
| 0,000 | 435,83 | 1,5752 | 1,5752 | 0,000 | – | – | – | 0 |
| 0,000 | 486,13 | 1,3019 | 1,3019 | 0,000 | – | – | – | 0 |
| 0,000 | 546,07 | 0,9641 | 0,9641 | 0,000 | – | – | – | 0 |
| 0,000 | 587,56 | 0,7157 | 0,7157 | 0,000 | – | – | – | 0 |
| 0,000 | 656,27 | 0,4371 | 0,4371 | 0,001 | – | – | – | 0 |
| 0,000 | 706,52 | 0,4253 | 0,4253 | 0,001 | – | – | – | 0 |
| 3,045 | 404,66 | 4,6495 | 4,5152 | 0,000 | – | – | -13,519 | 0 |
| 3,045 | 435,83 | 3,2934 | 2,9958 | 0,000 | – | – | -9,098 | 0 |
| 3,045 | 486,13 | 2,1054 | 1,8840 | 0,000 | – | – | -3,906 | 0 |
| 3,045 | 546,07 | 1,4018 | 1,3961 | 0,000 | – | – | 0,370 | 0 |
| 3,045 | 587,56 | 1,3053 | 1,2456 | 0,000 | – | – | 2,565 | 0 |
| 3,045 | 656,27 | 1,5577 | 1,1325 | 0,000 | – | – | 5,322 | 0 |
| 3,045 | 706,52 | 1,8105 | 1,1043 | 0,000 | – | – | 6,863 | 0 |
| 4,345 | 404,66 | 5,5533 | 5,4431 | 0,000 | – | – | -16,873 | 0 |
| 4,345 | 435,83 | 3,9352 | 3,6464 | 0,000 | – | – | -11,975 | 0 |
| 4,345 | 486,13 | 2,4988 | 2,2745 | 0,000 | – | – | -6,274 | 0 |
| 4,345 | 546,07 | 1,8208 | 1,8182 | 0,000 | – | – | -1,610 | 0 |
| 4,345 | 587,56 | 1,8418 | 1,7700 | 0,000 | – | – | 0,777 | 0 |
| 4,345 | 656,27 | 2,2210 | 1,7895 | 0,000 | – | – | 3,768 | 0 |
| 4,345 | 706,52 | 2,5167 | 1,8067 | 0,000 | – | – | 5,437 | 0 |

**Lesart.** RMS in Wellen bei der Wellenlänge des Tupels (das Beugungsziel λ/14 = 0,0714 gilt so), „ohne Kippung“ nach Abzug der Kippung (Verzeichnung und Brennweitenfehler stecken in der Kippung, die Merit behält sie); Strehl nach Maréchal aus dem RMS, über die FFT im Bezugspunkt der Kippung (`mikroptim eval --psf`) und vektoriell nach Richards-Wolf mit unpolarisiertem Eingangsfeld (`--vector --polarization unpolarized`, normiert auf die perfekte vektorielle PSF derselben NA). Für Z2 stimmen die beiden Referenzsphären auf drei Stellen überein, für das Endsystem nicht: Seine paraxiale Austrittspupille liegt weit weg von der realen, die Werte mit Referenz durch den Scheitel (wie in der Merit) sind dort aussagekräftiger (F11). Mechanik von Z2: alle Ränder auf ihrer Grenze bis auf die Restverletzung der Strafe (≤ 4·10⁻⁵ mm), alle Halbdurchmesser ≤ 3,81 mm, Luftspalte ≥ 0,1 mm. Bei NA 0,5 ist die Halbwertsbreite 0,57–0,59 µm, der Anteil von |E_z|² 6–7 %.

**Verzeichnung:** Z2 ≤ 0,07 %, Endsystem bis 17 % bei 405 nm (Farbquerfehler).

## 6. Befunde: Grenzen des Controllers und Gegenmaßnahmen

Vor dem endgültigen Lauf wurden rund 40 Varianten erkundet (Startsysteme, Wörterbuch, μ, Ordnungen, Quadratur, Stufenfolge, Strahlsuche, CPU/GPU); die Protokolle liegen nicht im Repository, die wichtigsten Zahlen stehen hier. „Monochromatisch“ heißt 546,07 nm.

| Nr. | Befund | Beobachtung | Gegenmaßnahme im Lauf | Empfehlung |
|---|---|---|---|---|
| F1 | **Fehler:** Sattelstarts außerhalb der Krümmungsschranken | `mikroptim synth` brach mit `FormatError` ab („Startwert −1,20199 … unter der unteren Schranke −1,2“), als c₀ von einer Fläche auf ihrer Schranke kam und c₀ − δ die Schranke des neuen Eintrags verletzte | im Code behoben (c₀ und Starts auf die Schranken begrenzt, Test) | – |
| F2 | **Feste Luftspalte** | Neue Spalte hinter einem Meniskus haben keine Dickenvariable, die Elemente liegen danach mit 0,1 mm Luft aneinander. Mit 8 Elementen blieb der Lauf bei NA 0,9075 stehen; nur die Spalte freigeben (ohne neues Element) senkte den RMS von 0,062 auf 0,014 Wellen und erlaubte NA 0,95. Eine Dickenvariable im geteilten Spalt hilft nicht: mit Untergrenze > 0 ist das zweite Einfügen an derselben Lage ein `FormatError` (Teilstück 0), mit Untergrenze 0 kollidieren die Flächen in der inneren Homotopie (F bis 10¹⁷), und die Penalty-Fortsetzung danach übernimmt den Zustand ohne Prüfung | `synth_loop` (Spalte zwischen den synth-Läufen frei) | Controller: nach `finish_insertion` die Spalte hinter dem neuen Element als eigene Variable (Untergrenze `min_air_gap`); Zustand nach Einfügen und Penalty-Fortsetzung bewerten (F_B) statt ihn ungeprüft zu übernehmen |
| F3 | **Kein Bedarf trotz F_B > F_max** | Häufigster Abbruch: „keine Geisterplatte aktiv“, obwohl F_B die Schwelle überschreitet. Der Fehler verteilt sich auf viele Ordnungen bzw. auf Feldfehler (F4), keine Ordnung überschreitet ihre Schwelle μ w_k. μ/2 half einmal (z. B. NA 0,8325 → 0,95 mit 5 Elementen), μ/4 und kleiner lösten Einfüge-Serien aus (P1b2, P1d: 4 → 14 Elemente ohne Fortschritt), weil die Platten dann billig genug sind, die Arbeit selbst zu tun | μ höchstens einmal halbieren, danach erzwungenes Einfügen | Bedarfskriterium aus F_B selbst ableiten (z. B. Anteil des Restfehlers, den die Platten bei kleinem μ beseitigen könnten), statt nur aus der Aktivität bei festem μ |
| F4 | **Feldfehler sind für die Geisterplatten unsichtbar** | Eine monochromatische Achsoptimierung erreichte NA 0,95 mit 5 Elementen (RMS 0,040 Wellen), aber schon bei 0,07° Feldwinkel betrug der RMS 0,28 Wellen (Koma: Sinusbedingung verletzt); die rotationssymmetrischen Platten zeigten keinen Bedarf. Mit 10 % Feld in der Merit von Anfang an blieb das Ergebnis nahezu aplanatisch (NA 0,95, 6 Elemente, RMS 0,043/0,053/0,071 bei 0/7/10 % Feld) | kleines Feld (10 %) ab der ersten Stufe | Plattenlagen mit großem Verhältnis Hauptstrahl- zu Randstrahlhöhe gezielt anbieten; Feld früh öffnen |
| F5 | **Petzval und Feld** | Petzval-Bedingung erst in der Feldstufe: Die Penalty-Fortsetzung erzwingt R_P ≥ 130 mm über einen dicken Meniskus vor dem Objekt (das gelingt dem LM ohne neues Element), aber der Feldfehler steigt so stark, dass schon der erste Feldschritt scheitert; ohne aktive Platte endet der Lauf. Mit Petzval von Anfang an und Apertur zuerst: NA 0,90–0,94 bei 10 % Feld mit nur 4 Elementen, danach kommt das Feld nicht über 10 % hinaus (auch nicht mit drei erzwungenen Einfügungen). Mit „Feld vor Apertur“ (Feld 10 → 100 % bei NA 0,5) erreichte der Controller volles Feld (9 Elemente), die Apertur kam danach aber nur bis NA 0,55 | Petzval ab Start, Feld bei NA 0,5 | Homotopie in Apertur und Feld gemeinsam (Diagonale) statt nacheinander; Reserve lassen: Der Controller führt jede Stufe bis F_B ≈ F_max und lässt für die nächste Stufe keine Reserve |
| F6 | **Pupillenrand bei NA 0,95** | Mit 10 Ringen (äußerster Ring bei ρ = 0,9935) fällt die Wellenfront am Rand steil ab, die Merit sieht das kaum: Achse 0,040 Wellen in der Merit, 0,083 auf dem 256er-Raster mit Referenz durch die Austrittspupille (OPD −0,30 Wellen bei ρ = 0,988). Mit 16 Ringen 0,054 | 16 Ringe (Maximum der CLI) | mehr Ringe oder ein Randring bei ρ = 1 für NA > 0,9 |
| F7 | **Blende weit hinter der Optik** | Mit „Blende bis Objekt = 45 mm“ lag die Austrittspupille (objektseitig) 0,4 mm hinter der letzten Fläche, der Feldfehler bei 10 % Feld war 2× so groß wie auf der Achse, die Apertur blieb bei 0,685 stehen | Blendenlage variabel, Abgleichlänge nur als Obergrenze | Anlagefläche festlegen (Maintainer) |
| F8 | **Pfadabhängigkeit, GPU** | Läufe auf CPU und GPU (Rundung 10⁻¹³) entwickeln sich nach wenigen Einfügungen völlig verschieden (z. B. NA 0,43 mit 5 Elementen gegen NA 0,95 mit 6 Elementen aus demselben Start); jeder Lauf ist für sich reproduzierbar. Die GPU war für Systeme mit drei Feldern 6,6-mal so schnell (1,2 s gegen 7,8 s für 200 Zyklen) | GPU, Strahlsuche nur in der Erkundung | Multistart über Konfigurationen statt eines Laufs; Strahlsuche (`--beam`) mit gestörten Starts |
| F9 | **Operator B wird nicht empfohlen** | Beim Wechsel auf zwei Wellenlängen (480/644 nm) bei NA 0,94 empfahl die Diagnose wiederholt Operator A (13 Elemente, RMS ≈ 1 Welle): Die Heuristik χ > 2 max χ_g erkennt den Farbanteil nicht, wenn λ_ref nicht unter den Merit-Wellenlängen liegt und sich die monochromatische Balance mit verschiebt | erzwungenes Kittglied (`forced_split`) in den Spektrumstufen | Diagnose der Farbe gegen das Spektrum der vorigen Stufe (Differenz der Korrekturen) statt gegen λ_ref |
| F10 | **LM-Konvergenz** | Fast alle LM-Läufe enden mit `max_iterations` (500 bzw. 1000 Zyklen); Ursache wie in M6-T3 die Randdicken mit Höhe aus den Strahlen (Maximum über Strahlen und zwei Flächen) | – | glattere Randdicken-Formulierung (eigene Aufgabe, siehe PR #44) |
| F11 | **Referenzsphäre** | Die Merit misst auf der Sphäre durch den Scheitel der letzten Fläche (hier der Hilfsebene 0,15 mm vor dem Objekt), die Bewertung durch die Austrittspupille; bei NA 0,95 weichen beide um Faktor 1,3–2 ab, und für Systeme mit paraxial stark verschobener Pupille verfehlen bis zu 70 % der Rasterstrahlen die Sphäre durch die Austrittspupille (`eval` meldet sie als versagt) | Bewertung mit beiden Referenzen | Merit und Bewertung mit derselben Referenz (z. B. Sphäre durch die reale Austrittspupille des Hauptstrahls) |

**F12 – Falle der Referenzsphäre durch eine Hilfsebene vor dem Objekt.** Die Merit misst die OPD auf der Sphäre um den idealen Bildpunkt durch den Scheitel der letzten Fläche (numerik.md §7). Ist die letzte Fläche eine Hilfsebene im Abstand w vor dem Objekt (hier für die bildnahen Geisterplatten), hat diese Sphäre den kleinen Radius w. Der LM kann dann den Fokus in den Scheitel der Hilfsebene legen: Alle Strahlen treffen die Sphäre praktisch im selben Punkt, ihre OPD wird klein, obwohl das Bild um w defokussiert ist. Im Vergleichszweig „Apertur zuerst“ (Abschnitt 4) geschah genau das: paraxiale Schnittweite 0,003 mm ab der Hilfsebene, die Bildebene aber w = 2,70 mm dahinter (Defokus −2,69 mm; w ist dann nahezu wirkungslos), Merit-RMS 0,076 Wellen auf der Achse. Dasselbe System ohne Hilfsebene (Sphäre durch die Frontlinse, Bildebene unverändert) hat 151,7 Wellen RMS (fast reiner Defokus), und eine Nachoptimierung mit freiem Arbeitsabstand kommt nur auf 2 Wellen. Im Hauptlauf liegt der Fokus in der Bildebene (Schnittweite gleich w, Tabelle in Abschnitt 3); `--tabellen` druckt die Prüfung für jede Stufe. Empfehlung: Hilfsebenen nicht als letzte Fläche zulassen (Geisterplatten im Arbeitsabstand erlauben) oder die Referenzsphäre unabhängig von der letzten Fläche festlegen (F11); bis dahin die Schnittweite jeder Stufe gegen w prüfen.

**F13 – Farbe nach dem monochromatischen Aufbau.** Das Ende der Aperturstufe (12 Elemente, fast alle aus dem hochbrechenden Glas und dem Schwerflint, NA 0,57, volles Feld, 0,08 Wellen monochromatisch) hat bei 480 und 644 nm 2,4–20 Wellen RMS. Gibt man **alle** Gläser zugleich frei (Glasvariablen mit Hülle, 2 × 1000 Zyklen, Gewicht 10⁴ und 10⁵), bleiben 1,4–4,0 Wellen; mit kleinem Hüllengewicht am Anfang (10², wie die Penalty-Fortsetzung der Hülle in M6-T1) brach `mikroptim optimize` ab, weil ein Versuchsschritt eine Brechzahl ≤ 0 erzeugte („SystemBatch: index[23] muss > 0 sein“) – ein solcher Schritt sollte verworfen werden, statt den Lauf zu beenden. Die Stufenfolge „monochromatisch zuerst, Farbe zuletzt“ legt die Struktur des Systems (viele dünne hochbrechende Menisken) fest, bevor die Farbe ein Wort mitredet; für ein Apochromat mit NA 0,95 braucht es Kittgruppen und Fluorit früh. Empfehlung: zwei Wellenlängen schon in der ersten Stufe (geringes Gewicht) und ein Wörterbuch, dessen Gläser die Achromasie zulassen; ungültige Brechzahlen im LM als verworfenen Schritt behandeln.

Ergänzend: Die Ordnung der Geisterplatten bestimmt, wie weit die Apertur kommt (K = 6: NA 0,8575 mit 3 Elementen, K = 10: NA 0,9075 mit 8 Elementen, beide monochromatisch auf der Achse ohne Spaltvariablen). Weite Schranken der Luftspalte (40 mm) führten mit μ/2 in ein pathologisches System (Spalte auf der Schranke, Platten mit 10⁻⁶ mm Koeffizienten, realer RMS 30 Wellen gegen 0,1 Wellen mit Platten): Schranken der Spalte im Lauf ≤ 10 mm.

## 7. Was erreicht wurde und was nicht

**Erreicht:**

- Ein vollständig skripteter, deterministischer und fortsetzbarer Lauf über alle Stufen von optimierung.md §7 (Apertur, Feld mit Petzval, zwei Wellenlängen, volles Spektrum, 405/700 nm schwach, Feinoptimierung, Glas-Einrasten, Bewertung mit skalarer und vektorieller PSF) mit vollständigen Protokollen.
- Automatisch aus einer Einzellinse: ein **monochromatisch beugungsbegrenztes Planobjektiv mit NA 0,5 über das volle Feld der Spezifikation** (9 Elemente, Petzvalradius 130 mm, Strehl ≥ 0,88, Verzeichnung ≤ 0,07 %), und monochromatisch auf der Achse (ohne Feld) NA 0,95 mit 5–8 Elementen in der Erkundung.
- Ein behobener Fehler im Controller (F1) und zwölf weitere, mit Zahlen belegte Befunde zu den Grenzen von Controller, Diagnose und Merit (F2–F13), mit Gegenmaßnahmen im Skript und Empfehlungen.

**Nicht erreicht:**

- NA 0,95 bei vollem Feld: Die Apertur kommt nach dem Feld nur bis 0,57; mit der Standardfolge (Apertur zuerst) kommt das Feld nicht über 10 % hinaus.
- Farbkorrektur: Kein Achromat, kein Apochromat; die Diagnose empfiehlt nie Operator B, erzwungene Kittglieder und freie Gläser reichen nicht (F9, F13). Das Endsystem ist mit 22 Elementen weit von der Beugungsgrenze entfernt.
- Die Annahme Arbeitsabstand ≥ 0,3 mm hält Z2 nicht ein (0,25 mm, Grenze der Synthese).
- Herstellbarkeit: nur frei erfundene Testgläser; keine Tolerierung.

## 8. Offene Punkte und Empfehlungen für den Maintainer

1. **Vergleich mit den Startsystemen des Maintainers:** offen. Die Startsysteme (Zeiss EC Epiplan-Apochromat 50x/0,95, Patente DE 103 16 415 und US 6,504,653) liegen nicht im Repository (AGENTS.md §8). Vorschlag: dieselbe Bewertung (`tools/m6t6_zielspezifikation.py`, Stufe 8, bzw. `mikroptim eval --grid 256 --psf --vector`) auf die Patentdaten anwenden und die Synthese von dort aus starten (Feinoptimierung, Glas-Einrasten), um Controller und Merit an einem bekannten guten System zu prüfen.
2. **Offene Punkte der Spezifikation** (Abschnitt 1): Arbeitsabstand, Durchmesser/BD, Anlagefläche und Lage der Pupille, spektrale Gewichte, zulässige Verzeichnung, DIC.
3. **Controller (Codeänderungen, je eigene Aufgabe):** Spaltvariablen für neue Elemente und Prüfung des Zustands nach dem Einfügen (F2), Bedarfskriterium aus F_B (F3), Plattenlagen für Feldfehler und gemeinsame Homotopie in Apertur und Feld (F4, F5), Farbdiagnose gegen die vorige Spektrumstufe (F9), Randdicken-Glättung (F10), Referenzsphäre unabhängig von der letzten Fläche und Hilfsebenen vor dem Objekt vermeiden (F11, F12), ungültige Brechzahlen im LM als verworfenen Schritt behandeln und Farbe früh einbeziehen (F13), mehr als 16 Ringe und mehr als 64 Tabellenflächen für große Systeme.
4. **Glaskataloge:** Mit den frei erfundenen Testgläsern ist das Ergebnis nicht herstellbar; für einen echten Entwurf werden lizenzierte Herstellerkataloge in `data/glass/private/` gebraucht.
5. **Nächster Versuch** (Vorschlag): Start aus Z2 (NA 0,5, volles Feld) mit zwei Wellenlängen ab sofort, Wörterbuch mit Kronglas/Fluorit/Flint in Kittgruppen-Lage, gemeinsame Homotopie in Apertur und Feld, Strahlsuche mit gestörten Starts; bzw. Start aus einem Patentsystem (Punkt 1).

## 9. Dateien

| Datei | Inhalt |
|---|---|
| [`data/systems/zielspezifikation_start.json`](../../data/systems/zielspezifikation_start.json) | Startsystem (frei erfunden) |
| [`tools/m6t6_zielspezifikation.py`](../../tools/m6t6_zielspezifikation.py) | Lauf (`--gpu`), Vergleichszweig (`--apertur-zweig`), Tabellen (`--tabellen`), Verkleinern des Ausgabeverzeichnisses (`--archiv`) |
| `docs/ergebnisse/m6-t6/m6t6_lauf.json`, `m6t6_apertur_zweig.json`, `m6t6_zeiten.json` | Zusammenfassung der Stufen (Runden, Status, Punkte, Elemente, erzwungene Einfügungen), Laufzeiten aller Programmaufrufe |
| `docs/ergebnisse/m6-t6/p*_r*_log.json`, `p*_r*.txt`, `a1_r*_log.json`, `a1_r*.txt` | Protokolle aller synth-Läufe (`mikroptim.synth/1`, im Vergleichszweig mit Stammbaum) |
| `p0_start.json`, `p1_final.json`, `p2_final.json`, `p3_r15_restart.json`, `p4_r3_restart.json`, `p5_r3_restart.json` | Ergebnis jeder Stufe (mit Geisterplatten, Gewichte auf der Basis, Feldwinkel 100 %) |
| `p6_s2.json`, `p7.json` (+ `.txt`) | Feinoptimierung und eingerastetes **Endsystem** (Syntheseergebnis, nicht herstellbar) |
| `z2_s0.json` (+ `.txt`) | **Zwischenkandidat Z2** (Syntheseergebnis, nicht herstellbar) |
| `p8_*.json`, `z2_*.json`, `a3_*.json` | Bewertungen `mikroptim eval --json` (Raster 256, Referenz Austrittspupille/Scheitel, skalare PSF, vektorielle PSF) |

Das Verzeichnis ist mit `--archiv` verkleinert (Nachoptimierungen und Kandidaten der erzwungenen Einfügungen entfernt, Pfade relativ); ein erneuter Lauf erzeugt alle Zwischenstände neu.
