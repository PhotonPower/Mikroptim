# Systeme

| Datei | Zweck |
|---|---|
| `einzellinse_bk7.json` | einfache Testlinse (Paraxialdaten, sphärische Aberration, Farbe) |
| `einzellinse_bk7_variablen.json` | Einzellinse mit Abschnitt `variables` (Beispiel für das Schema, M3-T1) |
| `einzellinse_bk7_nebenbedingungen.json` | Einzellinse mit `target` und Abschnitt `constraints` (Beispiel für das Schema, M3-T3; Lage der Anlagefläche angenommen) |
| `einzellinse_bk7_biegung.json` | Einzellinse mit Krümmungen und Arbeitsabstand als Variablen und schwach gewichteter Brennweite; Beispiel für `mikroptim optimize` (Biegung minimaler sphärischer Aberration, M3-T5, [docs/cli.md](../../docs/cli.md#mikroptim-optimize)) |
| `planplatte_bk7.json` | exakt bekannte optische Weglänge |
| `testsystem_mehrlinser.json` | frei erfundenes Mehrlinsensystem (kein reales Objektiv): vier Gläser, drei Felder, vier Wellenlängen; CPU/GPU-Abgleich der Residuen (M2-T2) |
| `testsystem_mehrlinser_dezentriert.json` | frei erfundenes Testsystem: `testsystem_mehrlinser.json` mit Dezentrierungen (bis 0,03 mm) und Verkippungen (bis 0,1°) an fünf Flächen (`decenter`, `tilt_deg`); Abnahme Dezentrierung und Verkippung (M8-T3, Spiegelsymmetrie, CPU/GPU) und Beispiel für `mikroptim trace`/`eval` ([docs/cli.md](../../docs/cli.md#dezentrierte-und-verkippte-systeme-ab-m8-t3)) |
| `weiches_versagen_start.json` | frei erfundenes Startsystem (kein reales Objektiv): Bikonvexlinse, deren Vorderfläche kleiner ist als die Blende; 30 % der Strahlen versagen (Fläche verfehlt, Totalreflexion). Abnahme des weichen Versagens (M4-T2): LM optimiert es ohne Eingriff versagensfrei |
| `einzellinse_geisterplatte.json` | frei erfundenes Testsystem: Einzellinse wie `einzellinse_bk7.json` mit zwei Geisterplatten im Luftspalt hinter der Blende (Abbildung Datei- ↔ Tabellenflächen, Paraxial, CPU/GPU, M4-T5) |
| `einzellinse_geisterplatten_exakt.json` | frei erfundenes Testsystem: Biegung der Einzellinse (NA 0,025) mit vier Geisterplatten (Startkoeffizienten ≠ 0), Geisterkoeffizienten als Variablen und L1-Strafe `ghost_penalty`; Abnahme Exaktheit (alle \|a\| < 10⁻⁸ mm, M4-T6) |
| `einzellinse_na05_geisterplatten.json` | frei erfundenes Testsystem: Einzellinse N-BK7 f = 2 mm bei NA 0,5 mit Geisterplatten-Wörterbuch (CaF₂, N-BK7, Flint als Buchdahl-Modell an drei Lagen), μ aus dem Beugungsziel λ/14; Abnahme Bedarf und Beispiel für den Diagnosebericht (M4-T6) |
| `einzellinse_glasvariable.json` | frei erfundene Einzellinse f = 50 mm (NA 0,05, F/e/C) mit Glasvariable (Start N-BK7), Brennweite und Glas-Hülle aus `basis.json` und den frei erfundenen Testgläsern `data/glass/synthetisch.json`; Abnahme „Optimierung bleibt innerhalb der Hülle“ (M6-T1) |
| `achromat_glasvariablen.json` | frei erfundenes verkittetes Doppel f = 50 mm (NA 0,05, F/e/C) mit zwei Glasvariablen (Start N-BK7 und ein Flint mit den Buchdahl-Koordinaten des Testglases SYN-21), Brennweite und Glas-Hülle; Abnahme und Beispiel für das Glas-Einrasten `mikroptim snap` (M6-T5, [docs/cli.md](../../docs/cli.md#mikroptim-snap)) |
| `zweilinser_mechanik.json` | frei erfundener Zweilinser (zwei Sammellinsen n = 1,6, f = 10 mm, Pupillenradius 4 mm) mit mechanischen Nebenbedingungen (Strahlhöhe, Luftabstand, Glas- und Luftrand, Einfallswinkel); das freie Optimum verletzt jede Art. Abnahme M4-T3 und Beispiel für die Tabelle in `mikroptim optimize` |
| `einzellinse_achromat_start.json` | frei erfundene Einzellinse N-BK7, f = 50 mm (NA 0,05, 480 und 644 nm), Krümmungen und Arbeitsabstand variabel, Brennweite und Glas-Hülle; Abnahme des Operators B (Kittglied mit freiem Glas, Farblängsfehler um mehr als den Faktor 10 kleiner, M6-T2) und Beispiel für `mikroptim insert` |
| `einzellinse_homotopie_start.json` | frei erfundene Einzellinse aus einem hochbrechenden Glas (Buchdahl-Koordinaten des Testglases SYN-36), f = 10 mm, NA 0,3, Achse, 546 nm, mit Geisterplatten-Wörterbuch (N-BK7 und das hochbrechende Glas vor und hinter der Linse), L1-Strafe aus dem Beugungsziel und Rand- und Spaltbedingungen; Start der Abnahme des Homotopie-Controllers (NA 0,3 → 0,6, M6-T3) und Beispiel für `mikroptim synth` |
| `zweilinser_petzval.json` | frei erfundener Zweilinser (Sammellinse n = 1,8, Zerstreuungslinse CaF₂, f = 20 mm) mit Variablen, Brennweiten- und Petzval-Bedingung (κ_max = 0,01/mm); Abnahme der Petzval-Nebenbedingung (M4-T4) |

Geplante Start- und Referenzsysteme (Linsendaten trägt der Maintainer ein, Agenten erfinden keine Daten realer Objektive):

- Zeiss EC Epiplan-Apochromat 50x/0,95 BD DIC (Modell)
- DE 103 16 415 (100x/0,95 Luft)
- US 6,504,653 (100x/1,45 Öl)

Format: [docs/dateiformat.md](../../docs/dateiformat.md).
