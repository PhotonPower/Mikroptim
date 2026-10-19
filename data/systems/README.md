# Systeme

| Datei | Zweck |
|---|---|
| `einzellinse_bk7.json` | einfache Testlinse (Paraxialdaten, sphärische Aberration, Farbe) |
| `einzellinse_bk7_variablen.json` | Einzellinse mit Abschnitt `variables` (Beispiel für das Schema, M3-T1) |
| `einzellinse_bk7_nebenbedingungen.json` | Einzellinse mit `target` und Abschnitt `constraints` (Beispiel für das Schema, M3-T3; Lage der Anlagefläche angenommen) |
| `einzellinse_bk7_biegung.json` | Einzellinse mit Krümmungen und Arbeitsabstand als Variablen und schwach gewichteter Brennweite; Beispiel für `mikroptim optimize` (Biegung minimaler sphärischer Aberration, M3-T5, [docs/cli.md](../../docs/cli.md#mikroptim-optimize)) |
| `planplatte_bk7.json` | exakt bekannte optische Weglänge |
| `testsystem_mehrlinser.json` | frei erfundenes Mehrlinsensystem (kein reales Objektiv): vier Gläser, drei Felder, vier Wellenlängen; CPU/GPU-Abgleich der Residuen (M2-T2) |
| `einzellinse_geisterplatte.json` | frei erfundenes Testsystem: Einzellinse wie `einzellinse_bk7.json` mit zwei Geisterplatten im Luftspalt hinter der Blende (Abbildung Datei- ↔ Tabellenflächen, Paraxial, CPU/GPU, M4-T5) |
| `zweilinser_petzval.json` | frei erfundener Zweilinser (Sammellinse n = 1,8, Zerstreuungslinse CaF₂, f = 20 mm) mit Variablen, Brennweiten- und Petzval-Bedingung (κ_max = 0,01/mm); Abnahme der Petzval-Nebenbedingung (M4-T4) |

Geplante Start- und Referenzsysteme (Linsendaten trägt der Maintainer ein, Agenten erfinden keine Daten realer Objektive):

- Zeiss EC Epiplan-Apochromat 50x/0,95 BD DIC (Modell)
- DE 103 16 415 (100x/0,95 Luft)
- US 6,504,653 (100x/1,45 Öl)

Format: [docs/dateiformat.md](../../docs/dateiformat.md).
