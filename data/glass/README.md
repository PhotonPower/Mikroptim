# Glasdaten

- `basis.json`: wenige Gläser mit öffentlich verfügbaren Sellmeier-Koeffizienten für Tests (N-BK7, CaF₂, Quarzglas).
- `synthetisch.json`: **frei erfundene** Testgläser `SYN-01` … `SYN-36` (Sellmeier-Koeffizienten zufällig aus Bereichen gezogen, die grob Fluorkron bis Schwerflint und Lanthankron überdecken; n_d 1,43–1,81, ν_d 22–89), dazu `SYN-UV` und `SYN-IR`, die 400–700 nm absichtlich nicht abdecken. Keine realen Gläser, keine Herstellerdaten; nur für Tests und Beispiele der Glas-Hülle (M6-T1), z. B. `data/systems/einzellinse_glasvariable.json`. Erzeugt mit festem Startwert (Beschreibung in der Datei).
- **Herstellerkataloge werden nie committet** (Entscheidung des Maintainers, auch nicht bei freier Lizenz). Sie werden mit dem Glasimport (unten) umgewandelt und nach `data/glass/private/` (in `.gitignore`) bzw. in den Arbeitsbereich der GUI gelegt und zur Laufzeit geladen.
- Format: siehe [docs/dateiformat.md](../../docs/dateiformat.md#glasdaten).

## Eigene Kataloge für die Glas-Hülle (ab M6-T1)

Die Glas-Hülle (Nebenbedingung `glass_hull`, [docs/optimierung.md](../../docs/optimierung.md#24-glas-hülle-ab-m6-t1)) entsteht aus **allen** Gläsern der in der Systemdatei unter `glass_catalogs` genannten Kataloge, deren `range_nm` 400–700 nm abdeckt. Für reale Rechnungen gehört dorthin der Katalog des Herstellers:

1. Den Katalog mit dem Glasimport nach `mikroptim.glass/1` umwandeln und als `data/glass/private/<hersteller>.json` ablegen (Abschnitt „Glasimport“ unten). Das Verzeichnis ist in `.gitignore`; solche Dateien werden nicht committet (AGENTS.md, Abschnitt 8).
2. In der Systemdatei eintragen, z. B. `"glass_catalogs": ["data/glass/basis.json", "data/glass/private/hersteller.json"]` (relative Pfade werden vom Verzeichnis der Systemdatei aus aufwärts gesucht). Gläser dürfen in den Katalogen nicht doppelt vorkommen.
3. Die Hülle umfasst dann alle diese Gläser. Wer nur eine Auswahl will (z. B. ohne teure oder stark färbende Gläser), legt dafür einen eigenen Katalog mit der Auswahl an.

Gläser, die 400–700 nm nicht abdecken, gehen nicht in die Hülle ein (`GlassHull::excluded()`); ein einzelnes Glas ergibt einen Punkt, drei Gläser eine Ebene (dann ist kein Sicherheitsabstand `margin` möglich).

## Glasimport (ab M9-T3)

`mikroptim.glass_import` (Python-Paket, keine weiteren Abhängigkeiten außer NumPy; für YAML zusätzlich PyYAML) wandelt um:

- **AGF-Dateien** (Zemax-Format) der Glashersteller, wie sie Schott, Ohara, Hoya, CDGM, Sumita, Hikari u. a. auf ihren Websites anbieten. Kostenlos heißt dort nicht frei weitergebbar; die Dateien bleiben deshalb privat.
- **YAML-Dateien der Datenbank refractiveindex.info** (`database/data/...`, gemeinfrei nach CC0 1.0; Zitat: M. N. Polyanskiy, *Scientific Data* 11, 94 (2024)). Unter `database/data/specs/<hersteller>/` liegen dort die aus den Herstellerkatalogen erzeugten Gläser, unter `database/data/main/` Kristalle wie CaF₂.

```bash
python -m mikroptim.glass_import SCHOTT.AGF -o data/glass/private/schott.json --report schott_bericht.json
python -m mikroptim.glass_import OHARA.agf -o data/glass/private/ohara.json --exclude-names-from data/glass/basis.json
python -m mikroptim.glass_import refractiveindex.info-database/database/data/specs/ohara/optical/*.yml -o data/glass/private/ohara_rii.json
python -m mikroptim.glass_import Malitson.yml --name CAF2_MALITSON -o data/glass/private/caf2.json
```

Aus Python (z. B. für den Upload in der GUI):

```python
from mikroptim import glass_import as gi
result = gi.convert_files(["SCHOTT.AGF"], exclude_names_from=["data/glass/basis.json"])
gi.write_catalog(result, "data/glass/private/schott.json")
print(gi.format_report(result))       # Text; gi.report_dict(result) als dict/JSON
```

Regeln:

- **Exakt** übernommen werden alle Formeln, die sich in den Modellen `sellmeier` (bis 8 Terme, Konstante) und `polynomial` (ab M9-T4, ADR-0014) darstellen lassen: AGF 1, 2, 4, 6, 9–13; refractiveindex.info `formula 1`, `2`, `3`. Jede exakte Darstellung wird gegen die Originalformel nachgerechnet (Abweichung > 10⁻⁹ gilt als Fehler, dann Fit).
- Mit `--legacy` (bzw. `legacy=True`) nur drei Sellmeier-Terme ohne Konstante, für Mikroptim vor M9-T4; die übrigen Formeln werden dann gefittet.
- Alle anderen Formeln (AGF 3, 5, 7, 8 – Herzberger, Conrady, Handbook of Optics; refractiveindex.info 5, 6, 7; Tabellen; Umrechnung auf Luft) werden mit drei Sellmeier-Termen **gefittet**: zwei Pole unterhalb, einer oberhalb des Bereichs, Fehler in der Brechzahl gewichtet. Übernommen wird ein Glas nur, wenn die größte Abweichung |Δn| gegen die Originalformel unter der Toleranz bleibt (Standard 5·10⁻⁶, `--tolerance`). Gelingt das über den ganzen Gültigkeitsbereich nicht, wird nacheinander 365–1100 nm und 400–800 nm versucht; `range_nm` ist dann der eingeschränkte Bereich (Hinweis im Bericht). Tabellen werden an ihren Stützstellen gefittet und geprüft.
- **Kontrolle gegen n_d:** Steht im Katalog ein n_d, wird es mit dem Ergebnis bei 587,56 nm verglichen. Ab 2·10⁻⁵ Abweichung gibt es einen Hinweis, ab 10⁻⁴ wird das Glas verworfen (Formel vermutlich falsch gelesen).
- Absolute Brechzahlen bzw. Vakuumwellenlängen (refractiveindex.info `SPECS.n_absolute`, `wavelength_vacuum`) werden auf Luft umgerechnet (Edlén/Birch-Downs, 15 °C, 101325 Pa; der Unterschied zu 20 °C liegt unter 5·10⁻⁶).
- AGF-Gläser mit Status *obsolete* werden übersprungen (`--include-obsolete`). Doppelte Namen und `AIR` werden verworfen; `--prefix` setzt ein Präfix, `--exclude-names-from` überspringt Namen eines vorhandenen Katalogs (der Leser lehnt doppelte Namen beim Zusammenführen ab).
- Reintransmission wird als `transmission` mitgeschrieben (AGF: Zeilen `IT` der häufigsten Probendicke, bevorzugt 10 mm; refractiveindex.info: aus dem Extinktionskoeffizienten k für 10 mm). Mikroptim wertet sie noch nicht aus.
- `comment` enthält Formel, Fit-Abweichung bzw. „exakt“, Status sowie n_d, ν_d und P_g,F des Ergebnisses.
- Plausibilitätslauf (lokal, Herstellergläser der Datenbank refractiveindex.info von Schott, Ohara, CDGM, Hoya, Sumita, Hikari; 1.636 Gläser): 1.633 exakt übernommen, vom C++-Leser ohne Warnung geladen, C++ gegen Python ≤ 4,4·10⁻¹⁶, gegen das n_d des Katalogs ≤ 3,8·10⁻⁵ (Rundung der Kataloge); verworfen zwei tabellierte technische Gläser und eines mit inkonsistenter Quelle.
- Bericht `mikroptim.glass_import/1` (`--report`): pro Glas Verfahren (exakt, Fit, verworfen), größte Abweichung, Abweichung von n_d, Bereich, Grund des Verwerfens, Hinweise.

