# Glasdaten

- `basis.json`: wenige Gläser mit öffentlich verfügbaren Sellmeier-Koeffizienten für Tests (N-BK7, CaF₂, Quarzglas).
- `synthetisch.json`: **frei erfundene** Testgläser `SYN-01` … `SYN-36` (Sellmeier-Koeffizienten zufällig aus Bereichen gezogen, die grob Fluorkron bis Schwerflint und Lanthankron überdecken; n_d 1,43–1,81, ν_d 22–89), dazu `SYN-UV` und `SYN-IR`, die 400–700 nm absichtlich nicht abdecken. Keine realen Gläser, keine Herstellerdaten; nur für Tests und Beispiele der Glas-Hülle (M6-T1), z. B. `data/systems/einzellinse_glasvariable.json`. Erzeugt mit festem Startwert (Beschreibung in der Datei).
- Vollständige Herstellerkataloge (Schott, Ohara, CDGM, Hikari …) werden **nur** committet, wenn ihre Lizenz das erlaubt. Sonst gehören sie nach `data/glass/private/` (in `.gitignore`) und werden zur Laufzeit geladen.
- Format: siehe [docs/dateiformat.md](../../docs/dateiformat.md#glasdaten).

## Eigene Kataloge für die Glas-Hülle (ab M6-T1)

Die Glas-Hülle (Nebenbedingung `glass_hull`, [docs/optimierung.md](../../docs/optimierung.md#24-glas-hülle-ab-m6-t1)) entsteht aus **allen** Gläsern der in der Systemdatei unter `glass_catalogs` genannten Kataloge, deren `range_nm` 400–700 nm abdeckt. Für reale Rechnungen gehört dorthin der Katalog des Herstellers:

1. Den Katalog nach `mikroptim.glass/1` umschreiben (Name, `"model": "sellmeier"`, `B`, `C` in µm², `range_nm`, optional `reference.n_d` und `source`; Format in [docs/dateiformat.md](../../docs/dateiformat.md#glasdaten)) und als `data/glass/private/<hersteller>.json` ablegen. Das Verzeichnis ist in `.gitignore`; solche Dateien werden nicht committet (AGENTS.md, Abschnitt 8). Einen Lader für `.agf` gibt es nicht.
2. In der Systemdatei eintragen, z. B. `"glass_catalogs": ["data/glass/basis.json", "data/glass/private/hersteller.json"]` (relative Pfade werden vom Verzeichnis der Systemdatei aus aufwärts gesucht). Gläser dürfen in den Katalogen nicht doppelt vorkommen.
3. Die Hülle umfasst dann alle diese Gläser. Wer nur eine Auswahl will (z. B. ohne teure oder stark färbende Gläser), legt dafür einen eigenen Katalog mit der Auswahl an.

Gläser, die 400–700 nm nicht abdecken, gehen nicht in die Hülle ein (`GlassHull::excluded()`); ein einzelnes Glas ergibt einen Punkt, drei Gläser eine Ebene (dann ist kein Sicherheitsabstand `margin` möglich).
