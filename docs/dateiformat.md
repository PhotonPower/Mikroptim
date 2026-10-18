# Dateiformate

Alle Dateien sind JSON (UTF-8). Jede Datei trägt ein Feld `format` mit Name und Hauptversion. Leser lehnen unbekannte Hauptversionen ab und ignorieren unbekannte Felder mit Warnung. Inkompatible Änderungen erhöhen die Hauptversion (Entscheidung des Maintainers, siehe AGENTS.md).

## Systembeschreibung: `mikroptim.system/1`

Beispiel: [data/systems/einzellinse_bk7.json](../data/systems/einzellinse_bk7.json).

| Feld | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `format` | string | ja | `"mikroptim.system/1"` |
| `name` | string | ja | Kurzname |
| `description` | string | nein | Beschreibung |
| `units` | string | ja | nur `"mm"` |
| `glass_catalogs` | string[] | ja | Pfade relativ zum Repository-Wurzelverzeichnis oder zur Datei |
| `spectrum.wavelengths_nm` | number[] | ja | Wellenlängen |
| `spectrum.weights` | number[] | ja | Gewichte, gleiche Länge |
| `spectrum.reference_nm` | number | ja | Referenzwellenlänge (OPD in Wellen, Paraxialdaten) |
| `source.type` | string | ja | `"collimated"` (Standard) oder `"point"` (Tests) |
| `source.field_angles_deg` | number[] | bei collimated | Feldwinkel θ in der y-z-Ebene |
| `source.field_weights` | number[] | bei collimated | Gewichte, gleiche Länge |
| `source.point` | number[3] | bei point | Quellpunkt in lokalen Koordinaten der ersten Fläche |
| `source.numerical_aperture` | number | bei point | NA im Medium der Quelle |
| `target` | object | nein | Sollwerte, z. B. `focal_length`, `parfocal_length`, `working_distance` (ab M3) |
| `surfaces` | object[] | ja | Flächen in Tracerichtung, siehe unten |
| `ghost_plates` | object[] | nein | Geisterplatten (M4-T5), siehe unten |
| `variables`, `constraints` | object | nein | Optimierungsangaben (ab M3; Schema wird dort festgelegt) |

### Flächen

| Feld | Typ | Bedeutung |
|---|---|---|
| `type` | string | `"stop"` (nur erste Fläche, plan), `"plane"`, `"sphere"` |
| `curvature` | number | 1/mm, nur bei `sphere`; Vorzeichen nach numerik.md |
| `thickness` | number | Abstand zum nächsten Scheitel; bei der letzten Fläche Abstand zur Bildebene |
| `material` | string oder object | Medium **nach** der Fläche: `"AIR"`, Glasname aus einem Katalog, `{"model": "constant", "n": 1.5}` oder `{"model": "buchdahl", "n0": …, "nu": [ν₁, ν₂, ν₃]}` |
| `semi_diameter` | number | Halbdurchmesser (mechanisch); bei `stop` der Pupillenradius |
| `comment` | string | optional |

Krümmungen statt Radien, damit Planflächen ohne Sonderwert `inf` auskommen.

### Geisterplatten

Seit M4-T5 (Beispiel: [data/systems/einzellinse_geisterplatte.json](../data/systems/einzellinse_geisterplatte.json)):

```json
{
  "after_surface": 7,
  "distance": 0.5,
  "glass": "CAF2",
  "norm_radius": 4.0,
  "order": 8,
  "coefficients": [0, 0, 0, 0, 0, 0, 0, 0]
}
```

| Feld | Typ | Bedeutung |
|---|---|---|
| `after_surface` | ganze Zahl | Fläche i (0-basiert wie `surfaces`), nach der der Luftspalt liegt; 0 ≤ i ≤ S − 2 |
| `distance` | number | Abstand vom Scheitel der Fläche i [mm], 0 ≤ `distance` ≤ `surfaces[i].thickness` |
| `glass` | string oder object | virtuelles Glas: Glasname aus einem Katalog oder ein Materialobjekt (`constant`, `buchdahl`) wie bei `material`; nicht `"AIR"` |
| `norm_radius` | number | Normierungsradius r_g des Dickenprofils [mm], > 0 |
| `order` | ganze Zahl | Ordnung K = Zahl der Koeffizienten, 1 ≤ K ≤ 32 (`kMaxGhostTerms`) |
| `coefficients` | number[] | a_1 … a_K [mm], Länge `order` |

Die Platte liegt im Luftspalt nach `after_surface`, im Abstand `distance` von deren Scheitel. Dickenprofil t(r) = Σ a_k P_k(2(r/r_g)² − 1) und Physik nach [numerik.md, Abschnitt 6](numerik.md#6-geisterplatte). Mehrere Platten im selben Spalt sind erlaubt (auch am selben Ort); die Liste ist nach (`after_surface`, `distance`) aufsteigend sortiert. Im Modell: `mko::GhostPlate` in `SystemDescription::ghost_plates` (die Ordnung ist die Länge von `coefficients`).

#### Abbildung Dateiflächen ↔ Tabellenflächen

Tracer, Flächentabelle (`HostSurfaceTable`), Batch und Kernel arbeiten mit **Tabellenflächen**: den Dateiflächen, zwischen die die Geisterplatten als zusätzliche Flächen vom Typ `GhostPlate` eingefügt sind. Nach Dateifläche i folgen die Platten mit `after_surface` = i in der Reihenfolge der Datei. Der Luftspalt (Dicke T) wird geteilt: Dateifläche i erhält die Dicke d_1, die Platten d_2 − d_1, …, T − d_last; das Medium nach jeder Platte ist das des Spalts (Luft). Geisterplatte g der Datei ist die g-te Tabellenfläche vom Typ `GhostPlate` (`ghost_surface[g]` im Batch).

Beispiel `einzellinse_geisterplatte.json` (Platten bei 1,0 und 3,5 mm im 5-mm-Spalt nach der Blende):

| Tabellenfläche j | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Typ | Stop | GhostPlate (g = 0) | GhostPlate (g = 1) | Sphere | Sphere |
| Dateifläche | 0 | – | – | 1 | 2 |
| Dicke | 1,0 | 2,5 | 1,5 | 6,0 | 98,23 |

Die Abbildung liefert `mko::make_surface_map(system)` (`include/mikroptim/model/system.hpp`) als `SurfaceMap` mit `table_of_file`, `file_of_table` (−1 bei Geisterplatten), `ghost_of_table`, `table_of_ghost` und den Funktionen `file_surface(j)` und `table_surface(i)`, die zusätzlich die Referenzsphäre abbilden (j = S_Tabelle ↔ i = S_Datei). **Ausgaben mit Flächennummern beziehen sich auf Tabellenflächen**, insbesondere `failed_surface` in `CpuTraceResult`, `WavefrontMap` und der Batch-Trace; für Meldungen in Dateiflächen ist `file_surface` anzuwenden. `table_surface_count(system)` = Zahl der Tabellenflächen; zusammen höchstens `kMaxSurfaces` = 64. Die Paraxialoptik rechnet über dieselben Tabellenflächen. Optimierungsvariablen (M3) sehen einen geteilten Luftspalt als mehrere Dicken.

### Einlesen und Validierung

API in `include/mikroptim/io/system_json.hpp` (Bibliothek `Mikroptim::model`), Datenmodell `mko::SystemDescription` in `include/mikroptim/model/system.hpp`:

```cpp
SystemDescription read_system_json(std::string_view text, std::vector<std::string>* warnings = nullptr);
SystemDescription read_system_file(const std::filesystem::path& file, std::vector<std::string>* warnings = nullptr);
std::string write_system_json(const SystemDescription& system);
void write_system_file(const SystemDescription& system, const std::filesystem::path& file);
void validate_system(const SystemDescription& system);
void validate_materials(const SystemDescription& system, const std::function<bool(std::string_view)>& glass_exists);
std::vector<std::filesystem::path> resolve_glass_catalogs(const SystemDescription& system, const std::filesystem::path& base_dir);
```

Fehler im Inhalt werfen `mko::FormatError`. `json_path()` liefert den JSON-Pfad der beanstandeten Stelle (`surfaces[1].material`, `spectrum.weights[2]`), `what()` die Meldung in der Form `<datei>: <pfad>: <meldung>`. Syntaxfehler tragen keinen Pfad, aber Zeile und Spalte. Nicht lesbare Dateien werfen `std::runtime_error`.

Regeln beim Einlesen (erst die Struktur, danach `validate_system`):

- `format` wird zuerst geprüft: fremde Formatnamen und andere Hauptversionen als 1 werden abgelehnt.
- Pflichtfelder laut Tabelle; Typen werden streng geprüft (eine Zahl als Zeichenkette ist ein Fehler, ganze Zahlen sind als Zahlen erlaubt).
- `units` ist `"mm"`, `name` nicht leer, Einträge von `glass_catalogs` nicht leer.
- `spectrum`: mindestens eine Wellenlänge, alle > 0; `weights` gleich lang, jedes ≥ 0, Summe > 0; `reference_nm` > 0.
- `source`: bei `collimated` mindestens ein Feldwinkel mit |θ| < 90°, `field_weights` gleich lang (Regeln wie bei Gewichten); bei `point` genau drei Koordinaten und `numerical_aperture` > 0. Felder des jeweils anderen Typs werden mit Warnung ignoriert.
- `target` (optional): `focal_length` ≠ 0, `parfocal_length` und `working_distance` > 0.
- `surfaces`: mindestens zwei Flächen; `stop` genau an Position 0 und nur dort; die Blende liegt in Luft (Material `"AIR"`, ADR-0006); `curvature` bei `sphere` Pflicht, bei anderen Typen verboten; `thickness`, `material`, `semi_diameter` Pflicht, `semi_diameter` > 0. Dicken dürfen negativ sein (keine Plausibilitätsprüfung).
- `material`: `"AIR"`, sonst Glasname (nicht leer); Objekte mit `model` `constant` (`n` > 0) oder `buchdahl` (`n0` > 0, `nu` mit 2 oder 3 Koeffizienten, fehlende = 0).
- `ghost_plates` (Liste von Objekten, Pfade `ghost_plates[g].<feld>`): Pflichtfelder laut Tabelle; `after_surface` und `order` nichtnegative ganze Zahlen (`7` und `7.0` sind erlaubt); `order` gleich der Länge von `coefficients`, 1 ≤ `order` ≤ 32, Koeffizienten endlich; `after_surface` ≤ S − 2 (nach der letzten Fläche liegt der Arbeitsabstand) und das Medium nach `after_surface` ist `"AIR"` (Luftspalt); 0 ≤ `distance` ≤ Dicke des Spalts; `norm_radius` > 0; `glass` wie `material`, aber nicht `"AIR"`; die Liste ist nach (`after_surface`, `distance`) aufsteigend sortiert (Fehler unter `ghost_plates[g]`). Eine leere Liste wird nicht geschrieben.
- Ob Glasnamen in den Katalogen existieren, prüft `validate_materials` mit einer vom Aufrufer gelieferten Abfrage (Fehler unter `surfaces[i].material` bzw. `ghost_plates[g].glass`), weil der Leser den Glaskatalog nicht kennt.
- `variables` und `constraints` (Objekte) werden bis zur Festlegung ihres Schemas (M3) als kompakter JSON-Text im Modell aufbewahrt und unverändert wieder geschrieben.
- Unbekannte Felder werden ignoriert; als Warnung `"<pfad>: unbekanntes Feld wird ignoriert"` an `warnings` angehängt (die Bibliothek schreibt nichts auf stderr).

Glaskataloge (`resolve_glass_catalogs`): Relative Einträge werden zuerst relativ zu `base_dir` (Verzeichnis der Systemdatei) gesucht, danach in den Elternverzeichnissen aufwärts; so findet `data/systems/x.json` den Eintrag `data/glass/basis.json` im Repository-Wurzelverzeichnis. Nicht gefundene Einträge werfen `FormatError` unter `glass_catalogs[i]`.

Schreiben: Der Schreiber validiert das Modell (keine unlesbaren Dateien), gibt die Felder in der Reihenfolge der Tabellen aus, Zahlen verlustfrei in kürzester Darstellung (`10.0`, `0.00967`), zwei Leerzeichen Einrückung, Listen und Objekte einzeilig, solange die Zeile höchstens 120 Zeichen hat. `curvature` wird nur bei `sphere` geschrieben, `nu` ohne verschwindenden dritten Koeffizienten. Lesen → Schreiben → Lesen ergibt dasselbe Modell, Schreiben → Lesen → Schreiben denselben Text.

## Glasdaten: `mikroptim.glass/1` {#glasdaten}

Beispiel: [data/glass/basis.json](../data/glass/basis.json).

| Feld | Typ | Bedeutung |
|---|---|---|
| `glasses[].name` | string | eindeutiger Name (Großschreibung) |
| `glasses[].model` | string | `"sellmeier"` (weitere Modelle bei Bedarf per ADR) |
| `glasses[].B`, `glasses[].C` | number[3] | Sellmeier-Koeffizienten, λ in µm, C in µm² |
| `glasses[].range_nm` | number[2] | Gültigkeitsbereich |
| `glasses[].reference.n_d` | number | Kontrollwert bei 587,56 nm (Test) |
| `glasses[].transmission` | object | optional: Reintransmission, z. B. `{"thickness_mm": 10, "nm": [...], "tau": [...]}` |
| `glasses[].source` | string | Herkunft der Daten |

Brechzahlen sind relativ zu Luft.

### Einlesen (`mikroptim/glass/catalog.hpp`, ab M1-T2)

```cpp
mko::GlassCatalog cat = mko::GlassCatalog::from_file("data/glass/basis.json");
cat.merge(mko::GlassCatalog::from_file("data/glass/private/schott.json")); // mehrere Kataloge
const mko::Glass& g = cat.at("N-BK7");   // find() liefert nullptr statt Ausnahme
double n = g.index(587.56);              // nm; Ausnahme außerhalb von range_nm
```

Regeln des Lesers (Fehler als `mko::GlassError`, Meldung mit Datei und JSON-Pfad, z. B. `basis.json: glasses[2].B: Feld mit genau 3 Zahlen erwartet`):

- `format` muss genau `"mikroptim.glass/1"` sein; `mikroptim.glass/<n>` mit n ≠ 1 wird als unbekannte Hauptversion abgelehnt.
- Pflichtfelder je Glas: `name`, `model` (nur `"sellmeier"`), `B`, `C` (je 3 endliche Zahlen), `range_nm` (2 Zahlen, 0 < min < max).
- Namen sind eindeutig, nicht leer und nicht `AIR` (Schlüsselwort der Systemdatei). Der Vergleich beachtet Groß-/Kleinschreibung. Doppelte Namen werden in einer Datei und beim Zusammenführen (`merge`) abgelehnt.
- Ein Sellmeier-Pol (λ² = C_i) innerhalb von `range_nm` wird abgelehnt.
- Optional: `reference.n_d` (wird gespeichert), `source` (wird gespeichert), `transmission` und `comment` (werden ignoriert). Unbekannte Felder werden ignoriert und in `GlassCatalog::warnings()` gemeldet.

Die Brechzahltabelle für ein System beschreibt [architektur.md](architektur.md#brechzahltabelle).
