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
| `ghost_plates` | object[] | nein | Geisterplatten (ab M4) |
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

### Geisterplatten (ab M4)

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

Die Platte liegt im Luftspalt nach `after_surface`, im Abstand `distance` von deren Scheitel. Dickenprofil nach numerik.md, Abschnitt 6.

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
- Ob Glasnamen in den Katalogen existieren, prüft `validate_materials` mit einer vom Aufrufer gelieferten Abfrage (Fehler unter `surfaces[i].material`), weil der Leser den Glaskatalog nicht kennt.
- `ghost_plates` (Liste), `variables` und `constraints` (Objekte) werden bis zur Festlegung ihres Schemas (M3/M4) als kompakter JSON-Text im Modell aufbewahrt und unverändert wieder geschrieben.
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
