# Kommandozeilenprogramm `mikroptim`

```
mikroptim [--version] [--devices]
mikroptim trace <system.json> [--json] [--rings N] [--spokes M] [--gpu [--device N] [--precision fp64|df64]]
mikroptim optimize <system.json> [-o <ergebnis.json>] [--json] [--quiet] [--max-iterations N]
                   [--rings N] [--spokes M] [--runs K [--perturbation s] [--seed n]] [--gpu [--device N]]
```

| Aufruf | Wirkung |
|---|---|
| `mikroptim --version` | Version ausgeben |
| `mikroptim --devices` | CUDA-Geräte auflisten (nur mit `MIKROPTIM_ENABLE_CUDA=ON`) |
| `mikroptim trace …` | System auswerten (M1-T8, mit `--gpu` M2-T5; [Abschnitt `trace`](#mikroptim-trace)) |
| `mikroptim optimize …` | Variablen eines Systems optimieren, Ergebnis als neue Systemdatei (M3-T5; [Abschnitt `optimize`](#mikroptim-optimize)) |

Das Programm liegt nach dem Bauen unter `build/<preset>/mikroptim` (Ninja, Makefiles) bzw. `build/<preset>/<Konfiguration>/mikroptim.exe` (Visual Studio).

## `mikroptim trace`

Lädt eine Systembeschreibung `mikroptim.system/1` ([dateiformat.md](dateiformat.md)) samt Glaskatalogen und gibt aus:

- **Paraxialdaten** bei der Referenzwellenlänge `spectrum.reference_nm` (`compute_paraxial`, Definitionen und Vorzeichen: [numerik.md, Abschnitt 9](numerik.md#9-paraxialoptik)),
- **RMS-Wellenfront** pro Feld und Wellenlänge, in Wellen bei λ_ref und in mm, sowie die **Anzahl versagter Strahlen** (`evaluate_wavefront` mit den idealen Bildpunkten, [numerik.md, Abschnitt 7](numerik.md#7-referenzsphäre-und-opd); Bibliothek `Mikroptim::merit`, [architektur.md](architektur.md#wellenfront-m1-t7)).

Gerechnet wird in FP64, die Wellenfront standardmäßig auf der CPU, mit `--gpu` auf einem CUDA-Gerät (siehe [GPU-Backend](#gpu-backend---gpu)).

### Optionen

| Option | Bedeutung | Standard |
|---|---|---|
| `<system.json>` | Systemdatei; Glaskataloge werden relativ zu ihr aufgelöst | – |
| `--json` | maschinenlesbare Ausgabe `mikroptim.trace/1` statt Text | aus |
| `--rings N` | radiale Ringe der Pupillenquadratur, 1 ≤ N ≤ 16 | 10 |
| `--spokes M` | Speichen der Pupillenquadratur, M ≥ 1 (Achsfeld immer 1) | 8 |
| `--gpu` | Wellenfront auf der GPU rechnen (Kernel `RESIDUALS` über `mko::cuda::TraceEngine`) | aus (CPU) |
| `--device N` | CUDA-Gerät für `--gpu`, N ≥ 0 (Liste: `mikroptim --devices`); nur zusammen mit `--gpu` | 0 |
| `--precision P` | Rechentyp der Wellenfront: `fp64` oder `df64` (Paar aus zwei float, M7-T1; auf GeForce-Karten wie der RTX 3090 im Kernel etwa 4× schneller, Abweichung zu FP64 ≤ 10⁻⁹ mm). `df64` nur zusammen mit `--gpu`, die CPU rechnet immer in FP64 | `fp64` |
| `--help`, `-h` | Hilfe | – |

Die Pupillenquadratur folgt [numerik.md, Abschnitt 8](numerik.md#8-pupillenquadratur): Auf der Achse ein meridionaler Fächer mit N Stützstellen, außerhalb der Achse die halbe Pupille mit N·M Stützstellen (Punktquelle außerhalb der y-z-Ebene: volle Pupille).

### Exit-Codes und Fehler

| Code | Bedeutung |
|---|---|
| 0 | Auswertung ausgegeben, auch wenn die Wellenfront nicht auswertbar ist (siehe unten) |
| 1 | Fehler beim Laden oder Auswerten: Datei fehlt oder ist nicht lesbar, JSON-Syntax- oder Formatfehler, unbekanntes Glas, Brechzahl außerhalb des Gültigkeitsbereichs, … Mit `--gpu` auch: Gerät `--device N` existiert nicht oder ist ungeeignet (unter sm_60), CUDA-Fehler, Speichermangel |
| 2 | ungültige Kommandozeile (unbekannte Option, fehlende Datei, ungültige Zahl, `--device` ohne `--gpu`, `--precision df64` ohne `--gpu`, unbekannter Rechentyp) |
| 3 | `--gpu`, aber das GPU-Backend ist nicht verfügbar: ohne CUDA gebaut (`MIKROPTIM_ENABLE_CUDA=OFF`) oder kein CUDA-Gerät sichtbar. Geprüft vor dem Laden der Systemdatei (`kExitNoGpu`) |

Fehlermeldungen gehen auf stderr, die Standardausgabe bleibt dann leer. Formatfehler nennen Datei und JSON-Pfad, z. B.

```
mikroptim trace: Fehler: /pfad/system.json: surfaces[1].material: unbekanntes Glas "N-GIBTSNICHT" (in keinem der Glaskataloge enthalten)
```

Warnungen des Laders (unbekannte, ignorierte Felder in System- und Glasdateien) erscheinen als `mikroptim trace: Warnung: …` auf stderr und zusätzlich im JSON-Feld `warnings`.

**Afokale Systeme** (Brechkraft exakt 0, z. B. `planplatte_bk7.json`): Die Paraxialdaten werden ausgegeben, Brennweite, Schnittweiten, Hauptebenen und Defokus sind nicht endlich. Ohne `target.focal_length` gibt es keine Sollbrennweite für die Bildpunkte der Feldwinkel; die Wellenfront ist dann nicht auswertbar. Die Textausgabe meldet das, im JSON sind `wavefront`, `rays`, `failed_rays` und `target_focal_length` `null` und `wavefront_error` enthält den Grund. Der Exit-Code bleibt 0.

### Textausgabe

Zahlen werden printf-artig mit fester Stellenzahl in der C-Locale (Dezimalpunkt) geschrieben: Längen in mm mit 6 Nachkommastellen (`%.6f`), Größen in 1/mm und RMS in mm mit 6 Stellen in Exponentialschreibweise (`%.6e`), Wellenlängen und Feldwinkel mit 3 Nachkommastellen, RMS in Wellen mit 6 Nachkommastellen. Nicht endliche Werte erscheinen als `unendlich` bzw. `-` (nicht definiert, NaN), ein Minuszeichen vor einer zu null gerundeten Zahl entfällt. So liefern Linux und Windows dieselbe Ausgabe. Lagen beziehen sich wie in `ParaxialData` auf den Scheitel von Fläche 0 bzw. der letzten Fläche, positiv in Tracerichtung (+z).

Beispiel (`mikroptim trace data/systems/einzellinse_bk7.json`, aus dem Repository-Wurzelverzeichnis):

```
System: Einzellinse N-BK7
  Datei                                       data/systems/einzellinse_bk7.json
  Quelle                                      kollimiert, Feldwinkel 0.000, 1.000, 2.000 deg
  Wellenlängen                                486.130, 587.560, 656.270 nm
  Referenzwellenlänge                         587.560 nm
  Flächen                                     3

Paraxialdaten bei 587.560 nm
  Brennweite f'                               101.049681 mm
  Brechkraft                                  9.896122e-03 1/mm
  Hintere Schnittweite (ab letzter Fläche)    99.052088 mm
  Vordere Schnittweite (ab Fläche 0)          -94.052088 mm
  Hauptebene H (ab Fläche 0)                  6.997593 mm
  Hauptebene H' (ab letzter Fläche)           -1.997593 mm
  Defokus der Bildebene                       0.002088 mm
  Petzvalsumme                                6.589473e-03 1/mm
  Petzvalkrümmung                             -6.589473e-03 1/mm
  Petzvalradius                               -151.757193 mm
  Austrittspupille Lage (ab letzter Fläche)   -9.515815 mm
  Austrittspupille Radius                     10.744012 mm
  Pupillenvergrößerung                        1.074401

Wellenfront (Quadratur 10 Ringe x 8 Speichen, RMS in Wellen bei 587.560 nm)
  Backend                                     CPU, FP64
  Sollbrennweite für die Bildpunkte           101.049681 mm
  Feld  Winkel [deg]  Wellenlänge [nm]  Strahlen  versagt  RMS [Wellen]       RMS [mm]
     0         0.000           486.130        10        0      4.502093   2.645250e-03
     0         0.000           587.560        10        0      1.936014   1.137524e-03
     0         0.000           656.270        10        0      0.866035   5.088477e-04
     1         1.000           486.130        80        0      4.602484   2.704235e-03
     1         1.000           587.560        80        0      2.082975   1.223873e-03
     1         1.000           656.270        80        0      1.098614   6.455015e-04
     2         2.000           486.130        80        0      4.896489   2.876981e-03
     2         2.000           587.560        80        0      2.478862   1.456480e-03
     2         2.000           656.270        80        0      1.613832   9.482230e-04
  Versagte Strahlen: 0 von 510
```

Die Tabelle hat eine Zeile pro Feld und Wellenlänge in der Reihenfolge der Ablage `maps[f · L + l]`. `Strahlen` ist die Zahl der Pupillenstützstellen, `versagt` die Zahl der versagten Strahlen (an einer Fläche oder an der Referenzsphäre); RMS über die gültigen Strahlen, `-`, wenn kein Strahl gültig ist. Die Bildebene der Testlinse liegt im paraxialen Fokus; die RMS-Werte von einigen Wellen sind die sphärische Aberration und der Farblängsfehler der unkorrigierten Einzellinse bei f/5.

### JSON-Ausgabe (`--json`), Format `mikroptim.trace/1`

Ein JSON-Objekt (eingerückt mit zwei Leerzeichen, abschließender Zeilenumbruch). Zahlen werden verlustfrei geschrieben (kürzeste Darstellung, die beim Einlesen denselben FP64-Wert ergibt), sind also bitgleich zu den Werten der Bibliotheksfunktionen. **Nicht endliche Werte (±∞, NaN) sind in JSON nicht darstellbar und werden `null`**; ob ein `null` „unendlich“ oder „nicht definiert“ bedeutet, ergibt sich aus den Flags (`afocal`, `exit_pupil.at_infinity`) und den Definitionen in `ParaxialData`. Längen in mm, Wellenlängen in nm.

| Feld | Typ | Inhalt |
|---|---|---|
| `format` | string | `"mikroptim.trace/1"` |
| `mikroptim_version` | string | Programmversion (`mko::kVersion`) |
| `system` | string | `name` der Systembeschreibung |
| `file` | string | Pfad der Systemdatei wie auf der Kommandozeile angegeben |
| `reference_nm` | number | Referenzwellenlänge λ_ref |
| `wavelengths_nm` | number[] | Wellenlängen des Spektrums, Index `l` |
| `source` | object | `{"type": "collimated", "field_angles_deg": [...]}` oder `{"type": "point", "point": [x, y, z], "numerical_aperture": NA}` |
| `paraxial` | object | Paraxialdaten bei λ_ref, siehe unten |
| `quadrature` | object | `{"rings": N, "spokes": M}` wie angefordert |
| `backend` | string | Rechenwerk der Wellenfront: `"cpu"` oder `"gpu"` (seit M2-T5; die Paraxialdaten entstehen immer auf dem Host) |
| `device` | object \| null | bei `"gpu"`: `{"index": N, "name": "<Gerätename>"}`, sonst `null` (seit M2-T5) |
| `precision` | string | Rechentyp der Wellenfront: `"fp64"` oder `"df64"` (nur mit `--gpu`); kompatible Erweiterung nach M7-T1 |
| `target_focal_length` | number \| null | verwendete Sollbrennweite f_soll für die Bildpunkte (`target.focal_length` oder paraxial); `null` bei der Punktquelle oder ohne Wellenfront |
| `wavefront` | object[] \| null | ein Eintrag pro Feld f und Wellenlänge l, Reihenfolge `f · L + l`; `null`, wenn nicht auswertbar |
| `wavefront_error` | string \| null | Grund, warum `wavefront` `null` ist |
| `rays` | integer \| null | Summe der Strahlen über alle Einträge |
| `failed_rays` | integer \| null | Summe der versagten Strahlen über alle Einträge |
| `warnings` | string[] | Warnungen des Laders |

`paraxial` (Bedeutung wie in `mko::ParaxialData`, [architektur.md](architektur.md#paraxialoptik)):

| Feld | Einheit | Inhalt |
|---|---|---|
| `wavelength_nm` | nm | Wellenlänge der Paraxialrechnung (= `reference_nm`) |
| `afocal` | bool | Brechkraft exakt 0 |
| `power` | 1/mm | Brechkraft φ |
| `focal_length` | mm | Brennweite 1/φ (`null` bei afokal) |
| `back_focal_distance` | mm | letzte Fläche → F′ |
| `front_focal_distance` | mm | Fläche 0 → F |
| `front_principal_plane` | mm | H relativ zu Fläche 0 |
| `rear_principal_plane` | mm | H′ relativ zur letzten Fläche |
| `image_defocus` | mm | Schnittweite − Dicke der letzten Fläche (> 0: Fokus hinter der Bildebene) |
| `petzval_sum` | 1/mm | Petzvalsumme P |
| `petzval_curvature` | 1/mm | κ_P = −n′ P |
| `petzval_radius` | mm | 1/κ_P (`null` bei κ_P = 0) |
| `exit_pupil.at_infinity` | bool | Austrittspupille im Unendlichen (bildseitig telezentrisch) |
| `exit_pupil.position` | mm | Lage relativ zur letzten Fläche |
| `exit_pupil.magnification` | – | Pupillenvergrößerung 1/D |
| `exit_pupil.semi_diameter` | mm | Radius der Austrittspupille |
| `abcd` | – | Systemmatrix `[A, B, C, D]` in reduzierten Winkeln |

Einträge von `wavefront`:

| Feld | Typ | Inhalt |
|---|---|---|
| `field` | integer | Feldindex f |
| `field_angle_deg` | number \| null | Feldwinkel θ_f (kollimiert), `null` bei der Punktquelle |
| `wavelength` | integer | Wellenlängenindex l |
| `wavelength_nm` | number | Wellenlänge |
| `image_point` | number[3] | Mittelpunkt **M**_f der Referenzsphäre, lokale Koordinaten der letzten Fläche [mm] |
| `pupil_mode` | string | `"axis"`, `"half"` oder `"full"` |
| `rays` | integer | Pupillenstützstellen |
| `failed_rays` | integer | versagte Strahlen |
| `rms_waves` | number \| null | RMS-Wellenfront in Wellen bei λ_ref (`null` ohne gültigen Strahl) |
| `rms_mm` | number \| null | RMS-Wellenfront [mm] |
| `piston_mm` | number \| null | Kolben Σ W·OPL / Σ W [mm]; mit `--gpu` immer `null` (der Kernel `RESIDUALS` gibt den Kolben nicht aus) |

**Stabilität:** Neue Felder können in `mikroptim.trace/1` hinzukommen; Leser sollen unbekannte Felder ignorieren. Umbenennen, Entfernen oder Bedeutungsänderungen erhöhen die Version auf `mikroptim.trace/2`.

Beispiel (`mikroptim trace data/systems/einzellinse_bk7.json --json`):

<details>
<summary>Vollständige Ausgabe</summary>

```json
{
  "format": "mikroptim.trace/1",
  "mikroptim_version": "0.1.0",
  "system": "Einzellinse N-BK7",
  "file": "data/systems/einzellinse_bk7.json",
  "reference_nm": 587.56,
  "wavelengths_nm": [
    486.13,
    587.56,
    656.27
  ],
  "source": {
    "type": "collimated",
    "field_angles_deg": [
      0.0,
      1.0,
      2.0
    ]
  },
  "paraxial": {
    "wavelength_nm": 587.56,
    "afocal": false,
    "power": 0.009896122289808292,
    "focal_length": 101.04968094724019,
    "back_focal_distance": 99.05208836399525,
    "front_focal_distance": -94.05208836399525,
    "front_principal_plane": 6.9975925832449395,
    "rear_principal_plane": -1.9975925832449377,
    "image_defocus": 0.002088363995255804,
    "petzval_sum": 0.006589473496335327,
    "petzval_curvature": -0.006589473496335327,
    "petzval_radius": -151.75719282521442,
    "exit_pupil": {
      "at_infinity": false,
      "position": -9.515814775176269,
      "magnification": 1.074401246213303,
      "semi_diameter": 10.74401246213303
    },
    "abcd": [
      0.980231579510994,
      8.856853813893544,
      -0.009896122289808292,
      0.9307509680619526
    ]
  },
  "quadrature": {
    "rings": 10,
    "spokes": 8
  },
  "backend": "cpu",
  "device": null,
  "precision": "fp64",
  "target_focal_length": 101.04968094724019,
  "wavefront": [
    {
      "field": 0,
      "field_angle_deg": 0.0,
      "wavelength": 0,
      "wavelength_nm": 486.13,
      "image_point": [
        0.0,
        0.0,
        99.05
      ],
      "pupil_mode": "axis",
      "rays": 10,
      "failed_rays": 0,
      "rms_waves": 4.502092950462526,
      "rms_mm": 0.0026452497339737612,
      "piston_mm": 14.130347293916047
    },
    {
      "field": 0,
      "field_angle_deg": 0.0,
      "wavelength": 1,
      "wavelength_nm": 587.56,
      "image_point": [
        0.0,
        0.0,
        99.05
      ],
      "pupil_mode": "axis",
      "rays": 10,
      "failed_rays": 0,
      "rms_waves": 1.9360135431732859,
      "rms_mm": 0.0011375241174268957,
      "piston_mm": 14.099531981210557
    },
    {
      "field": 0,
      "field_angle_deg": 0.0,
      "wavelength": 2,
      "wavelength_nm": 656.27,
      "image_point": [
        0.0,
        0.0,
        99.05
      ],
      "pupil_mode": "axis",
      "rays": 10,
      "failed_rays": 0,
      "rms_waves": 0.8660353587622437,
      "rms_mm": 0.0005088477353943438,
      "piston_mm": 14.085840661889346
    },
    {
      "field": 1,
      "field_angle_deg": 1.0,
      "wavelength": 0,
      "wavelength_nm": 486.13,
      "image_point": [
        0.0,
        1.763828741909749,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 4.602483961671361,
      "rms_mm": 0.0027042354765196243,
      "piston_mm": 14.128888270713023
    },
    {
      "field": 1,
      "field_angle_deg": 1.0,
      "wavelength": 1,
      "wavelength_nm": 587.56,
      "image_point": [
        0.0,
        1.763828741909749,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 2.082974772167356,
      "rms_mm": 0.0012238726571346514,
      "piston_mm": 14.098071862450222
    },
    {
      "field": 1,
      "field_angle_deg": 1.0,
      "wavelength": 2,
      "wavelength_nm": 656.27,
      "image_point": [
        0.0,
        1.763828741909749,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 1.0986136789180152,
      "rms_mm": 0.0006455014531850688,
      "piston_mm": 14.08438005235029
    },
    {
      "field": 2,
      "field_angle_deg": 2.0,
      "wavelength": 0,
      "wavelength_nm": 486.13,
      "image_point": [
        0.0,
        3.5287326155732273,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 4.896489155733337,
      "rms_mm": 0.002876981168342679,
      "piston_mm": 14.124511478829488
    },
    {
      "field": 2,
      "field_angle_deg": 2.0,
      "wavelength": 1,
      "wavelength_nm": 587.56,
      "image_point": [
        0.0,
        3.5287326155732273,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 2.478862452445385,
      "rms_mm": 0.0014564804225588102,
      "piston_mm": 14.093691785779615
    },
    {
      "field": 2,
      "field_angle_deg": 2.0,
      "wavelength": 2,
      "wavelength_nm": 656.27,
      "image_point": [
        0.0,
        3.5287326155732273,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 1.6138317561135485,
      "rms_mm": 0.0009482229866220763,
      "piston_mm": 14.079998504173323
    }
  ],
  "wavefront_error": null,
  "rays": 510,
  "failed_rays": 0,
  "warnings": []
}
```

</details>

Für `planplatte_bk7.json` (afokal) sind in `paraxial` die Felder `focal_length`, `back_focal_distance`, `front_focal_distance`, `front_principal_plane`, `rear_principal_plane`, `image_defocus` und `petzval_radius` `null`, `afocal` ist `true`, und es gilt `"wavefront": null` mit `"wavefront_error": "Wellenfront: Sollbrennweite ist nicht endlich (afokales System?); target.focal_length setzen"`.

### GPU-Backend (`--gpu`)

Mit `--gpu` entsteht die Wellenfront über die Host-API `mko::cuda::TraceEngine` ([architektur.md](architektur.md), Abschnitt „Host-API TraceEngine“) auf dem Gerät `--device N` (Standard 0):

1. Sollbrennweite f_soll wie im CPU-Pfad mit `target_focal_length` (nur kollimierte Quelle), dann `make_system_batch(system, index, 1, f_soll)` – eine Variante mit den Feldzielen h_f = f_soll · tan θ_f, also denselben idealen Bildpunkten wie `evaluate_wavefront`.
2. `TraceEngine::run(batch, TraceMode::Residuals, {quadrature = {N, M}, opd = true, status = true})`.
3. `tuple_wavefronts(out.residuals, out.layout, 0)` liefert RMS [mm] und versagte Strahlen pro Feld und Wellenlänge; RMS in Wellen = RMS / λ_ref wie im CPU-Pfad.

Paraxialdaten, Sollbrennweite und Bildpunkte entstehen auf dem Host und sind bitgleich zur CPU-Variante; Text und JSON werden von denselben Ausgabefunktionen geschrieben. Unterschiede zur CPU-Variante:

- `rms_waves`, `rms_mm`: Abweichung höchstens max(`kCpuGpuFp64`, `kCpuGpuFp64Relative` · |RMS|) (`tests/tolerances.hpp`; FMA-Kontraktion und CUDA-Mathematikbibliothek). Gemessen auf der RTX 3090: höchstens 7,7·10⁻¹⁵ mm (relativ ≤ 4,8·10⁻¹²). Die Textausgabe ist bei 6 Stellen deshalb praktisch immer identisch; sie unterscheidet sich nur in der Zeile `Backend`.
- `failed_rays` und `rays` sind identisch (gleiches Versagenskriterium im header-only Tracer).
- `piston_mm` ist `null`: Der Kernel `RESIDUALS` zieht den Kolben ab, gibt ihn aber nicht aus.
- `backend` ist `"gpu"`, `device` nennt Index und Namen des Geräts.

**df64** (`--gpu --precision df64`): Die Kernel rechnen im Zahlentyp df64 (numerik.md, Abschnitt 12), Batch und Ausgaben bleiben FP64. RMS-Werte weichen von der CPU höchstens um max(`kDf64VsFp64`, `kDf64VsFp64Relative` · |RMS|) ab (10⁻⁹ mm bzw. 10⁻¹⁰ relativ); Strahlzahlen und versagte Strahlen sind in den Testsystemen identisch (an der Versagensgrenze kann der Status in Ausnahmefällen kippen). Im JSON ist `precision` `"df64"`, die Zeile `Backend` endet auf `df64`. Test: `CliTraceGpu.Df64MatchesCpuWithinDf64Tolerance`.

Afokale Systeme ohne `target.focal_length` verhalten sich wie im CPU-Pfad (`wavefront: null`, gleiche Meldung in `wavefront_error`, Exit-Code 0); die Punktquelle (Achse, y-z-Ebene, allgemein) wird unterstützt. Ohne CUDA gebaut oder ohne sichtbares Gerät endet `--gpu` mit einer Meldung auf stderr und Exit-Code 3, z. B.

```
mikroptim trace: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), GPU-Backend nicht verfügbar
mikroptim trace: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht verfügbar)
```

Beispiel (`mikroptim trace data/systems/einzellinse_bk7.json --gpu`, RTX 3090): Ausgabe wie oben, nur die Zeile `Backend` lautet

```
Wellenfront (Quadratur 10 Ringe x 8 Speichen, RMS in Wellen bei 587.560 nm)
  Backend                                     GPU 0 (NVIDIA GeForce RTX 3090), FP64
  Sollbrennweite für die Bildpunkte           101.049681 mm
```

Mit `--json` ändern sich gegenüber dem Beispiel oben nur

```json
  "backend": "gpu",
  "device": {
    "index": 0,
    "name": "NVIDIA GeForce RTX 3090"
  },
  "precision": "fp64",
```

sowie in den Einträgen von `wavefront` die RMS-Werte in den letzten Stellen und der Kolben, z. B. für Feld 0 bei 486,13 nm `"rms_waves": 4.5020929504633`, `"rms_mm": 0.002645249733974216`, `"piston_mm": null` (CPU: `4.502092950462526`, `0.0026452497339737612`, `14.130347293916047`).

### Tests

`tests/test_cli_trace.cpp` ruft `mko::cli::run_trace_command` direkt auf und prüft die JSON-Ausgabe gegen `compute_paraxial` und `evaluate_wavefront` (bitgleich), das afokale System, die Punktquelle, Warnungen, Fehlerfälle und die Textausgabe. Die ctest-Kommandos `cli_trace_json`, `cli_trace_text` und `cli_trace_missing_file` starten das gebaute Programm.

`--gpu`: Im CPU-Build prüfen `CliTrace.GpuUnavailableWithoutCuda` und das ctest-Kommando `cli_trace_gpu_unavailable` Meldung und Exit-Code 3 (läuft in der CI). Im CUDA-Build vergleichen die Tests `CliTraceGpu.*` (übersprungen ohne Gerät) GPU- gegen CPU-Ausgabe für `einzellinse_bk7.json`, `testsystem_mehrlinser.json` (auch mit 16 × 12 Stützstellen), ein System mit versagten Strahlen, die Punktquelle und das afokale System; dazu die ctest-Kommandos `cli_trace_gpu_json` (Exit-Code 3 ohne Gerät = übersprungen, `SKIP_RETURN_CODE`) und `cli_trace_gpu_no_device` (`CUDA_VISIBLE_DEVICES=-1`).

Implementierung: `src/cli/trace_command.cpp` (Bibliothek `mikroptim_cli`, interner Header `src/cli/trace_command.hpp`; im CUDA-Build gelinkt gegen `mikroptim_cuda`, GPU-Pfad unter `#ifdef MIKROPTIM_HAS_CUDA`), `src/cli/main.cpp` verteilt nur die Unterbefehle.

## `mikroptim optimize`

Optimiert die Variablen einer Systemdatei (Abschnitt `variables`, [dateiformat.md](dateiformat.md)) mit dem gebündelten Levenberg-Marquardt ([optimierung.md, Abschnitt 4](optimierung.md#4-gebündelter-levenberg-marquardt)) und schreibt das Ergebnis als neue Systemdatei. Ausgegeben werden ein Iterationsprotokoll und eine Zusammenfassung: Status, Zyklen, Merit vorher/nachher, Variablen vorher/nachher mit Schranken, Nebenbedingungen Ist/Soll und RMS-Wellenfront pro Feld und Wellenlänge vorher/nachher.

Die Merit ist dieselbe wie in der Bibliothek (`optimize_lm_cpu` bzw. `cuda::optimize_lm_gpu`): F = ‖r‖² aus den Strahlresiduen der Wellenfront (Pupillenquadratur `--rings` × `--spokes`) und den paraxialen Zusatzresiduen der Nebenbedingungen (Abschnitt `constraints`, `make_paraxial_residuals(make_paraxial_constraints(system))`, [optimierung.md, Abschnitt 2](optimierung.md#2-residuenvektor)). Die Einheit ist die der Bibliothek: seit M4-T1 **Wellen² bei λ_ref** (Strahlresiduen √(ŵ_f ŵ_λ W_i V_j)·OPD/λ_ref, paraxiale Zusatzresiduen √w·q mit w in 1/mm²; Wahl der Gewichte: [optimierung.md, Abschnitt 2.2](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3)). Text und JSON nennen sie (`[Wellen²]`, `merit_unit`). Gerechnet wird in FP64.

### Optionen

| Option | Bedeutung | Standard |
|---|---|---|
| `<system.json>` | Systemdatei mit Abschnitt `variables` (optional `constraints`) | – |
| `-o DATEI`, `--output DATEI` | Ergebnis als neue Systemdatei schreiben (siehe [Ergebnisdatei](#ergebnisdatei)); ohne `-o` nur Protokoll | keine Datei |
| `--json` | maschinenlesbare Ausgabe `mikroptim.optimize/1` statt Text | aus |
| `--quiet`, `-q` | ohne Iterationsprotokoll (Text: nur Kopf und Zusammenfassung, JSON: `"iterations": null`) | aus |
| `--max-iterations N` | Zyklen pro Lauf (Versuchsschritte einschließlich verworfener, `LmOptions::max_iterations`), N ≥ 0; 0 wertet nur den Startpunkt aus | 100 |
| `--rings N`, `--spokes M` | Pupillenquadratur der Strahlresiduen wie bei `trace` | 10, 8 |
| `--runs K` | Multistart mit K ≥ 1 Läufen (siehe unten) | 1 |
| `--perturbation s` | relative Störung der Startpunkte, s ≥ 0; nur mit `--runs` ≥ 2 | 0,05 |
| `--seed n` | Startwert der Störungen (0 … 2⁶⁴ − 1); nur mit `--runs` ≥ 2 | 0 |
| `--gpu` | auf der GPU rechnen (`cuda::optimize_lm_gpu`: Jacobi-Matrix und Normalgleichungen auf dem Gerät, Merit im Modus MERIT der TraceEngine) | aus (CPU, `optimize_lm_cpu`) |
| `--device N` | CUDA-Gerät für `--gpu`; nur zusammen mit `--gpu` | 0 |
| `--help`, `-h` | Hilfe | – |

Die übrigen Einstellungen des Verfahrens (Toleranzen, Dämpfung, zentrale finite Differenzen, Ablehnung neuer versagter Strahlen) sind die Standardwerte von `LmOptions`. Die Zahl der Host-Threads wählt die Bibliothek; das Ergebnis ist davon unabhängig bitgleich.

**Multistart.** Mit `--runs K` laufen K Läufe gemeinsam: Lauf 0 startet im Punkt x₀ der Datei, Lauf j ≥ 1 im gestörten Punkt `perturbed_vector(set, x₀, options, j)` mit `relative = s` für alle Variablentypen, `seed = n` und `first_is_base` (gleichverteilt, plattformunabhängig, auf die Schranken projiziert; [optimierung.md, Variantengenerator](optimierung.md#variantengenerator-ab-m3-t1)). Lauf 0 ist bitgleich zum Aufruf ohne `--runs` (die Läufe sind voneinander unabhängig), der Multistart ist also nie schlechter. **Bester Lauf:** endliche Merit, dann die wenigsten versagten Strahlen (hart versagte Strahlen haben r = 0, eine Merit mit mehr versagten Strahlen ist nicht vergleichbar), dann die kleinste Merit, bei Gleichstand der kleinste Index. Die Auswahl ist deterministisch; zwei Aufrufe mit gleichen Argumenten liefern dieselbe Ausgabe.

### Ergebnisdatei

Die Ergebnisdatei ist die Eingabe mit den Variablenwerten des besten Laufs (`VariableSet::apply_to_description`, danach `write_system_file`: Zahlen verlustfrei; Name, Beschreibung, Variablen, Nebenbedingungen und Kommentare unverändert). Sie lässt sich mit `mikroptim trace` und `mikroptim optimize` laden und liefert dort **bitgleich dieselbe Merit** wie die Endmerit der Optimierung, sofern dasselbe Backend rechnet: Die Datei enthält x exakt (kürzeste Darstellung, die beim Einlesen denselben FP64-Wert ergibt), der Lader baut daraus denselben Batch wie `lm_point_batch` (auch die Dicke eines Luftspalts mit Geisterplatten, letztes Teilstück fl(x − d)), und die Merit ist eine deterministische Funktion des Batches. Prüfen lässt sich das mit `mikroptim optimize <ergebnis.json> --max-iterations 0`. CPU und GPU unterscheiden sich um die Rundung der Strahlresiduen: Weicht die OPD jedes Strahls um höchstens δ = `kCpuGpuFp64` ab, gilt ‖Δr‖₂ ≤ N·δ mit N = √(Σ_{f,l} s_{f,l}²) (`residual_scale_norm`) und damit |ΔF| ≤ 2√F·Nδ + (Nδ)².

- **Glaskataloge:** Ein Eintrag von `glass_catalogs` bleibt unverändert, wenn er vom Verzeichnis der Ergebnisdatei aus (mit der Suche in den Elternverzeichnissen, [dateiformat.md](dateiformat.md)) dieselbe Datei findet; sonst wird er durch den relativen Pfad dorthin ersetzt (bzw. den absoluten, wenn es keinen relativen gibt, z. B. anderes Laufwerk).
- **Eingabedatei:** Ist `-o` die Eingabedatei (auch über einen anderen Pfad oder eine Verknüpfung), bricht der Befehl vor dem Laden mit Exit-Code 2 ab; die Eingabe wird nie überschrieben, eine Option zum Erzwingen gibt es bewusst nicht. Eine andere vorhandene Datei wird überschrieben.
- **Feldziele:** Die idealen Bildpunkte h_f = f_soll · tan θ_f bleiben während der Optimierung fest (Batch des Startsystems). Ohne `target.focal_length` ist f_soll die paraxiale Brennweite des *Startsystems*; beim Laden des Ergebnisses ergeben sich dann andere Feldziele und, bei Feldwinkeln ≠ 0, eine andere Merit. Der Befehl warnt in diesem Fall (`mikroptim optimize: Warnung: Feldziele aus der paraxialen Brennweite des Startsystems …`, auch im JSON-Feld `warnings`); für reproduzierbare Ergebnisse `target.focal_length` setzen.

### Exit-Codes und Fehler

Wie bei `trace`:

| Code | Bedeutung |
|---|---|
| 0 | Optimierung abgeschlossen, auch ohne Konvergenz (Status `max_iterations`, `damping_limit`; siehe `result.status`) |
| 1 | Fehler beim Laden (wie `trace`), **keine Variablen** (Abschnitt `variables` fehlt oder ist leer), Nebenbedingung passt nicht zum System, Merit in keinem Lauf endlich, Fehler beim Schreiben der Ergebnisdatei (z. B. Verzeichnis fehlt); mit `--gpu` auch: Gerät `--device N` existiert nicht, CUDA-Fehler |
| 2 | ungültige Kommandozeile (wie `trace`; zusätzlich `-o` ohne Datei oder gleich der Eingabedatei, `--runs` < 1, `--max-iterations` < 0, `--perturbation` < 0, `--perturbation`/`--seed` ohne `--runs` ≥ 2) |
| 3 | `--gpu`, aber ohne CUDA gebaut oder kein CUDA-Gerät (vor dem Laden geprüft) |

Fehlermeldungen beginnen mit `mikroptim optimize: Fehler: `, z. B.

```
mikroptim optimize: Fehler: data/systems/einzellinse_bk7.json: keine Variablen (Abschnitt variables fehlt oder ist leer)
```

Bei Exit-Code 1 bleibt die Standardausgabe leer, außer wenn der Fehler erst nach dem Start der Optimierung auftritt (Textausgabe: Kopf und Protokoll stehen dann schon dort). Die Ergebnisdatei wird nur nach erfolgreicher Optimierung geschrieben.

### Textausgabe

Zahlenformat wie bei `trace` (C-Locale, plattformunabhängig; Linux/GCC und Windows/MSVC liefern dieselbe Ausgabe): Merit mit 9 Stellen in Exponentialschreibweise, Krümmungen in 1/mm ebenso, Längen in mm mit 6 Nachkommastellen, nicht endliche Werte als `unendlich`/`-unendlich` bzw. `-`. Das Protokoll wird während der Rechnung geschrieben, eine Zeile pro Zyklus und Lauf (in Laufreihenfolge, `LmOptions::on_iteration`):

| Spalte | Inhalt (`LmIterationInfo`) |
|---|---|
| `Lauf`, `Zyklus` | Index des Laufs, Zyklen des Laufs nach diesem |
| `Schritt` | `angenommen`, `verworfen` oder `gescheitert` (Cholesky-Zerlegung gescheitert, kein Versuchspunkt) |
| `Merit` | F nach dem Zyklus (bei Ablehnung unverändert) |
| `Testmerit` | F am Versuchspunkt |
| `Gewinn` | Gewinnverhältnis ρ = (F − F_neu)/(L(0) − L(Δx)) |
| `Dämpfung` | μ, mit dem der Schritt gerechnet wurde |
| `Schrittnorm` | max_k \|Δx_k\|/s_k des Versuchsschritts |
| `versagt` | versagte Strahlen am Versuchspunkt |
| `Status` | Endstatus, wenn der Lauf in diesem Zyklus endet; das meldet die Bibliothek nur für `damping_limit` und `converged_merit` bei F = 0. Die übrigen Endzustände (`converged_*`, `max_iterations`) stellt sie erst vor dem nächsten Zyklus fest, ohne Rückruf; sie stehen in der Zusammenfassung |

Die Wellenfronttabelle der Zusammenfassung rechnet immer die CPU (`evaluate_wavefront` mit den idealen Bildpunkten, wie `mikroptim trace` ohne `--gpu`) für das Start- und das Ergebnissystem; „nachher“ ist also genau das, was `mikroptim trace <ergebnis.json>` ausgibt.

### Beispiel

Die Einzellinse `data/systems/einzellinse_bk7_biegung.json` (wie `einzellinse_bk7.json` mit Blendenradius 2,5 mm, also NA ≈ 0,025, eine Wellenlänge, nur Achsfeld) hat als Variablen beide Krümmungen in [−0,05; 0,05] und den Arbeitsabstand in [50; 150] mm; die Mittendicke 6 mm ist fest. Die Brennweite 100 mm ist eine schwach gewichtete Gleichung (Gewicht 0,029/mm², das Gewicht der Abnahmetests von M3-T4 nach der Faustregel in [optimierung.md, Abschnitt 2.2](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3); ein viel größeres Gewicht verdeckt die Biegung in der Rundung der Normalgleichungen, [optimierung.md, Abschnitt 4](optimierung.md#verfahren-ab-m3-t4)). Das Optimum ist die Biegung minimaler sphärischer Aberration bei fester Brennweite.

Aufruf aus dem Repository-Wurzelverzeichnis (CPU; gekürzt, … markiert ausgelassene Zeilen):

```
mikroptim optimize data/systems/einzellinse_bk7_biegung.json -o einzellinse_bk7_biegung_opt.json
```

```
Optimierung: Einzellinse N-BK7, Biegung
  Datei                                       data/systems/einzellinse_bk7_biegung.json
  Ergebnisdatei                               einzellinse_bk7_biegung_opt.json
  Backend                                     CPU, FP64
  Quadratur                                   10 Ringe x 8 Speichen
  Variablen                                   3
  Nebenbedingungen (paraxial)                 1
  Läufe                                       1
  Zyklen höchstens                            100

Iterationsprotokoll (Merit F = Summe r² in Wellen² bei 587.560 nm, docs/optimierung.md)
  Lauf  Zyklus     Schritt            Merit        Testmerit     Gewinn   Dämpfung Schrittnorm versagt  Status
     0       1  angenommen  6.747402936e-06  6.747402936e-06  9.999e-01  1.000e-03   1.116e-02       0
     0       2  angenommen  3.797180076e-06  3.797180076e-06  1.000e+00  3.333e-04   9.135e-05       0
     0       3  angenommen  3.796146522e-06  3.796146522e-06  9.999e-01  1.111e-04   1.312e-04       0
…
     0      42  angenommen  1.557334269e-06  1.557334269e-06  6.601e-01  1.914e-08   4.349e-04       0
     0      43  angenommen  1.557334158e-06  1.557334158e-06  8.857e-01  1.851e-08   1.135e-04       0
     0      44  angenommen  1.557334153e-06  1.557334153e-06  4.909e-02  1.001e-08   1.489e-04       0
     0      45  angenommen  1.557334135e-06  1.557334135e-06  5.096e-01  1.735e-08   6.196e-05       0
…
     0      53   verworfen  1.557334134e-06  1.557334138e-06 -1.405e+04  4.738e-05   3.283e-09       0
     0      54   verworfen  1.557334134e-06  1.557334137e-06 -1.006e+05  3.790e-04   4.109e-10       0

Ergebnis
  Status                                      converged_step
  Zyklen                                      54
  Angenommene Schritte                        46
  Jacobi-Matrizen                             47
  Merit vorher [Wellen²]                      3.134515536e-02
  Merit nachher [Wellen²]                     1.557334134e-06
  Merit nachher/vorher                        4.968341e-05
  Versagte Strahlen vorher/nachher            0 / 0

Variablen
  Name    Einheit   untere Schranke           vorher          nachher   obere Schranke
  c[1]    1/mm     -5.000000000e-02  9.670000000e-03  1.672853973e-02  5.000000000e-02
  c[2]    1/mm     -5.000000000e-02 -9.670000000e-03 -2.714119101e-03  5.000000000e-02
  w       mm              50.000000        99.050000        96.548727       150.000000

Nebenbedingungen (Istwerte in mm, Residuum r = Wurzel(w) q)
   Nr  Typ              Modus     Gewicht         Soll   Ist vorher  Ist nachher      r vorher     r nachher
    0  focal_length     equal   2.900e-02   100.000000   101.049681   100.000002  1.768974e-01  2.730606e-07

Wellenfront (Quadratur 10 Ringe x 8 Speichen, CPU, FP64, RMS in Wellen bei 587.560 nm)
  Feld  Winkel [deg]  Wellenlänge [nm]  Strahlen    RMS vorher   RMS nachher  versagt vorher/nachher
     0         0.000           587.560        10      0.007244      0.001248                   0 / 0

Ergebnis geschrieben: einzellinse_bk7_biegung_opt.json
```

Die Startmerit 3,1·10⁻² Wellen² ist fast ganz die Brennweitenbedingung (r = 0,177, Wellenfront 0,0072 Wellen RMS). Im ersten Zyklus bringt LM die Brennweite fast auf 100 mm und fokussiert über den Arbeitsabstand; danach wandert die Linse mit Gewinnverhältnissen um 0,7 zur besten Biegung (die Gauß-Newton-Krümmung in Biegerichtung verschwindet am Optimum, [optimierung.md, Abschnitt 4](optimierung.md#verfahren-ab-m3-t4)). Am Ende liegen die Merit-Änderungen im Rundungsrauschen, die Schritte werden verworfen, bis die Schrittnorm unter `xtol` fällt (`converged_step`). Der Formfaktor q = (c₁ + c₂)/(c₁ − c₂) = 0,7208 ist der Wert der Bibliothekstests (Dünnlinsentheorie 2(n² − 1)/(n + 2) = 0,7397; der Unterschied ist der Dickeneffekt).

**Ergebnisdatei** (Auszug; der Glaskatalog `data/glass/basis.json` bleibt unverändert, weil die Datei im Wurzelverzeichnis liegt):

```json
  "surfaces": [
    { "type": "stop", "thickness": 5.0, "material": "AIR", "semi_diameter": 2.5 },
    {
      "type": "sphere",
      "curvature": 0.01672853972858829,
      "thickness": 6.0,
      "material": "N-BK7",
      "semi_diameter": 12.5
    },
    {
      "type": "sphere",
      "curvature": -0.002714119101320638,
      "thickness": 96.54872747022759,
      "material": "AIR",
      "semi_diameter": 12.5
    }
  ],
```

**Vorher/nachher mit `mikroptim trace`** (`data/systems/einzellinse_bk7_biegung.json` gegen `einzellinse_bk7_biegung_opt.json`, alle Zeilen, die sich unterscheiden):

| Größe | vorher | nachher |
|---|---|---|
| Brennweite f′ | 101,049681 mm | 100,000002 mm |
| Brechkraft | 9,896122·10⁻³ 1/mm | 1,000000·10⁻² 1/mm |
| Hintere Schnittweite (ab letzter Fläche) | 99,052088 mm | 96,580179 mm |
| Vordere Schnittweite (ab Fläche 0) | −94,052088 mm | −94,445153 mm |
| Hauptebene H (ab Fläche 0) | 6,997593 mm | 5,554848 mm |
| Hauptebene H′ (ab letzter Fläche) | −1,997593 mm | −3,419822 mm |
| Defokus der Bildebene | 0,002088 mm | 0,031452 mm |
| Petzvalsumme | 6,589473·10⁻³ 1/mm | 6,624451·10⁻³ 1/mm |
| Austrittspupille Lage / Radius | −9,515815 mm / 2,686003 mm | −9,301383 mm / 2,647039 mm |
| RMS-Wellenfront (Achse, 587,56 nm) | 0,007244 Wellen (4,256352·10⁻⁶ mm) | 0,001248 Wellen (7,332349·10⁻⁷ mm) |

Die RMS-Wellenfront sinkt auf ein Sechstel; die Endmerit 1,557·10⁻⁶ Wellen² ist praktisch ihr Quadrat (0,001248² Wellen², eine Wellenlänge, ein Feld), der Beitrag der Brennweite ist (2,7·10⁻⁷)². Die Bildebene liegt jetzt 0,031 mm vor dem paraxialen Fokus: Bei sphärischer Aberration liegt der Fokus kleinster RMS-Wellenfront nicht im paraxialen Fokus, die Wellenfront-Merit findet ihn über den Arbeitsabstand.

**Multistart** (`--runs 8 --perturbation 0.2 --seed 1 --quiet`, Auszug): Alle Läufe enden im selben Optimum; die Endmerits unterscheiden sich nur im Rundungsrauschen (≈ 10⁻⁹ relativ), hier gewinnt Lauf 4.

```
Läufe
  Lauf  Status               Zyklen      Merit Start       Merit Ende  versagt
     0  converged_step           54  3.134515536e-02  1.557334134e-06        0
     1  converged_step           55  1.544867972e+00  1.557334136e-06        0
     2  converged_step           52  5.204384846e+00  1.557334136e-06        0
     3  converged_step           59  4.067522266e+00  1.557334134e-06        0
     4  converged_step           51  1.154977668e+00  1.557334134e-06        0  bester
     5  converged_step           51  4.200103488e+00  1.557334136e-06        0
     6  converged_step           63  7.160670488e+00  1.557334135e-06        0
     7  converged_step           47  2.021973972e+00  1.557334135e-06        0
```

**GPU** (`--gpu`, RTX 3090): Kopfzeile `Backend  GPU 0 (NVIDIA GeForce RTX 3090), FP64`; konvergiert nach 56 Zyklen (`converged_step`) mit q = 0,720768 und Endmerit 1,557334138·10⁻⁶ Wellen² (CPU: 54 Zyklen, q = 0,720808, 1,557334134·10⁻⁶; die Rundungsunterschiede der Strahlresiduen ändern den Weg im flachen Tal, nicht das Optimum). Die GPU-Ergebnisdatei ergibt auf der CPU eine um 1,1·10⁻¹⁵ Wellen² andere Merit (Schranke oben: 4,2·10⁻¹²).

### JSON-Ausgabe (`--json`), Format `mikroptim.optimize/1`

Ein JSON-Objekt am Ende der Rechnung (eingerückt mit zwei Leerzeichen, abschließender Zeilenumbruch); Zahlen verlustfrei wie bei `trace`, nicht endliche Werte `null`. Längen in mm, Krümmungen in 1/mm, Wellenlängen in nm.

| Feld | Typ | Inhalt |
|---|---|---|
| `format` | string | `"mikroptim.optimize/1"` |
| `mikroptim_version` | string | Programmversion |
| `system`, `file` | string | `name` der Systembeschreibung, Pfad wie angegeben |
| `output` | string \| null | Ergebnisdatei wie angegeben, `null` ohne `-o` |
| `backend`, `device`, `precision` | | wie bei `trace` (`precision` immer `"fp64"`) |
| `merit_unit` | string | Einheit aller Merit-Werte (`merit`, `trial_merit`, `start_merit`): `"waves^2"` (Wellen² bei λ_ref, seit M4-T1) |
| `quadrature` | object | `{"rings": N, "spokes": M}` |
| `settings` | object | `max_iterations`, `runs`, `perturbation` und `seed` (beide `null` bei `runs` = 1) |
| `variables` | object[] | pro Variable k: `name` (`"c[1]"`, `"t[2]"`, `"w"`), `type`, `surface` (Dateifläche), `lower`, `upper` (`null` = offen), `typical`, `start` (x₀), `result` (x des besten Laufs) |
| `constraints` | object[] | pro paraxialer Nebenbedingung: `item` (Eintrag in `constraints.items`), `type`, `mode`, `weight`, `target` (Soll), `value_start`, `value_result` (Ist: Brennweite 1/φ, Arbeitsabstand, Abgleichlänge; bei `paraxial_focus` die Randstrahlhöhe q in der Bildebene), `residual_start`, `residual_result` (r = √w·q̃ wie im Residuenvektor) |
| `iterations` | object[] \| null | Protokoll in der Reihenfolge der Rückrufe: `run`, `iteration`, `status`, `evaluated`, `accepted`, `merit`, `trial_merit`, `gain_ratio`, `damping`, `step_norm`, `failed_rays` (wie `LmIterationInfo`); `null` mit `--quiet` |
| `runs` | object[] | pro Lauf: `run`, `status`, `iterations`, `accepted`, `jacobians`, `start_merit` (F am Startpunkt), `merit`, `failed_rays`, `start` (Startpunkt), `x` (Endpunkt) |
| `best_run` | integer | Index des besten Laufs |
| `result` | object | bester Lauf: `status`, `converged` (einer der drei Converged-Zustände), `iterations`, `accepted`, `jacobians`, `start_merit` (F(x₀) der Datei, = `runs[0].start_merit`), `merit`, `start_failed_rays`, `failed_rays`, `damping`, `gradient_norm`, `model_decrease` (wie `LmRunResult`) |
| `reference_nm` | number | λ_ref für `rms_waves` |
| `wavefront` | object \| null | `{"start": [...], "result": [...]}`, Einträge pro Feld und Wellenlänge (Reihenfolge f · L + l): `field`, `field_angle_deg`, `wavelength`, `wavelength_nm`, `rays`, `failed_rays`, `rms_waves`, `rms_mm` (wie bei `trace`); `null`, wenn nicht auswertbar |
| `wavefront_error` | string \| null | Grund, warum `wavefront` `null` ist (afokales System ohne `target.focal_length`) |
| `warnings` | string[] | Warnungen des Laders und des Befehls (Feldziele) |

Status-Namen: `converged_merit`, `converged_step`, `converged_gradient`, `max_iterations`, `damping_limit`, `numerical_error` (`to_string(LmStatus)`), im Protokoll zusätzlich `running`. **Stabilität** wie bei `mikroptim.trace/1`: Neue Felder können hinzukommen; Umbenennen, Entfernen oder Bedeutungsänderungen erhöhen die Version auf `mikroptim.optimize/2`.

Beispiel (Auszug aus `mikroptim optimize data/systems/einzellinse_bk7_biegung.json --json`, zur Kürze teilweise in einer Zeile; die echte Ausgabe schreibt jedes Feld in eine eigene Zeile):

```json
{
  "format": "mikroptim.optimize/1",
  "mikroptim_version": "0.1.0",
  "system": "Einzellinse N-BK7, Biegung",
  "file": "data/systems/einzellinse_bk7_biegung.json",
  "output": null,
  "backend": "cpu",
  "device": null,
  "precision": "fp64",
  "merit_unit": "waves^2",
  "quadrature": { "rings": 10, "spokes": 8 },
  "settings": { "max_iterations": 100, "runs": 1, "perturbation": null, "seed": null },
  "variables": [
    { "name": "c[1]", "type": "curvature", "surface": 1, "lower": -0.05, "upper": 0.05,
      "typical": 0.01, "start": 0.00967, "result": 0.01672853972858829 },
    …
  ],
  "constraints": [
    { "item": 0, "type": "focal_length", "mode": "equal", "weight": 0.029, "target": 100.0,
      "value_start": 101.04968094724019, "value_result": 100.00000160346728,
      "residual_start": 0.17689736616623206, "residual_result": 2.730606328905618e-07 }
  ],
  "iterations": [
    { "run": 0, "iteration": 1, "status": "running", "evaluated": true, "accepted": true,
      "merit": 6.7474029362767126e-06, "trial_merit": 6.7474029362767126e-06,
      "gain_ratio": 0.9999097450120137, "damping": 0.001, … },
    …
  ],
  "runs": [ { "run": 0, "status": "converged_step", "iterations": 54, … } ],
  "best_run": 0,
  "result": {
    "status": "converged_step",
    "converged": true,
    "iterations": 54,
    "accepted": 46,
    "jacobians": 47,
    "start_merit": 0.031345155357940434,
    "merit": 1.557334133980554e-06,
    "start_failed_rays": 0,
    "failed_rays": 0,
    "damping": 0.0060647987285953695,
    "gradient_norm": 5.76557505989509e-09,
    "model_decrease": 8.035301977234237e-06
  },
  "reference_nm": 587.56,
  "wavefront": {
    "start": [ { "field": 0, …, "rms_waves": 0.007244114948732903, "rms_mm": 4.256352179277504e-06 } ],
    "result": [ { "field": 0, …, "rms_waves": 0.0012479319129737986, "rms_mm": 7.33234874786885e-07 } ]
  },
  "wavefront_error": null,
  "warnings": []
}
```

### Tests

`tests/test_cli_optimize.cpp` ruft `mko::cli::run_optimize_command` direkt auf:

- `CliOptimize.MeritDecreasesAndResultReproducesMerit`: Biegung der Einzellinse, Merit 3,13·10⁻² → 1,56·10⁻⁶ Wellen², x und Merit bitgleich zu `optimize_lm_cpu`, q = 0,7208; die Ergebnisdatei liefert beim Neuladen denselben Variablenvektor und **bitgleich dieselbe Merit** (über `CpuLmBackend` und über `optimize --max-iterations 0`); `mikroptim trace` lädt sie (Glaskatalog relativ zum neuen Verzeichnis) und gibt dieselbe Wellenfront aus wie „nachher“.
- `CliOptimize.GhostPlateResultReproducesMerit`: Einzellinse mit Geisterplatten, Dicke des geteilten Spalts als Variable, bitgleiche Merit nach dem Neuladen; absoluter Glaskatalog bleibt unverändert.
- `CliOptimize.JsonIterationsMatchLibraryHistory`: ein Protokolleintrag pro Zyklus, Merit bitgleich zu `LmRunResult::history`; `--quiet` ohne Protokoll.
- `CliOptimize.ConstraintsReportedBeforeAndAfter`: Istwerte und Residuen gegen `compute_paraxial` und die Dicken (`einzellinse_bk7_nebenbedingungen.json`).
- `CliOptimize.MultistartIsDeterministic`: zwei Aufrufe gleiche Ausgabe, Startpunkte bitgleich zu `perturbed_vector`, bester Lauf nach der Regel oben, Lauf 0 gleich dem Einzellauf.
- `CliOptimize.WarnsAboutFieldTargetsFromStartFocalLength`, `CliOptimize.TextOutput`, `CliOptimize.Errors` (keine Variablen, fehlende Datei, `-o` gleich der Eingabe, fehlendes Ausgabeverzeichnis, Kommandozeilenfehler), `CliOptimize.GpuUnavailableWithoutCuda` (CPU-Build).
- Im CUDA-Build `CliOptimizeGpu.MatchesCpuAndReproducesMerit` (übersprungen ohne Gerät): GPU-Optimierung konvergiert mit demselben Formfaktor; Neuladen auf der GPU bitgleich, auf der CPU innerhalb der Schranke oben; Multistart auf der GPU deterministisch. `CliOptimizeGpu.InvalidDevice`.

ctest-Kommandos mit dem gebauten Programm: `cli_optimize_text` (schreibt `einzellinse_bk7_biegung_opt.json` ins Build-Verzeichnis), danach `cli_optimize_result_trace` (`mikroptim trace` auf dieser Ergebnisdatei, f′ = 100,0000 mm) und `cli_optimize_no_variables`; im CPU-Build `cli_optimize_gpu_unavailable` (Exit-Code 3), im CUDA-Build `cli_optimize_gpu_json` (übersprungen ohne Gerät) und `cli_optimize_gpu_no_device`.

Implementierung: `src/cli/optimize_command.cpp` (interner Header `src/cli/optimize_command.hpp`, Bibliothek `mikroptim_cli`, gelinkt gegen `mikroptim_optim`, im CUDA-Build `mikroptim_cuda`); Exit-Codes, Zahlen aus Argumenten und Textformatierung teilen sich `trace` und `optimize` in `src/cli/cli_util.hpp`.
