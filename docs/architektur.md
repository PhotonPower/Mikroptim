# Architektur

## Schichten

```mermaid
flowchart TD
    IO[Datenmodell und IO<br/>JSON-Systeme, Glasdaten] --> GL[Glasmodell<br/>Sellmeier, Buchdahl, Brechzahltabellen]
    IO --> BATCH[SystemBatch<br/>SoA, Host- und Device-Sicht]
    GL --> BATCH
    BATCH --> CPU[CPU-Referenztracer]
    BATCH --> GPU[CUDA-Backend<br/>Kernel-Modi MERIT, RESIDUALS, ADJOINT, EVAL]
    CORE[Mathe-Kern host+device<br/>Flächenoperationen, Geisterplatte, Referenzsphäre, Quadratur] --> CPU
    CORE --> GPU
    GPU --> MERIT[Merit und Residuen]
    CPU --> MERIT
    MERIT --> LM[Gebündelter Levenberg-Marquardt]
    LM --> SYN[Synthese<br/>Homotopie, Geisterdiagnose, Einfüge-Operatoren, Glas-Einrasten]
    SYN --> BATCH
    GPU --> EVAL[Auswertung<br/>PSF, MTF, Strehl, Tolerierung]
    SYN --> CLI[CLI und Python-Bindings]
    EVAL --> CLI
```

## Module und geplante Pfade

| Modul | Header | Quellen | Meilenstein |
|---|---|---|---|
| Mathe-Kern | `include/mikroptim/math/`, `include/mikroptim/trace/` | header-only | M0, M1, M4 |
| Datenmodell und IO | `include/mikroptim/model/`, `include/mikroptim/io/` | `src/model/`, `src/io/` | M1 |
| Glasmodell | `include/mikroptim/glass/` | `src/glass/` | M1, M6 |
| Pupillenquadratur | `include/mikroptim/pupil/` | header-only | M1 |
| Paraxialoptik | `include/mikroptim/paraxial/` | `src/paraxial/` | M1 |
| SystemBatch | `include/mikroptim/batch/` | `src/batch/` | M2 |
| CPU-Tracer, Lader | `include/mikroptim/cpu/` | `src/cpu/` | M1 |
| CUDA-Backend | `include/mikroptim/cuda/` (reines C++) | `src/cuda/` | M2–M8 |
| Merit (ab M1-T7: Wellenfront) | `include/mikroptim/merit/` | `src/merit/` | M1, M3, M4 |
| Optimierer (ab M3-T1: Variablen, Variantengenerator; M3-T2: FD-Jacobi) | `include/mikroptim/optim/` | `src/optim/`, `src/cuda/` | M3, M5 |
| Synthese (ab M6-T3: Homotopie-Controller, `Mikroptim::synth`) | `include/mikroptim/synth/` | `src/synth/` | M6 |
| Auswertung (ab M8-T1: Modus EVAL, `Mikroptim::eval`; M8-T2: PSF, MTF, Strehl) | `include/mikroptim/eval/` | `src/eval/`, `src/cuda/` | M8 |
| CLI ([cli.md](cli.md)) | – (intern `src/cli/trace_command.hpp`) | `src/cli/` | M1 ff. |
| Python | – | `python/` | M9 |

## Datenmodell und IO

Statische Host-Bibliothek `mikroptim_model` (Alias `Mikroptim::model`), gelinkt in CLI und Tests:

- `include/mikroptim/model/system.hpp`: `mko::SystemDescription` mit `Spectrum`, `Source`, `Target`, `Surface` (Typen `SurfaceType`, `SourceType`), Validierung `validate_system` und `validate_materials`.
- `include/mikroptim/model/variables.hpp`: Abschnitt `variables` (`mko::Variables`, `VariableSpec`, `VariableType`), Auflösung und Prüfung `resolve_variables` (ab M3-T1).
- `include/mikroptim/model/material.hpp`: `mko::MaterialSpec` (Medium nach einer Fläche; gemeinsam mit dem Glasmodell).
- `include/mikroptim/model/format_error.hpp`: `mko::FormatError` mit JSON-Pfad.
- `include/mikroptim/io/system_json.hpp`: Einlesen und Schreiben von `mikroptim.system/1`, Auflösung der Glaskatalog-Pfade (Details: [dateiformat.md](dateiformat.md#einlesen-und-validierung)).

nlohmann/json (v3.12.0, FetchContent) wird nur in Quellen unter `src/` eingebunden und privat gelinkt; öffentliche Header bleiben frei von JSON-Typen.

## Mathe-Kern

Alle Rechnungen, die pro Strahl laufen, liegen header-only im Mathe-Kern, sind `MKO_HD` und über den Zahlentyp `R` templatisiert (ADR-0005). CPU-Tracer und CUDA-Kernel rufen dieselben Funktionen auf. Der sequenzielle Trace über ein System ist eine Funktion

```cpp
template <class R, class SurfaceTable, class Observer = NoSurfaceObserver>
MKO_HD TraceStatus trace_ray(const SurfaceTable& sys, int wavelength_index, Ray<R>& ray,
                             TraceRecord<R>* record, int* failed_surface = nullptr,
                             Observer* observer = nullptr);
```

`SurfaceTable` ist eine schlanke Sicht (Zeiger und Längen) auf die Flächendaten einer Variante, im Kernel auf Shared Memory. `record` ist optional und speichert Zwischenzustände für den adjungierten Modus. `observer` (optional, ab M2-T3) sieht jede passierte Fläche, z. B. für Strahlhöhen und Einfallswinkel ([Max-Reduktionen](#max-reduktionen-m2-t3)). Details im folgenden Abschnitt.

## Sequenzieller Trace (M1-T5)

Ablauf, Start- und Endzustand und Statuscodes: [numerik.md](numerik.md#sequenzieller-trace-m1-t5).

### Mathe-Kern (header-only, `MKO_HD`)

`include/mikroptim/trace/surface_table.hpp`:

```cpp
inline constexpr int kMaxSurfaces = 64;                     // inkl. Blende und Geisterplatten
inline constexpr int kMaxGhostTerms = 32;                   // größte Ordnung K einer Geisterplatte
enum class SurfaceKind : std::uint8_t { Stop = 0, Plane = 1, Sphere = 2, GhostPlate = 3 };
template <class R> struct SurfaceTableView {                // trivial kopierbar
    int surfaces, wavelengths;                              // S, L
    const SurfaceKind* type; const R* curvature; const R* thickness;  // [S]
    const R* index;                                         // [S·L], index[s·L + l] = Medium nach Fläche s
    int ghost_plates = 0, ghost_terms = 0;                  // G, K (M4-T5)
    const R* ghost_coeff = nullptr;                         // [G·K], a_k der Geisterplatte g bei g·K + k − 1
    const R* ghost_index = nullptr;                         // [G·L], n_G bei g·L + l
    const R* ghost_norm_radius = nullptr;                   // [G], r_g
};
```

`trace_ray` greift nur über die Namen `surfaces`, `wavelengths`, `type[s]`, `curvature[s]`, `thickness[s]`, `index[s * wavelengths + l]` und die Geisterfelder (Abschnitt [Geisterplatten](#geisterplatten-m4-t5)) auf die Tabelle zu. Jeder Typ mit diesen Namen ist eine gültige Tabelle, z. B. feste Felder im Shared Memory oder (M2) eine Batch-Sicht, deren Zeiger um v·S bzw. v·S·L versetzt sind. Der CUDA-Prüfkernel `src/cuda/trace_check.cu` lädt die Tabelle pro Block in den Shared Memory (dynamisch, (2S + S·L)·8 + S Byte) und baut die Sicht darauf.

`include/mikroptim/trace/trace.hpp`:

```cpp
template <class R> struct TraceRecord {          // optional (Tests, Diagnose)
    int count;                                   // gültige Einträge; bei Versagen an Fläche s: count = s
    Vec3<R> p[kMaxSurfaces], d[kMaxSurfaces];    // Schnittpunkt (lokal) und Richtung nach Fläche s
    R opl[kMaxSurfaces];                         // Eintrag 0 = Startzustand
    unsigned char branch[kMaxSurfaces];          // Zweig an Fläche s (M5-T1, kBranch…)
    R image_s; unsigned char image_branch;       // Schritt auf die Referenzsphäre (M5-T1)
};
struct TraceProgress { int surface, ghost, soft_surface; };   // M5-T1, Zustand zwischen Flächen
struct NoTraceRecorder;                          // Aufzeichnung ohne Wirkung (Recorder-Schnittstelle)
template <class R, class T, class Rec, class Obs = NoSurfaceObserver>   // einzige Flächenschleife (M5-T1)
MKO_HD TraceStatus trace_segment(const T& sys, int l, Ray<R>&, TraceProgress& at, int end, Rec*,
                                 int* failed_surface = nullptr, Obs* = nullptr, SoftFailure<R>* = nullptr);
template <class R, class T, class Rec, class Obs = NoSurfaceObserver>   // trace_ray mit beliebigem Recorder
MKO_HD TraceStatus trace_ray_recorded(const T& sys, int l, Ray<R>&, Rec*, int* = nullptr, Obs* = nullptr,
                                      SoftFailure<R>* = nullptr);
template <class R> MKO_HD Ray<R> collimated_ray(R a, R theta, R rho, R phi);        // OPL₀ = p₀·d₀
template <class R> MKO_HD Ray<R> point_source_ray(Vec3<R> q, R sin_max, R rho, R phi); // OPL₀ = 0
template <class R> MKO_HD TraceStatus propagate_surface(Ray<R>&, R t, SurfaceKind, R c, R n_before, R n_after,
                                                        const GhostPlateParams<R>* ghost = nullptr,
                                                        SoftFailure<R>* soft = nullptr,
                                                        unsigned char* branch = nullptr);  // M5-T1
template <class R, class T, class Obs = NoSurfaceObserver>
MKO_HD TraceStatus trace_ray(const T& sys, int l, Ray<R>&, TraceRecord<R>*, int* failed_surface = nullptr,
                             Obs* observer = nullptr,    // Beobachter: surface_extrema.hpp (M2-T3)
                             SoftFailure<R>* soft = nullptr);   // weiches Versagen (M4-T2)
template <class R, class T> MKO_HD TraceStatus transfer_to_image_plane(const T& sys, int l, Ray<R>&);
// surface_ops.hpp (M4-T2): Akkumulator und Statushilfen
template <class R> struct SoftFailure { R penalty; TraceStatus first; void add(R, TraceStatus); };
constexpr bool is_soft_failure(TraceStatus), in_piston(TraceStatus), is_hard_failure(TraceStatus);
enum class FailureMode : unsigned char { Soft = 0, Hard = 1 };
```

**Weiches Versagen (M4-T2):** Ein Zeiger auf `SoftFailure<R>` (Start {0, Ok}) schaltet die Fortsetzungsregeln aus [numerik.md, Abschnitt 11](numerik.md#11-weiches-versagen) ein; ohne ihn (nullptr, Standard) bleibt jede Funktion bitgleich zum harten Versagen. Der Zeiger reicht von `batch_ray_opl` über `trace_to_reference_sphere`, `trace_ray`, `propagate_surface` bis zu `intersect_sphere`, `refract_sphere`, `phase_surface_step` und `reference_sphere_step`; alle Parameter dieser Art sind nicht ableitbar (`detail::NonDeduced`), damit `nullptr` übergeben werden kann.

**Aufzeichnung (M5-T1):** `trace_segment` ist die einzige Schleife über die Flächen; `trace_ray` (mit `TraceRecord`), `trace_ray_recorded` (beliebiger Recorder) und das Neu-Vorwärtsrechnen aus Checkpoints (`replay_segment`) rufen sie auf. Ein Recorder ist jeder Typ mit `start(ray, at, soft)`, `surface(ray, at, branch, soft)` und `image(s_ref, branch, ray)`; `trace_to_reference_sphere` und `batch_ray_opl` nehmen ihn als letzten, optionalen Parameter (Standardtyp `NoTraceRecorder`). Die Aufrufe lesen nur den Strahlzustand, den der Trace ohnehin hält: Ergebnisse mit und ohne Aufzeichnung sind bitgleich (Abschnitt [Modus ADJOINT: Vorwärtsdurchlauf mit Aufzeichnung](#modus-adjoint-vorwärtsdurchlauf-mit-aufzeichnung-m5-t1)).

Erweiterungspunkte: Ein neuer Flächentyp bekommt einen Zweig in `propagate_surface` und seine Daten als zusätzliche Felder der Tabelle (so umgesetzt für die Geisterplatte, M4-T5); für den adjungierten Modus zusätzlich seine Zwischengrößen in `recompute_surface_step` und einen Zweig im Rückwärtsdurchlauf. Der adjungierte Modus (M5) liest Schnittpunkte und Richtungen aus der Aufzeichnung (`AdjointRecord` bzw. `CheckpointRecord`; die Richtung vor der Brechung an Fläche s ist `d[s − 1]`). Strahlhöhe und Einfallswinkel pro Fläche (M4-T3) liefert der Beobachter `RaySurfaceRecord` ohne `TraceRecord` (Abschnitt [Max-Reduktionen](#max-reduktionen-m2-t3)); dieselben Werte folgen aus `p[s]`, `d[s − 1]` und der Flächennormale (Test `SurfaceExtrema.ObserverMatchesTraceRecord`).

### Host-Schicht (Bibliothek `Mikroptim::cpu`, `src/cpu/`)

Statische Bibliothek `mikroptim_cpu`, linkt `Mikroptim::model` und `Mikroptim::glass`.

- `include/mikroptim/cpu/host_surface_table.hpp`: `HostSurfaceTable` (FP64-Vektoren `type`, `curvature`, `thickness`, `index`, Geisterfelder `ghost_coeff`, `ghost_index`, `ghost_norm_radius` mit G, K, zusätzlich `semi_diameter`; `view()` liefert die `SurfaceTableView<double>`), `make_system_index_table(SystemDescription, GlassCatalog)` (Brechzahlen der Dateiflächen und der Gläser der Geisterplatten) und `make_surface_table(SystemDescription, IndexTable)`, die die Geisterplatten als Flächen einfügt ([dateiformat.md](dateiformat.md#abbildung-dateiflächen--tabellenflächen)). Krümmungen von Stop, Plane und GhostPlate werden 0; mehr als `kMaxSurfaces` Tabellenflächen oder unpassende Tabellen werfen `std::invalid_argument`.
- `include/mikroptim/cpu/load_system.hpp`: `load_system(path)` verbindet `read_system_file`, `resolve_glass_catalogs` (relativ zum Verzeichnis der Datei), `GlassCatalog::from_file` + `merge`, `validate_materials`, `make_index_table` und `make_surface_table` zu einem `LoadedSystem { system, catalog_files, catalog, index, table, warnings }`. `prepare_system(description, base_dir)` macht dasselbe für eine bereits eingelesene Beschreibung.
- `include/mikroptim/cpu/cpu_trace.hpp`: `trace_system_cpu(system, table, quadrature)` traced alle Felder × Wellenlängen × Pupillenstützstellen (eine Quadratur für alle Felder) und liefert `CpuTraceResult` mit Endzustand (Schnittpunkt der letzten Fläche), Status und versagender Fläche pro Strahl, abgelegt unter `(f · L + l) · N + k` wie die Blöcke und Threads im Kernel. `source_ray(system, table, f, l, sample)` erzeugt den Startstrahl (kollimiert mit a = Halbdurchmesser der Blende, Punktquelle mit sin ϑ_max = NA / index[0][l]); `source_field_count` liefert die Zahl der Felder (1 bei der Punktquelle).

## Wellenfront (M1-T7)

Referenzsphäre, Kolbenabzug und RMS-Wellenfront pro Feld und Wellenlänge. Formeln, Wahl des Schnittpunkts, Versagensbehandlung, f_soll-Regel, Summationsreihenfolge und Einheiten: [numerik.md, Abschnitt 7](numerik.md#7-referenzsphäre-und-opd).

### Mathe-Kern (header-only, `MKO_HD`)

```cpp
// include/mikroptim/trace/reference_sphere.hpp
template <class R> MKO_HD TraceStatus reference_sphere_step(Ray<R>&, Vec3<R> center, R n_image,
                                                            R* s_ref = nullptr);   // Missed bei Versagen
template <class R, class T> MKO_HD TraceStatus trace_to_reference_sphere(const T& sys, int l, Ray<R>&,
                                                            Vec3<R> center, int* failed_surface = nullptr);
// include/mikroptim/merit/opd.hpp
template <class R> struct OpdStatistics { R valid_weight, piston, rms; int failed; };
template <class R> MKO_HD OpdStatistics<R> opd_statistics(const R* opl, const R* weight,
                                const TraceStatus* status, int count, R* opd /* oder nullptr */);
```

`trace_to_reference_sphere` ist der Pro-Strahl-Teil (Thread im Kernel): `trace_ray`, dann der Schritt auf die Sphäre um **M**_f im Medium index[S−1][l]; versagt die Referenzsphäre, ist `*failed_surface = S`. `opd_statistics` ist die sequenzielle Referenz der Block-Reduktion (Kolben, OPD, RMS über die Stützstellen eines Blocks); Der Kernel `RESIDUALS` (M2-T2, Abschnitt [Kernel RESIDUALS](#kernel-residuals-m2-t2)) baut sie als Baumreduktion nach.

### Host-API (Bibliothek `Mikroptim::merit`, `src/merit/`)

Statische Bibliothek `mikroptim_merit`, linkt `Mikroptim::cpu` und `Mikroptim::paraxial`. Die Wellenfrontauswertung gehört zum Modul Merit, weil M3/M4 aus denselben OPD-Werten die Residuen bilden; der CPU-Tracer bleibt reiner Strahltransport.

```cpp
struct WavefrontOptions { int rings = 10, spokes = 8; std::optional<PupilMode> pupil_mode; };
struct WavefrontMap {                        // ein Tupel (Feld f, Wellenlänge l)
    int field, wavelength; double wavelength_nm; Vec3<double> image_point;   // M_f
    PupilQuadrature<double> quadrature; std::vector<PupilSample<double>> pupil;
    std::vector<double> opl, opd, opd_waves; // [k]; NaN bei versagten Strahlen
    std::vector<TraceStatus> status; std::vector<int> failed_surface;       // S = Referenzsphäre
    double piston, rms, rms_waves, valid_weight; int failed;
    int samples() const; bool valid() const; // valid: mindestens ein gültiger Strahl
};
struct WavefrontResult { int fields, wavelengths; double reference_nm, focal_length;
                         std::vector<WavefrontMap> maps;          // maps[f · L + l]
                         const WavefrontMap& at(int f, int l) const; std::size_t failed_count() const; };

double target_focal_length(const SystemDescription&, const GlassCatalog&);   // f_soll
std::vector<Vec3<double>> ideal_image_points(const SystemDescription&, const HostSurfaceTable&,
                                             double focal_length);           // (0, f tan θ_f, w)
PupilMode wavefront_pupil_mode(const SystemDescription&, int field, const WavefrontOptions& = {});
WavefrontResult evaluate_wavefront(const SystemDescription&, const HostSurfaceTable&,
                                   std::span<const Vec3<double>> image_points, const WavefrontOptions& = {});
WavefrontResult evaluate_wavefront(const LoadedSystem&, const WavefrontOptions& = {});
```

Jedes Feld hat eine eigene Quadratur (Achsfächer auf der Achse, sonst halbe Pupille), daher die Ablage pro Tupel statt eines gemeinsamen Arrays wie in `CpuTraceResult`. Fehler: `std::invalid_argument` (Tabelle passt nicht, falsche Zahl von Bildpunkten, afokales System ohne `target.focal_length`, ungültige Quadratur), `std::out_of_range` (Feld), `GlassError` (Brechzahl bei `reference_nm`). Ausgabe auf der Kommandozeile: `mikroptim trace` ([cli.md](cli.md)).

## SystemBatch und Datenlayout

Ein Batch enthält V Varianten mit derselben Topologie (Flächenzahl und Flächentypen). Varianten mit anderer Topologie gehören in einen anderen Batch.

### Topologie und Felder (M2-T1)

Die Topologie `BatchTopology` umfasst S (Flächen, 2 … `kMaxSurfaces`), L (Wellenlängen), F (Felder), G (Geisterplatten), K (Koeffizienten pro Geisterplatte), die Quellenart (`SourceKind::Collimated` oder `Point`) und `type[s]`. Regeln (`validate_topology`, sonst `std::invalid_argument`): L, F ≥ 1; G = Anzahl der Flächen vom Typ `GhostPlate`; G > 0 ⇒ K ≥ 1, G = 0 ⇒ K = 0; Punktquelle ⇒ F = 1; `pupil_mode` leer oder mit F gültigen Einträgen.

**Pupillenart pro Feld** (`pupil_mode[f]`, ab M2-T2): eine Eigenschaft des Batches, nicht der Variante, weil sie Zahl und Ablage der Residuen festlegt (`ResidualLayout`). `topology_of` setzt sie beim Aufbau aus dem Ausgangssystem mit derselben Regel wie `wavefront_pupil_mode`: kollimiert θ_f = 0 → `Axis` (N Stützstellen), sonst `Half` (N·M); Punktquelle auf der Achse → `Axis`, mit x = 0 → `Half`, sonst `Full`. Leer heißt „nicht festgelegt“ (erlaubt für reine Speicherrechnungen; der Residuen-Kernel verlangt die Festlegung). Varianten dürfen ein Achsfeld nicht verlassen: `check_pupil_modes(batch, first, count)` verlangt bei `Axis` den Feldwinkel 0 in jeder Variante (bei der Punktquelle x = y = 0, bei `Half` x = 0); `Half` und `Full` gelten für jeden Winkel, auch 0 (eine Feld-Homotopie darf also bei θ = 0 beginnen, ohne die Ablage zu ändern). Geprüft wird in `validate_values`, beim Upload (`DeviceSystemBatch::upload`) und in `residuals_cpu`.

Structure of Arrays, alle Werte FP64 (außer `type` und `ghost_surface`), Variante v immer als äußerster Index:

| Feld | Typ | Ablage | Bedeutung |
|---|---|---|---|
| `type[s]` | uint8 | gemeinsam | Flächentyp `SurfaceKind`: Stop, Plane, Sphere, GhostPlate |
| `ghost_surface[g]` | int32 | gemeinsam | Fläche der Geisterplatte g (abgeleitet: g-te Fläche vom Typ GhostPlate) |
| `ghost_norm_radius[g]` | R | gemeinsam | Normierungsradius r_g des Dickenprofils [mm] |
| `wavelength_nm[l]` | R | gemeinsam | Wellenlängen, für die `index` berechnet wurde [nm] |
| `wavelength_weight[l]` | R | gemeinsam | Gewicht w_λ |
| `field_weight[f]` | R | gemeinsam | Gewicht w_f |
| `reference_wavelength_nm` | R | Skalar | λ_ref (OPD in Wellen, optimierung.md, Abschnitt 2) |
| `source_point[3]`, `source_na` | R | Skalar | nur Punktquelle: Quellpunkt (lokal zu Fläche 0), NA im Medium nach Fläche 0 |
| `curvature[v][s]` | R | pro Variante | Krümmung c = 1/R [1/mm], 0 bei Stop und Plane |
| `thickness[v][s]` | R | pro Variante | Abstand zum nächsten Scheitel [mm]; `thickness[v][S−1]` = Arbeitsabstand |
| `index[v][s][l]` | R | pro Variante | Brechzahl nach Fläche s bei Wellenlänge l (vorab aus dem Glasmodell) |
| `pupil_radius[v]` | R | pro Variante | Pupillenradius a = Halbdurchmesser der Blende [mm] |
| `field_angle[v][f]` | R | pro Variante | Feldwinkel θ_f [rad] (Punktquelle: 0) |
| `field_target[v][f]` | R | pro Variante | ideale Bildhöhe h_f = f_soll · tan θ_f [mm] (numerik.md, Abschnitt 7) |
| `ghost_coeff[v][g][k]` | R | pro Variante | Koeffizienten a_{g,k} der Geisterplatte g |
| `ghost_index[v][g][l]` | R | pro Variante | Brechzahl des virtuellen Glases der Geisterplatte |
| `paraxial_index[v][s]` | R | pro Variante | Brechzahl nach Fläche s bei λ_ref (Paraxialrechnung, ab M3-T3) |
| `paraxial_ghost_index[v][g]` | R | pro Variante | Brechzahl des Glases der Geisterplatte g bei λ_ref (ab M3-T3) |
| `buchdahl_coeff[v][s][j]` | FP64 | pro Variante, **nur Host** | Buchdahl-Koeffizienten (n₀, ν₁, ν₂, ν₃) des Mediums nach Fläche s (Glasvariablen, ab M6-T1) |

Indexformeln (`BatchShape`, header-only, `MKO_HD`; Element-, nicht Byteindizes):

| Feld | Index | Elemente pro Variante |
|---|---|---|
| `curvature`, `thickness` | v·S + s | S |
| `index` | (v·S + s)·L + l | S·L |
| `pupil_radius` | v | 1 |
| `field_angle`, `field_target` | v·F + f | F |
| `ghost_coeff` | (v·G + g)·K + k | G·K |
| `ghost_index` | (v·G + g)·L + l | G·L |
| `paraxial_index` | v·S + s (wie `curvature`) | S |
| `paraxial_ghost_index` | v·G + g (`ghost_at`) | G |
| `buchdahl_coeff` (nur Host) | (v·S + s)·4 + j | 4·S |

Die Daten einer Variante liegen in jedem Feld zusammenhängend; Variante v der Flächentabelle ist daher eine `SurfaceTableView` mit um v·S bzw. v·S·L versetzten Zeigern (`SystemBatchView::surface_table(v)`), `type` ist für alle Varianten derselbe Zeiger. `index` hat für eine Variante dasselbe Layout wie `IndexTable::n`.

Festlegungen:

- **Arbeitsabstand:** Ein eigenes Feld `image_distance[v]` gibt es nicht. Einzige Quelle ist `thickness[v][S−1]` (so lesen es `trace_ray`, `transfer_to_image_plane` und die Paraxialrechnung); `image_distance(v)` ist nur ein Zugriff darauf. Die Variable „Arbeitsabstand w“ (M3) ist dieses Element.
- **Pro Variante** liegt alles, was der Optimierer oder eine Homotopie für einzelne Läufe ändert: Krümmungen, Dicken (inklusive Arbeitsabstand), Brechzahlen (Glasvariablen M6), Geisterkoeffizienten (M4), Pupillenradius (NA-Homotopie), Feldwinkel und Feldziele (Feld-Homotopie).
- **Gemeinsam** liegt, was die Meritfunktion definiert und für alle Varianten gleich sein muss, damit Merits vergleichbar sind: Gewichte, Wellenlängen, λ_ref, Quellenart und Punktquelle, Normierungsradien. Eine Spektrum-Homotopie verschiebt die Wellenlängen aller Varianten eines Batches gemeinsam (neue `index`-Zeilen pro Variante).
- **Feldziel** beim Aufbau aus einem System: f_soll = `target.focal_length`, sonst die paraxiale Brennweite bei λ_ref; afokale Systeme (z. B. `planplatte_bk7.json`), θ_f = 0 und die Punktquelle haben h_f = 0.
- **Feldwinkel** werden mit demselben Ausdruck wie im CPU-Tracer umgerechnet (θ = Grad · (π/180)), damit CPU-Trace über Batch und Einzeltabelle bitgleich sind.
- **Geisterplatten** (M4-T5): `type` enthält die eingefügten Flächen vom Typ GhostPlate, G = deren Zahl, K = größte Ordnung (kürzere Profile mit 0 ergänzt, K ≤ `kMaxGhostTerms`, geprüft in `validate_topology`); `make_system_batch` füllt `ghost_norm_radius`, `ghost_coeff` und `ghost_index` aus der Flächentabelle. `surface_table(v)` setzt die Geisterzeiger auf `ghost_coeff + ghost_coeff_at(v, 0, 0)` und `ghost_index + ghost_index_at(v, 0, 0)` (gleiches Layout wie `SurfaceTableView`).
- Brechzahlen werden pro Variante und Wellenlänge auf dem Host (oder einem kleinen Kernel) berechnet; der Trace-Kernel rechnet keine Dispersionsformeln.
- **Glasvariablen** (M6-T1): `buchdahl_coeff[v][s][0 … 3]` = (n₀, ν₁, ν₂, ν₃) des Mediums nach Tabellenfläche s, wenn es ein Buchdahl-Material ist (`make_system_batch` füllt sie aus der Systemdatei), sonst 0. Das Feld ist die Quelle der Glasvariablen (`VariableSet::extract`), weil die Brechzahlen allein die Koordinaten nicht immer bestimmen (z. B. eine Wellenlänge und λ_ref). `VariableSet::apply` schreibt die Koeffizienten und rechnet daraus `index[v][s][l]` für alle Wellenlängen und `paraxial_index[v][s]` mit demselben Ausdruck wie `material_index` (Batch nach `apply` bitgleich zum Batch aus der geänderten Datei). Das Feld liegt nur im Host-Container: Die Kernel brauchen es nicht, `DeviceSystemBatch` lädt es nicht hoch, `download` lässt es unverändert, `view()` enthält es nicht; Kopieren, Teil-Batches, `validate_values` (endlich) und `operator==` berücksichtigen es. Kosten 32·S Byte pro Variante im Host-Speicher.
- **Brechzahlen bei λ_ref** (M3-T3): Die Paraxial-Nebenbedingungen (Brennweite bei λ_ref, numerik.md §9) brauchen Brechzahlen bei λ_ref, die nicht unter den Wellenlängen des Batches sein muss (Spektrum-Homotopie: z. B. nur 480 und 644 nm bei λ_ref = 546 nm). Deshalb hat der Batch zwei Spalten pro Variante: `paraxial_index[v][s]` (Medium nach Tabellenfläche s; bei Geisterplatten das Medium des Spalts) und `paraxial_ghost_index[v][g]`. Sie sind immer vorhanden (eine Regel statt „Spalte von λ_ref, falls vorhanden“) und gehören zu `index`/`ghost_index`: Wer die Brechzahlen einer Variante ändert (Glasvariablen M6), ändert beide. `make_system_batch` füllt sie aus `IndexTable::reference_n`/`reference_ghost_n` (`make_system_index_table`), sonst aus der Spalte von λ_ref; fehlt beides, wirft es. Die Trace-Kernel lesen sie nicht; sie liegen trotzdem im Gerätepuffer (Upload/Download bleiben verlustfrei, Kosten (S + G)·8 Byte pro Variante). `validate_values` verlangt nur Endlichkeit (von Hand gebaute Batches für reine Traces dürfen sie auf 0 lassen); die Paraxialrechnung prüft n > 0 selbst.

### Host-Container (Bibliothek `Mikroptim::batch`, `src/batch/`)

Statische Bibliothek `mikroptim_batch`, linkt `model`, `glass`, `cpu` und privat `paraxial`. Header unter `include/mikroptim/batch/`:

```cpp
// system_batch_view.hpp (header-only, MKO_HD, ohne std-Container)
struct BatchShape { int variants, surfaces, wavelengths, fields, ghost_plates, ghost_terms;
                    size_t surface_at(v, s), index_at(v, s, l), field_at(v, f),
                           ghost_coeff_at(v, g, k), ghost_index_at(v, g, l), ghost_at(v, g);
                    /* + *_stride() */ };
template <class R> struct SystemBatchView {          // trivial kopierbar, per Wert an Kernel
    BatchShape shape; SourceKind source; R source_point[3], source_na, reference_wavelength_nm;
    const SurfaceKind* type; const int* ghost_surface; const R *ghost_norm_radius, *wavelength_nm,
        *wavelength_weight, *field_weight, *curvature, *thickness, *index, *pupil_radius,
        *field_angle, *field_target, *ghost_coeff, *ghost_index, *paraxial_index,
        *paraxial_ghost_index;
    SurfaceTableView<R> surface_table(int v) const;  // Eingabe für trace_ray
    R image_distance(int v) const;                   // = thickness[v][S−1]
};
template <class R> Ray<R> batch_source_ray(const SystemBatchView<R>&, int v, int f, int l,
                                           const PupilSample<R>&);   // wie source_ray (cpu_trace)

// system_batch.hpp
void validate_topology(const BatchTopology&);
class SystemBatchHost {                              // besitzt alle Felder (std::vector<double>)
    explicit SystemBatchHost(BatchTopology, int variants = 0);   // alle Werte 0
    void resize(int); int add_variant_copy(int from); void copy_variant(int from, int to);
    void copy_variant_from(const SystemBatchHost& other, int from, int to); // gleiche Topologie
    void assign_variants(const SystemBatchHost& other, int first, int count); // Teil-Batch (M7-T3)
    double& curvature(int v, int s); /* … jedes Feld, jeder Index geprüft (std::out_of_range) */
    std::span<double> curvature_data(); /* … Rohfelder für Upload und schnelle Schleifen */
    SystemBatchView<double> view() const; SurfaceTableView<double> surface_table(int v) const;
    void validate_values() const;                    // endlich, n > 0, a > 0, Gewichte >= 0 …
    friend bool operator==(const SystemBatchHost&, const SystemBatchHost&); // bitweise
};
BatchTopology topology_of(const SystemDescription&, const IndexTable&);
SystemBatchHost make_system_batch(const SystemDescription&, const IndexTable&, int variants = 1,
                                  std::optional<double> focal_length = std::nullopt);
SystemBatchHost make_system_batch(const LoadedSystem&, int variants = 1);

// batch_trace_cpu.hpp: CPU-Referenz pro Variante, Ablage wie trace_system_cpu
CpuTraceResult trace_variant_cpu(const SystemBatchHost&, int v, const PupilQuadrature<double>&);
```

Für eine aus einem System gebaute Variante ist `trace_variant_cpu` bitgleich zu `trace_system_cpu` über dessen Einzeltabelle (Test `BatchTrace.*`).

### Gerätespeicher (`Mikroptim::cuda`)

`include/mikroptim/cuda/device_system_batch.hpp` (reines C++):

```cpp
class DeviceSystemBatch {                            // RAII, nur verschiebbar
    explicit DeviceSystemBatch(const SystemBatchHost&, int device = -1);
    void upload(const SystemBatchHost&, int first = 0, int count = -1); // Varianten-Bereich
    void download(SystemBatchHost&, int first = 0) const; SystemBatchHost download() const;
    void release() noexcept;
    SystemBatchView<double> view() const;            // Gerätezeiger
    int device() const; int variants() const; const SystemBatchLayout& layout() const;
};
int max_variants_on_device(const BatchTopology&, const OutputRequest&,
                           const MemoryReserve& = {}, size_t extra_bytes_per_variant = 0,
                           int device = -1);
// device_info.hpp
struct DeviceMemoryInfo { size_t free_bytes, total_bytes; };
DeviceMemoryInfo device_memory_info(int device = -1);   // cudaMemGetInfo
```

Alle Felder liegen in **einem** Puffer (ein `cudaMalloc`) im Layout `make_batch_layout`: erst die gemeinsamen, dann die Felder pro Variante, in der Reihenfolge der Tabelle oben, jedes auf `kBatchAlignment` = 256 Byte ausgerichtet. Die Skalare liegen nur in der Sicht. `upload` lädt einen Variantenbereich [first, first + count) hoch, der auf dem Gerät die Nummern 0 … count − 1 trägt (Tranchen in M2-T4, Aufteilung auf GPUs in M7), und verwendet den Puffer weiter, wenn er groß genug ist. Kopien sind synchron. Das Gerät wird beim ersten Upload festgelegt; alle Aufrufe stellen das aktuelle Gerät des Aufrufers wieder her.

Fehlerbehandlung: Konfigurations- und Speicherfehler sind Host-Fehler (AGENTS.md, Abschnitt 5) und werfen `mko::cuda::CudaError` (`include/mikroptim/cuda/cuda_error.hpp`, Makro `MKO_CUDA_CHECK` in `src/cuda/cuda_check.hpp`); falsche Argumente werfen `std::invalid_argument` bzw. `std::out_of_range`. Die älteren Prüfkernel (M1) melden Fehler weiterhin per `bool` (`MKO_CUDA_TRY`).

Der Prüfkernel `trace_batch_gpu` (`include/mikroptim/cuda/batch_trace_check.hpp`) traced jeden Strahl (v, f, l, k) mit `batch_source_ray` und `trace_ray` über `view.surface_table(v)` direkt aus dem globalen Speicher. Er prüft nur das Layout gegen `trace_variant_cpu`; Grid-Abbildung, Shared Memory und Reduktionen hat der Kernel `RESIDUALS` (unten).

Für df64 (M7-T1) bleiben Host-Container, Gerätepuffer und Sicht FP64; die Kernel wandeln die Daten eines Blocks beim Laden in den Rechentyp um ([Zahlentypen der Kernel](#zahlentypen-der-kernel-m7-t1)). Eine eigene Sicht `SystemBatchView<df64>` gibt es nicht.

## Brechzahltabelle

Modul Glasmodell (`include/mikroptim/glass/`, Bibliothek `Mikroptim::glass`):

- `dispersion.hpp` – header-only, `MKO_HD`, templatisiert über `R`: `sellmeier_index`, `buchdahl_omega`, `buchdahl_index` (λ in µm) und `nm_to_um` als einzige Umrechnungsstelle nm → µm.
- `catalog.hpp` – `GlassCatalog` (Host), Einlesen siehe [dateiformat.md](dateiformat.md#glasdaten).
- `index_table.hpp` – `material_index` und `make_index_table`.
- `buchdahl_fit.hpp` – Kleinste-Quadrate-Fit des Buchdahl-Modells (Verfahren: [numerik.md](numerik.md#10-glasmodelle)):

```cpp
struct BuchdahlFitOptions { double lambda_min_nm = 400, lambda_max_nm = 700; int samples = 301;
                            BuchdahlFitNodes nodes = Chebyshev; int check_samples = 3001; };
struct BuchdahlFit { int dispersion_terms; double n0; std::array<double, 3> nu;
                     double max_abs_error, rms_error; BuchdahlFitOptions options;
                     double index(double lambda_nm) const; };
BuchdahlFit fit_buchdahl(const Glass& glass, int dispersion_terms /* 2 oder 3 */,
                         const BuchdahlFitOptions& = {});
BuchdahlFit fit_buchdahl(const IndexFunction& index_nm, int dispersion_terms,
                         const BuchdahlFitOptions& = {}); // beliebiges n(lambda_nm)
MaterialSpec to_buchdahl_material(const BuchdahlFit& fit);   // kind = Buchdahl
```

```cpp
struct IndexTable { int surfaces; int wavelengths; std::vector<double> n; double at(int s, int l) const;
                    int ghost_plates; std::vector<double> ghost_n;              // M4-T5
                    double reference_nm; std::vector<double> reference_n,       // M3-T3: bei λ_ref
                                                             reference_ghost_n; };
IndexTable make_index_table(std::span<const MaterialSpec> media,
                            std::span<const double> wavelengths_nm, const GlassCatalog& catalog);
```

`make_system_index_table(system, catalog)` (`include/mikroptim/cpu/host_surface_table.hpp`) füllt zusätzlich `ghost_n` und die Brechzahlen bei `spectrum.reference_nm` (`reference_n[s]`, `reference_ghost_n[g]`, mit `material_index`, also bitgleich zur Spalte, wenn λ_ref im Spektrum liegt); liegt λ_ref außerhalb des Gültigkeitsbereichs eines Glases, wirft es `GlassError`. `make_index_table` lässt sie leer.

`media[s]` ist das Medium **nach** Fläche s (`mko::MaterialSpec` aus `include/mikroptim/model/material.hpp`), das Medium vor Fläche 0 ist Luft. Die Tabelle ist zeilenweise abgelegt: `n[s * wavelengths + l]`, also dasselbe Layout wie `index[v][s][l]` für eine Variante. Air → 1, `constant` → n, Katalogname → Sellmeier mit Prüfung von `range_nm`, `buchdahl` → Formel aus numerik.md, Abschnitt 10. Fehler (unbekanntes Glas, Wellenlänge außerhalb des Gültigkeitsbereichs) sind `GlassError` mit Fläche, Wellenlänge und Glasname.

## Paraxialoptik

Modul `include/mikroptim/paraxial/`, Bibliothek `mikroptim_paraxial` (Alias `Mikroptim::paraxial`, linkt `model` und `glass`). Definitionen und Vorzeichen: [numerik.md, Abschnitt 9](numerik.md#9-paraxialoptik).

- `ynu_trace.hpp` – header-only, `MKO_HD`, templatisiert über `R`: `ParaxialRay<R>{y, u}`, `surface_power`, `paraxial_refract`, `paraxial_transfer`, `paraxial_trace(c, t, n, count, ray, at_surface)` und `petzval_sum(c, n, count)` über nackte Arrays (Layout wie eine Variante im SystemBatch). Für Paraxialdaten vieler Varianten (M3-T3: `batch_paraxial` mit `paraxial_index`, M4-T4) auf Host oder GPU.
- `paraxial.hpp` – Host-API:

```cpp
struct ParaxialData {
    std::array<double, 4> abcd;          // Systemmatrix {A, B, C, D}, reduzierte Winkel
    double power; bool afocal;
    double focal_length, back_focal_distance, front_focal_distance;
    double front_principal_plane, rear_principal_plane, image_defocus;
    double petzval_sum, petzval_curvature, petzval_radius;
    bool exit_pupil_at_infinity;
    double exit_pupil_position, exit_pupil_magnification, exit_pupil_semi_diameter;
};
ParaxialData compute_paraxial(std::span<const double> curvatures, std::span<const double> thicknesses,
                              std::span<const double> indices, double stop_semi_diameter);
ParaxialData compute_paraxial(const SystemDescription&, const IndexTable&, int wavelength_index);
ParaxialData compute_paraxial(const SystemDescription&, const GlassCatalog&, double wavelength_nm);
ParaxialData compute_paraxial(const SystemDescription&, const GlassCatalog&);   // reference_nm
```

Fehler: `std::invalid_argument` (leere oder ungleich lange Arrays, nicht endliche Werte, n ≤ 0, Blendenradius ≤ 0, Brechzahltabelle passt nicht zum System), `std::out_of_range` (Wellenlängenindex), `GlassError` (Brechzahl nicht bestimmbar). Geisterplatten (M4-T5) gehen als dünne Elemente ein: Überladung mit den Spannen `power` (φ_G pro Fläche) und `petzval` (φ_G/(n_m n_G)); die Überladungen für Systembeschreibungen rechnen über die Tabellenflächen und berücksichtigen Geisterplatten automatisch ([numerik.md, Abschnitt 9](numerik.md#9-paraxialoptik)). `ParaxialData` bleibt unverändert.

## Geisterplatten (M4-T5)

Physik und Formeln: [numerik.md, Abschnitt 6](numerik.md#6-geisterplatte); Dateiformat und Abbildung Dateiflächen ↔ Tabellenflächen: [dateiformat.md](dateiformat.md#geisterplatten).

```cpp
// include/mikroptim/trace/ghost_plate.hpp (header-only, MKO_HD)
template <class R> struct GhostPlateParams { const R* coeff; int terms; R norm_radius; R n_glass; };
template <class R> void legendre(int k, R xi, R& p, R& dp);                     // P_k, P'_k (Rekursionen)
template <class R> struct GhostProfile { R t, dt_du; };
template <class R> GhostProfile<R> ghost_profile(const R* coeff, int K, R u);   // u = (r/r_g)²
template <class R> R ghost_profile_t2(const R* coeff, int K, R r_g);            // paraxial t₂
template <class R> R ghost_plate_power(const R* coeff, int K, R r_g, R n_G, R n_m);   // φ_G
template <class R> R ghost_plate_petzval(const R* coeff, int K, R r_g, R n_G, R n_m); // φ_G/(n_m n_G)
template <class R> TraceStatus phase_surface_step(Ray<R>&, R phase, R grad_x, R grad_y, R n, R n_prime);
template <class R> TraceStatus ghost_plate_step(Ray<R>&, const GhostPlateParams<R>&, R n, R n_prime);
// include/mikroptim/model/system.hpp
struct GhostPlate { int after_surface; double distance; MaterialSpec glass; double norm_radius;
                    std::vector<double> coefficients; };            // SystemDescription::ghost_plates
struct SurfaceMap { std::vector<int> table_of_file, file_of_table, ghost_of_table, table_of_ghost;
                    int file_surface(int table_surface) const; int table_surface(int file_surface) const; };
SurfaceMap make_surface_map(const SystemDescription&); int table_surface_count(const SystemDescription&);
// include/mikroptim/glass/index_table.hpp: IndexTable::ghost_plates, ghost_n[g·L + l], ghost_at(g, l)
```

`trace_ray` zählt die Flächen vom Typ GhostPlate in Tracerichtung mit; die g-te erhält `GhostPlateParams{ghost_coeff + g·K, K, ghost_norm_radius[g], ghost_index[g·L + l]}`, Flächen vom Typ GhostPlate ohne Daten (g ≥ G) melden `Unsupported`. Tabellen ohne Geisterplatten (G = 0, Nullzeiger als Standardwerte) funktionieren unverändert.

## Kernel-Abbildung

- **Grid:** ein Block pro Tupel (Variante v, Feld f, Wellenlänge l), eindimensional linearisiert: b = (v·F + f)·L + l, Indexrechnung in 64 Bit; V·F·L ≤ 2³¹ − 1 (`kMaxGridBlocks`; größere Batches teilt die TraceEngine in Tranchen, M2-T4).
- **Block:** `kResidualThreads` = 128 Threads (Konstante in `include/mikroptim/cuda/residuals.hpp`, zu tunen in M7). Thread t bearbeitet die Stützstellen k = t, t + 128, … < n_f in aufsteigender Reihenfolge. Bei der Standardquadratur (n_f = 80, auf der Achse 10) bleibt ein Teil der Threads leer; das ist ein Thema für M7.
- **Shared Memory:** Flächentabelle der Variante mit den Brechzahlen nur der Wellenlänge l des Blocks, Quadratur des Feldes, Teilsummen pro Warp, Kolben (Belegung unten, ca. 1,9 KB).
- **Pupillenstützstellen** werden im Thread aus dem Index berechnet (`sample`: Ring i, Speiche j, Gewicht).
- **Versagen:** Strahlen, die eine Fläche verfehlen, totalreflektieren oder rückwärts laufen, laufen maskiert weiter: kein früher Ausstieg aus Schleifen oder Reduktion, jeder Thread erreicht jedes `__syncthreads` und jeden Warp-Shuffle. Seit M4-T2 ist Versagen in RESIDUALS und MERIT standardmäßig weich: verfehlte Flächen, Totalreflexion und negative Radikanden an Geisterplatten werden fortgesetzt, der Strahl bleibt im Kolben und trägt ein Strafresiduum; `ResidualSettings::failure = FailureMode::Hard` schaltet auf das harte Versagen von M2 zurück ([numerik.md, Abschnitt 11](numerik.md#11-weiches-versagen)). Der Schalter ist ein Laufzeitflag des Kernels; im gültigen Pfad kostet er nur die Prüfungen auf ein weiches Ereignis pro Fläche.

## Kernel RESIDUALS (M2-T2)

Implementiert in `src/cuda/residuals.cu` (Kernel `residuals_kernel<R, kExtrema>`, templatisiert über den Rechentyp; instanziiert für `double` und `df64`, M7-T1), Host-API in `include/mikroptim/cuda/residuals.hpp` (reines C++), Ablage und CPU-Referenz in `include/mikroptim/batch/batch_residuals.hpp` (Bibliothek `Mikroptim::batch`), Pro-Strahl-Teil header-only in `include/mikroptim/merit/ray_residual.hpp`. Definition des Residuums: [numerik.md, Abschnitt 7](numerik.md#residuen-pro-strahl-m2-gewichte-m4-t1).

### Residuenlayout

Der Residuenvektor einer Variante ist die Konkatenation der Tupel (f, l), f-major, dann l, dann der Stützstellenindex k = i·M + j der Quadratur des Feldes. Achsfelder haben n_f = N, alle anderen n_f = N·M Einträge pro Wellenlänge:

offset(f, l) = L · Σ_{f′ < f} n_f′ + l · n_f,  R_v = L · Σ_f n_f = `rays_per_variant(F, L, N·M, F_Achse, N)`.

Die Ablage ist für alle Varianten eines Batches gleich, weil die Pupillenart pro Feld zur Topologie gehört (`BatchTopology::pupil_mode`, oben). Im weichen Modus (Standard, M4-T2) folgen auf die R_v Strahlresiduen P_v = R_v **Strafresiduen** in derselben Ablage (Strafresiduum des Strahls i in Zeile R_v + i, `ResidualLayout::penalty_rows`, im harten Modus P_v = 0); die Zeilen des Traces pro Variante sind `trace_rows` = R_v + P_v. Danach folgen optional E weitere Residuen pro Variante (Paraxial usw., ab M3; `ResidualRunOptions::extra_residuals`), Stride pro Variante R_v + P_v + E; der Lauf setzt sie auf 0. OPD (optional) und Status (1 Byte, `TraceStatus`) liegen ohne Straf- und weitere Residuen ab v·R_v. Der Speicher folgt `make_output_layout` mit `OutputRequest{residuals, ray_status, opd, penalty_residuals}`. Ein Strafresiduum pro Strahl (Summe der Strafgrößen seiner Ereignisse) statt eines pro Ereignis hält den Residuenvektor bei genau einem zusätzlichen Block fester Größe; FD-Jacobi und Normalgleichungen sehen ihn als R = R_v + P_v + E Zeilen.

```cpp
// include/mikroptim/batch/batch_residuals.hpp
struct ResidualSettings { int rings = 10, spokes = 8;
                          FailureMode failure = FailureMode::Soft;          // M4-T2
                          double fail_weight = kDefaultFailWeight; };      // w_fail = 10⁶ Wellen²
struct ResidualFieldPlan { int64_t offset; int samples; PupilMode mode; };   // [F], für den Kernel
class ResidualLayout {                              // aus Topologie + Einstellungen
    ResidualLayout(const BatchTopology&, const ResidualSettings& = {});   // wirft ohne pupil_mode
    int fields(), wavelengths(), axis_fields(); PupilMode pupil_mode(f); int samples(f);
    const PupilQuadrature<double>& quadrature(f); int64_t offset(f, l), rays_per_variant();
    bool soft_failure(); int64_t penalty_rows(), trace_rows();          // M4-T2
    std::span<const ResidualFieldPlan> plan();
};
struct ResidualResult { int variants; int64_t rays_per_variant, penalty_rows, stride;
    std::vector<double> residuals /*[V·stride]*/, opd /*[V·R_v] oder leer*/;
    std::vector<uint8_t> status /*[V·R_v]*/;
    double residual(v, i), penalty(v, i), opd_at(v, i); int64_t trace_rows();
    TraceStatus status_at(v, i); size_t failed_count(), hard_failed_count(); };
ResidualResult residuals_cpu(const SystemBatchHost&, int v, const ResidualLayout&); // CPU-Referenz
ResidualResult residuals_cpu(const SystemBatchHost&, const ResidualLayout&);        // alle Varianten
void check_residual_layout(const BatchTopology&, const ResidualLayout&);
void check_pupil_modes(const SystemBatchHost&, int first = 0, int count = -1);   // system_batch.hpp

// include/mikroptim/merit/ray_residual.hpp (header-only, MKO_HD, CPU und Kernel)
// Rechentyp R (Tabelle, Stützstelle) und Speichertyp B des Batches dürfen verschieden sein (M7-T1)
template <class B, class T> Vec3<R> batch_image_point(const SystemBatchView<B>&, const T& table, int v, int f);
template <class R, class B, class T> TraceStatus batch_ray_opl(const SystemBatchView<B>&, const T& table, int table_l,
                                                      int v, int f, int l, const PupilSample<R>&, Vec3<R> center, R* opl,
                                                      Observer* = nullptr, SoftFailure<R>* soft = nullptr);
template <class B> B residual_scale(const SystemBatchView<B>&, int f, int l);   // √(ŵ_f ŵ_λ)/λ_ref (M4-T1)
template <class R> R ray_residual(R scale, const PupilSample<R>&, R opd);       // scale·√(W)·OPD
// weiches Versagen (M4-T2)
template <class B> B penalty_scale(const SystemBatchView<B>&, int f, int l, B fail_weight); // √(w_fail ŵ_f ŵ_λ)
template <class R> R ray_penalty(R scale, const PupilSample<R>&, TraceStatus, R soft_penalty);
```

`residuals_cpu` ist zugleich der CPU-Pfad für eine Optimierung ohne GPU (CI). Sie rechnet mit denselben Funktionen wie der Kernel (`batch_ray_opl`, `ray_residual`, `residual_scale`, `ray_penalty`, `penalty_scale`) und dem sequenziellen `opd_statistics`; im harten Modus ist ihre OPD für eine aus einem System gebaute Variante bitgleich zu `evaluate_wavefront` (Test `ResidualsCpu.MatchesEvaluateWavefront`), im weichen Modus ohne Versagen ebenfalls (Strafresiduen 0).

### Ablauf im Block

1. **Laden:** Threads s = t, t + 128, … kopieren `type[s]`, `curvature[v][s]`, `thickness[v][s]` und `index[v][s][l]` in den Shared Memory, die ersten N Threads die Knoten u_i und Gewichte W_i, Thread 0 Ringzahl, Speichenzahl (1 auf der Achse) und Pupillenart aus `ResidualFieldPlan[f]`; danach `__syncthreads`. Die Sicht auf die Tabelle hat `wavelengths = 1` (Wellenlängenindex 0) und liefert dieselben Zahlen wie `surface_table(v)` auf der CPU.
2. **Durchlauf 1:** pro Stützstelle `batch_ray_opl` (Startstrahl `batch_source_ray`, `trace_to_reference_sphere` um **M**_f = (0, `field_target[v][f]`, `thickness[v][S−1]`), im weichen Modus mit einem `SoftFailure<R>` im Register); die OPL wird vorläufig auf den Residuenplatz geschrieben, der Status in das Statusfeld, im weichen Modus das Strafresiduum p_k = `ray_penalty(penalty_scale(f, l, w_fail), sample_k, Status, Strafe)` direkt auf seinen Platz (es hängt nicht vom Kolben ab). Jeder Thread summiert (W_k, W_k·OPL_k) seiner Strahlen im Kolben (`in_piston`: Ok oder weich versagt) in aufsteigender Reihenfolge.
3. **Reduktion:** Baumreduktion im Warp mit `__shfl_down_sync` (Abstände 16, 8, 4, 2, 1, volle Maske), Lane 0 schreibt die Warp-Summe in den Shared Memory, `__syncthreads`, Thread 0 summiert die 4 Warp-Summen in fester Reihenfolge und schreibt den Kolben OPL̄ = Σ W·OPL / W_ok und ein Gültigkeitsflag, `__syncthreads`. Keine Gleitkomma-Atomics (ADR-0007): Zwei Läufe mit gleichen Eingaben sind auf derselben GPU bitgleich (Test `CudaResiduals.RepeatedRunsAreBitIdentical`, 10 Läufe). Die Summationsreihenfolge hängt von `kResidualThreads` ab, nicht von der Batchgröße oder der Lage der Variante im Batch.
4. **Durchlauf 2:** derselbe Thread liest OPL und Status seiner Stützstellen zurück und schreibt OPD_k = OPL_k − OPL̄ (0 bei hartem Versagen) und r_k = `ray_residual(residual_scale(f, l), sample_k, OPD_k)`.

**Tupel ohne Strahl im Kolben** (W_ok = 0, Kolben undefiniert): alle OPD und Wellenfrontresiduen des Tupels sind 0, Status und Strafresiduen bleiben gesetzt (gleiche Regel wie `opd_statistics`).

**Shared Memory** (statisch, unabhängig von S bis `kMaxSurfaces` = 64), für R = double: Typ 64 B + Krümmung, Dicke, Brechzahl 3 · 64 · 8 B = 1.536 B + Quadratur 272 B + Teilsummen 2 · 4 · 8 B + Kolben und Flag 12 B, zusammen ca. 1,9 KB. Dazu **dynamisch** die Geisterplatten der Variante (M4-T5): Koeffizienten G·K, Brechzahlen der Gläser bei der Wellenlänge des Blocks G und Normierungsradien G, also (G·K + 2G)·8 B, höchstens (64·32 + 128)·8 B = 17 KiB; ohne Geisterplatten 0 B. Zusammen ≤ 19 KB < 48 KB auf beiden Karten (kein Opt-in; `static_assert` und Prüfung beim Start). Geladen werden sie in Schritt 1 zusammen mit der Flächentabelle; die Sicht im Shared Memory hat dieselben Geisterfelder wie `surface_table(v)` mit `wavelengths = 1`.

**Abweichung CPU/GPU:** Die Pro-Strahl-Rechnung ist dieselbe Funktion; Unterschiede kommen aus FMA-Kontraktion, `sin`/`cos` der CUDA-Bibliothek und der Summationsreihenfolge des Kolbens (≈ 10⁻¹⁶·OPL); der Faktor `residual_scale` ist bitgleich. Für je 1.000 gestörte Varianten der Testsysteme liegen alle Strahlen innerhalb `kCpuGpuFp64` = 10⁻¹² mm (OPD; Residuen r_k = u_k·OPD_k mit u_k = s_{f,l}√(W_i V_j) entsprechend innerhalb u_k·10⁻¹², Testhilfe `residual_units`), außer einzelnen schlecht konditionierten Strahlen (nahe Totalreflexion oder streifend, |OPD| von mm), die relativ ≤ 1,3·10⁻¹² abweichen (`kCpuGpuFp64Relative`, Test `CudaResiduals.MatchCpuForPerturbedVariants`). Im weichen Modus (M4-T2) gelten für Tupel mit weich versagten Strahlen absolut `kCpuGpuSoftTuple` = 10⁻⁹ mm (gemessen ≤ 1,35·10⁻¹⁰ mm: die fortgesetzten Strahlen laufen weit außerhalb des Systems und wirken über den Kolben auf das Tupel) und für die Strafgrößen relativ `kCpuGpuPenaltyRelative` = 10⁻⁶ (gemessen ≤ 5,6·10⁻¹⁰, ein Strahl tief im Versagensbereich 7·10⁻⁸).

### Host-API

```cpp
// include/mikroptim/cuda/residuals.hpp (reines C++)
inline constexpr int kResidualThreads = 128;
struct ResidualRunOptions { bool opd = false; int64_t extra_residuals = 0;
                            bool extrema = false;      // Max-Reduktionen (M2-T3)
                            Precision precision = Precision::Fp64; };   // Rechentyp (M7-T1)
class DeviceResiduals {                  // Ausgabepuffer auf dem Gerät, RAII, nur verschiebbar
    void run(const DeviceSystemBatch&, const ResidualLayout&, const ResidualRunOptions& = {});
    ResidualResult download() const; void download(ResidualResult&) const; void release() noexcept;
    const double* device_residuals() const; const uint8_t* device_status() const;
    const double* device_opd() const;     // für Folgekernel (M3), gültig bis zum nächsten run
    const double* device_extrema() const; // [V][S][2], nur mit extrema (M2-T3)
    int64_t rays_per_variant(), penalty_rows(), stride();   // Ablage der Ausgabe (M4-T2)
};
ResidualResult run_residuals(const DeviceSystemBatch&, const ResidualLayout&,
                             const ResidualRunOptions& = {});   // ein Lauf mit Download
```

`run` legt einen Puffer an (ein `cudaMalloc`, Layout `make_output_layout` plus `ResidualFieldPlan[F]`) oder verwendet ihn weiter, lädt die Feldablage hoch, startet den Kernel auf dem Gerät des Batches, prüft `cudaGetLastError` und wartet auf das Ende (`cudaDeviceSynchronize`; Streams und Überlappung sind Sache von M2-T4/M7). Fehler: `CudaError` (CUDA), `std::invalid_argument` (Layout passt nicht, zu großes Grid, negative Zahl weiterer Residuen), `std::logic_error` (leerer Batch, Download ohne Lauf). Die Werte der Varianten prüft der Aufrufer auf dem Host (`validate_values`; `check_pupil_modes` läuft beim Upload). Die [TraceEngine](#host-api-traceengine-m2-t4) (M2-T4) kapselt Gerätewahl, Prüfung, Tranchen und diese Klasse; Programme benutzen nur sie.

**Gewichte (M4-T1):** `residual_scale(view, f, l)` = √(ŵ_f·ŵ_λ)/λ_ref mit auf Summe 1 normierten Gewichten (CPU und Kernel gemeinsam, einmal pro Block in den Rechentyp gewandelt); die Residuen sind damit in Wellen bei λ_ref, die OPD-Ausgabe bleibt in mm ([numerik.md, Abschnitt 7](numerik.md#residuen-pro-strahl-m2-gewichte-m4-t1)). **Weiches Versagen (M4-T2):** Kernelparameter `soft` und `fail_weight` aus `ResidualLayout::settings()`; die Maske in den Durchläufen 1 und 2 ist `in_piston(Status)` statt `Status == Ok`, die Strafresiduen entstehen in Durchlauf 1. Der Kernel `MERIT` (M2-T3, unten) bildet aus denselben Größen die Quadratsumme pro Block; Laden, Kolben und Reduktionen teilen sich beide Kernel in `src/cuda/wavefront_block.hpp`. Mit `extrema` liefert auch `RESIDUALS` die [Max-Reduktionen](#max-reduktionen-m2-t3) (`ResidualResult::extrema`, bitgleich zu denen aus `MERIT`).

**Erste Durchsatzzahl** (RTX 3090, FP64, Test `CudaResiduals.LargeBatch`, kein Benchmark): 10.000 Varianten von `testsystem_mehrlinser.json` (S = 8, 680 Strahlen pro Variante) in ca. 26 ms für den Kernel allein, also ca. 2,6·10⁸ Strahlen/s; über die TraceEngine mit Upload und Download (seit M2-T4) ca. 33 ms, 2,1·10⁸ Strahlen/s. Messungen über Flächenzahl und Batchgröße: [benchmarks.md](benchmarks.md) (M2-T6, `bench/trace_throughput`).

## Kernel MERIT (M2-T3)

Implementiert in `src/cuda/merit.cu` (Kernel `merit_kernel<R, kExtrema>` und `merit_finalize_kernel<R>`), Host-API in `include/mikroptim/cuda/merit.hpp` (reines C++), Ergebnis und CPU-Referenz in `include/mikroptim/batch/batch_merit.hpp` (Bibliothek `Mikroptim::batch`). Programme benutzen den Modus über die TraceEngine (`TraceMode::Merit`).

**Definition.** Pro Variante v

F_v = Σ_k r_k² + Σ_k p_k² (+ E_v),

über alle R_v Strahlresiduen in der Ablage von `ResidualLayout`, mit denselben r_k wie im Kernel `RESIDUALS` (r_k = s_{f,l}·√(W_i V_j)·OPD_k; hart versagte Strahlen r_k = 0, Tupel ohne Strahl im Kolben alle r_k = 0) und im weichen Modus (M4-T2) den Strafresiduen p_k. E_v ist der optionale Anteil weiterer Residuen (Paraxial usw.), den der Aufrufer pro Variante liefert (`MeritRunOptions::extra_merit` bzw. `TraceSettings::extra_merit`); der Kernel berechnet ihn nicht. Zusätzlich `failed[v]` = Zahl der Strahlen mit Status ≠ Ok (weich und hart) und `hard_failed[v]` = Zahl der hart versagten (seit M4-T2; im harten Modus gleich `failed[v]`). Im harten Modus hat **eine Variante ohne gültigen Strahl F_v = 0 (+ E_v)** und ist nur an `failed[v] = R_v` erkennbar; im weichen Modus tragen versagte Strahlen Strafen.

**Ablauf im Block** (b = (v·F + f)·L + l, 128 Threads, wie RESIDUALS):

1. Laden der Flächentabelle und Quadratur (`load_block_table`, gemeinsam mit RESIDUALS).
   Geisterplatten (M4-T5) lädt `load_block_ghosts` wie in RESIDUALS an den Anfang des dynamischen Shared Memory ((G·K + 2G)·8 Byte, höchstens 17 KiB).
2. Durchlauf 1: `batch_ray_opl` pro Stützstelle; OPL und Status in den **Shared Memory** (dynamisch hinter den Geisterdaten, n_max·9 Byte, n_max = größte Stützstellenzahl eines Feldes), Teilsummen (W_k, W_k·OPL_k) der gültigen Strahlen in aufsteigender Reihenfolge der eigenen Stützstellen.
3. Kolben per `block_piston` – dieselbe Funktion, Threadzuordnung und Reihenfolge wie in RESIDUALS. Daher sind Kolben, OPD und r_k in beiden Kernels **bitgleich**; MERIT braucht keinen Residuenvektor im globalen Speicher.
4. Durchlauf 2: r_k aus der OPL im Shared Memory, Teilsumme Σ r_k² pro Thread in aufsteigender Reihenfolge, Zähler versagter und hart versagter Strahlen. Im weichen Modus hat jeder Thread schon in Durchlauf 1 Σ p_k² seiner Stützstellen gebildet (in aufsteigender Reihenfolge, ohne zusätzlichen Shared Memory) und addiert sie zuletzt zu seiner Teilsumme.
5. Blocksumme `block_sum`: Baumreduktion im Warp mit `__shfl_down_sync` (Abstände 16, 8, 4, 2, 1, volle Maske), Thread 0 summiert die 4 Warp-Summen in fester Reihenfolge; Ablage `merit_partial[b]`, `failed_rays_partial[b]` und `hard_failed_rays_partial[b]` (int32).
6. **Zweite Stufe** (`merit_finalize_kernel`, ein Thread pro Variante): F_v = ((Σ_{b = f·L + l} partial in der Reihenfolge f, dann l) + E_v), `failed[v]` und `hard_failed[v]` als int64.

Keine Gleitkomma-Atomics (ADR-0007). Die Summationsreihenfolge hängt nur von `kMeritThreads` = `kResidualThreads` = 128, F und L ab, nicht von der Batchgröße, der Lage der Variante oder der Aufteilung in Tranchen: zehn Läufe sind bitgleich (Test `CudaMerit.RepeatedRunsAreBitIdentical`), ebenso dieselben Varianten an anderer Stelle eines Batches und über mehrere Tranchen (`TraceEngine.MeritTranchesMatchSingleRun`).

**Vergleich mit Σ RESIDUALS²** (Abnahme, Test `CudaMerit.MatchesSumOfResiduals`): Der Test lädt die RESIDUALS-Ausgabe herunter und summiert r_k² pro Variante auf dem Host **kompensiert (Neumaier) in der Ablagereihenfolge (f, l, k)**, also nahezu exakt; die Abweichung ist allein der Rundungsfehler der Baumsumme (Schranke ≈ (⌈n/128⌉ + 7 + F·L)·ε). Gemessen für je 1.000 gestörte Varianten von `einzellinse_bk7.json` und `testsystem_mehrlinser.json` (N × M = 10 × 8 und 6 × 5): ≤ 4,1·10⁻¹⁶ relativ; mit Geisterplatten (`einzellinse_geisterplatte.json`, Koeffizienten und Gläser gestört, Test `CudaMerit.GhostPlates`) ≤ 3,7·10⁻¹⁶ (Toleranz `kMeritVsResidualSum` = 10⁻¹³). Zur Information rechnet der Test zusätzlich die Kernel-Reihenfolge auf dem Host nach (`std::fma` pro Thread, Warp-Baum, Warps, Tupel): 2.999 von 3.000 Varianten sind dazu bitgleich.

**CPU-Referenz** `merit_cpu(batch, v, layout, extrema)`: F_v = Σ_{(f,l)} (Σ_k r_k²) aus `residuals_cpu`, innen aufsteigend in k, außen in der Tupelreihenfolge wie die zweite Stufe; ohne E_v. GPU gegen CPU: |√F_GPU − √F_CPU| ≤ ‖r_GPU − r_CPU‖₂, geprüft gegen die 2-Norm der Toleranzen pro Strahl aus M2-T2 (max(`kCpuGpuFp64`, `kCpuGpuFp64Relative`·|r|)); gemessen ≤ 3,5·10⁻¹⁴ mm (1.000 Varianten), 1,2·10⁻¹² mm (10.000 Varianten mit einzelnen schlecht konditionierten Strahlen).

**Shared Memory** (R = double): statisch 2,0 KB (Tabelle, Quadratur, Kolben, Warp-Summen), mit Max-Reduktionen 6,1 KB, plus dynamisch Geisterdaten (G·K + 2G)·8 Byte und n_max·9 Byte. Höchstens 48 KB pro Block (`kMeritMaxSharedBytes`, ohne Opt-in auf beiden Karten): ohne Geisterplatten n_max ≤ 5.239 ohne bzw. 4.784 mit Max-Reduktionen, mit Geisterplatten entsprechend weniger (`merit_max_samples(extrema, G, K)`, aus den Kernelattributen; im ungünstigsten Fall G = 64, K = 32 mit Max-Reduktionen noch ca. 2.800); die Standardquadratur hat 80. Größere Quadraturen weist `DeviceMerit::run` bzw. die TraceEngine mit `std::invalid_argument` ab.

**Register und Local Memory** (`cuobjdump --dump-resource-usage`, CUDA 12.9):

| Kernel | sm_60 | sm_86 |
|---|---|---|
| `merit_kernel<double, false>` | 110 Register, Stack 72 B | 110 Register, Stack 72 B |
| `merit_kernel<double, true>` | 105 Register, Stack 2.120 B | 112 Register, Stack 2.120 B |
| `residuals_kernel<double, false>` | 118 Register, Stack 72 B (wie main nach M4-T5) | 110 Register, Stack 72 B (main nach M4-T5: 96 Register, Stack 104 B) |
| `residuals_kernel<double, true>` | 116 Register, Stack 2.120 B | 96 Register, Stack 2.144 B |

Ohne Max-Reduktionen kostet die Erweiterung nichts Wesentliches (sm_60 identisch zu RESIDUALS auf main nach M4-T5; auf sm_86 verteilt ptxas anders: mehr Register, weniger Stack). Der Registerbedarf stammt überwiegend aus dem Zweig der Geisterplatte (M4-T5) im Trace. Mit Max-Reduktionen liegen `RaySurfaceRecord` und `SurfaceExtrema` (je 2·`kMaxSurfaces`·8 = 1 KB) im Local Memory; die Laufzeitreserve dafür beträgt etwa 2 KB × 2.048 Threads × SM-Zahl (RTX 3090: ca. 350 MB, P100: ca. 240 MB) und ist in der Standardreserve (≥ 5 % des freien Speichers) enthalten. Kleinere Felder für kleine S sind ein Thema für M7.

**Durchsatz** (RTX 3090, FP64, Test `CudaMerit.LargeBatch`, über die TraceEngine mit Upload und Download, kein Benchmark): 10.000 Varianten von `testsystem_mehrlinser.json` (680 Strahlen): MERIT ca. 29 ms (2,3·10⁸ Strahlen/s), MERIT mit Max-Reduktionen ca. 38 ms (1,8·10⁸ Strahlen/s), RESIDUALS ca. 36 ms (1,9·10⁸ Strahlen/s). Der Kernel allein ist in beiden Modi gleich schnell (ca. 24 ms; die FP64-Rechnung pro Strahl dominiert); MERIT spart den Download der Residuen (54 MB) und vor allem Gerätespeicher: ca. 0,6 kB statt 6,5 kB pro Variante beim Mehrlinser (ohne Max-Reduktionen; mit ihnen ca. 2,3 kB wegen der Teilergebnisse pro Block und Fläche), also entsprechend größere Tranchen für Liniensuche und Multistart-Screening.

### Host-API

```cpp
// include/mikroptim/cuda/merit.hpp (reines C++)
inline constexpr int kMeritThreads = kResidualThreads;
inline constexpr size_t kMeritMaxSharedBytes = 48 * 1024;
struct MeritRunOptions { bool extrema = false; std::span<const double> extra_merit; /* leer oder [V] */
                         Precision precision = Precision::Fp64; };   // Rechentyp (M7-T1)
class DeviceMerit {                      // Ausgabepuffer auf dem Gerät, RAII, nur verschiebbar
    void run(const DeviceSystemBatch&, const ResidualLayout&, const MeritRunOptions& = {});
    MeritResult download() const; void download(MeritResult&) const; void release() noexcept;
    const double* device_merit() const; const int64_t* device_failed() const;
    const int64_t* device_hard_failed() const;   // M4-T2
    const double* device_extrema() const; // [V][S][2] oder nullptr
};
MeritResult run_merit(const DeviceSystemBatch&, const ResidualLayout&, const MeritRunOptions& = {});
int merit_max_samples(bool extrema, int ghost_plates = 0, int ghost_terms = 0,
                      Precision precision = Precision::Fp64);

// include/mikroptim/batch/batch_merit.hpp, batch_extrema.hpp (Host, auch ohne CUDA)
struct MeritResult { int variants; int64_t rays_per_variant; std::vector<double> merit;  // [V]
                     std::vector<int64_t> failed, hard_failed; SurfaceExtremaResult extrema; };
struct SurfaceExtremaResult { int variants, surfaces; std::vector<double> values;   // [V][S][2]
    double max_height(v, s), min_cos_incidence(v, s), max_incidence_angle(v, s); };
MeritResult merit_cpu(const SystemBatchHost&, int v, const ResidualLayout&, bool extrema = true);
MeritResult merit_cpu(const SystemBatchHost&, const ResidualLayout&, bool extrema = true);
SurfaceExtremaResult surface_extrema_cpu(const SystemBatchHost&, int v, const ResidualLayout&);
SurfaceExtremaResult surface_extrema_cpu(const SystemBatchHost&, const ResidualLayout&);
void check_residual_variant(const SystemBatchHost&, int v, const ResidualLayout&);
```

`run` legt einen Puffer an (Layout `make_output_layout` mit `merit` und optional `surface_max`, plus `ResidualFieldPlan[F]` und `extra_merit[V]`) oder verwendet ihn weiter, startet Block-Kernel, zweite Stufe und gegebenenfalls die zweite Stufe der Max-Reduktionen, prüft nach jedem Start `cudaGetLastError` und wartet auf das Ende. Fehler: `CudaError`, `std::invalid_argument` (Layout, Grid, Länge von `extra_merit`, zu viele Stützstellen), `std::logic_error` (leerer Batch, Download ohne Lauf).

### Max-Reduktionen (M2-T3)

Pro Variante v und Fläche s über alle **gültigen** Strahlen (Status Ok, einschließlich Referenzsphäre) aller Felder, Wellenlängen und Stützstellen; Ablage `[(v·S + s)·2 + {0, 1}]`:

- {0} **größte Strahlhöhe** h = √(x² + y²) [mm] im Schnittpunkt mit Fläche s (lokale Koordinaten),
- {1} **kleinster cos I** = **d**·**N**, **d** = Richtung vor der Brechung, **N** = (−c x, −c y, 1 − c z) die Einheitsnormale (numerik.md, Abschnitt 5), also der größte Einfallswinkel I = arccos(cos I) (`max_incidence_angle` begrenzt cos I auf [−1, 1]).

**Flächen sind Tabellenflächen** (wie `failed_surface` seit M4-T5): Dateiflächen plus Geisterplatten in Tracerichtung, s = 0 … S − 1 der `SurfaceTableView`; die Zuordnung zu Dateiflächen liefert `SurfaceMap` (`make_surface_map`, Modell). Eine Geisterplatte ist eine Ebene (c = 0): Höhe im Durchstoßpunkt, cos I = d_z vor dem Phasenschritt. Fläche 0 (Blende, keine Brechung): Startpunkt in ihrer Ebene und cos I = d_z des Startstrahls (c = 0). Versagte Strahlen tragen nichts bei, auch nicht zu den Flächen vor der versagenden (nur Strahlen mit Status Ok, in beiden Modi des Strahlversagens; weich versagte Strahlen sind fortgesetzt und laufen nicht durch das System). Ohne gültigen Strahl bleiben die neutralen Elemente h = 0 und cos I = 1. Gespeichert wird cos I statt des Winkels: die Reduktion bleibt exakt und ohne `acos` im Kernel, und eine Nebenbedingung „I ≤ I_max“ lautet cos I ≥ cos I_max (M4-T3).

**Pro-Strahl-Teil** (`include/mikroptim/trace/surface_extrema.hpp`, header-only, `MKO_HD`): `trace_ray`, `trace_to_reference_sphere` und `batch_ray_opl` nehmen einen optionalen Beobachter (Template-Parameter, Standard `NoSurfaceObserver`; ohne Beobachter entfällt der Aufruf zur Übersetzungszeit). Er wird nach jeder passierten Fläche mit (s, wirksame Krümmung `surface_curvature(kind, c)`, Richtung vor der Brechung, Strahl im Schnittpunkt) aufgerufen, für s = 0 mit dem Startzustand. `RaySurfaceRecord<R>` legt pro Strahl x² + y² und cos I ab; erst nach dem Status Ok übernimmt `SurfaceExtrema<R>::add_ray` die Werte (Maximum der Höhenquadrate, Minimum von cos I mit `extremum_max`/`extremum_min`). Dieselben Bausteine nutzen später M4-T3 (mechanische Nebenbedingungen) und M5 (eigener Beobachter für die Adjungierte).

**Reduktion:** pro Thread `SurfaceExtrema`, im Block pro Fläche `warp_max`/`warp_min` mit `__shfl_down_sync` (gleiche Struktur wie die Summen), Lane 0 in den Shared Memory [Warp][Fläche][2], Threads s < S kombinieren die 4 Warps in fester Reihenfolge und schreiben `surface_max_partial[b][s][2]` (Höhe²). Zweite Stufe `extrema_finalize_kernel` (ein Thread pro (v, s)): Maximum/Minimum über die Blöcke in der Reihenfolge (f, l), dann √ der Höhe (√ ist monoton und korrekt gerundet, also gleich dem Maximum der Wurzeln). Max und Min sind exakt und reihenfolgeunabhängig; die feste Reihenfolge und der Verzicht auf einen `atomicMax`-Trick halten trotzdem dasselbe Schema wie die Summen. Beide Modi (`MERIT`, `RESIDUALS`) liefern bitgleiche Max-Reduktionen.

**CPU-Referenz** `surface_extrema_cpu` (gleiche Bausteine über `batch_ray_opl`). GPU gegen CPU: max ist 1-Lipschitz, also gelten die Toleranzen pro Strahl: Höhe ≤ max(`kCpuGpuFp64`, `kCpuGpuFp64Relative`·h), gemessen ≤ 9,9·10⁻¹³ mm; cos I ≤ `kCpuGpuFp64`, gemessen ≤ 6·10⁻¹⁴. Geschlossene Formeln (Tests `SurfaceExtrema.CollimatedBeamMatchesClosedForm`, `CudaMerit.FailedRaysAndClosedForm`): achsparalleles Bündel der Einzellinse, h = a·ρ_N an Blende und erster Fläche, cos I = 1 bzw. √(1 − (c h)²); mit Feldwinkeln bis 2° cos I = cos 2° an der Blende.

## Zahlentypen der Kernel (M7-T1)

Die Kernel `RESIDUALS` und `MERIT` sind über den Rechentyp `R` templatisiert und für `double` und `df64` (Paar aus zwei float, [numerik.md, „Zahlentyp df64“](numerik.md#zahlentyp-df64-m7-t1)) instanziiert. Gewählt wird pro Lauf:

```cpp
// include/mikroptim/cuda/precision.hpp (reines C++)
enum class Precision { Fp64 = 0, Df64 = 1 };
const char* to_string(Precision);          // "FP64", "DF64"
void check_precision(Precision);           // std::invalid_argument bei unbekanntem Wert
// TraceSettings::precision, ResidualRunOptions::precision, MeritRunOptions::precision,
// merit_max_samples(extrema, G, K, precision); Standard überall Precision::Fp64
```

**Datenfluss.** Batch im Gerätespeicher, Kernelparameter (`SystemBatchView<double>`, Quadratur) und alle Ausgaben (Residuen, OPD, Status, Merit, Teilsummen, Max-Reduktionen) bleiben FP64; der Rechentyp ist ein reines Implementierungsdetail des Kernels. `load_block_table<R>` und `load_block_ghosts<R>` wandeln Flächentabelle, Quadraturknoten und Geisterdaten beim Laden in den Shared Memory in `R` um, `batch_source_ray` wandelt Pupillenradius und Feldwinkel des Strahls (`R(…)`), `residual_scale` liefert FP64 und wird einmal pro Block gewandelt. Trace, Referenzsphäre, Kolben (Baumreduktion in `R`), OPD, Residuen und die Quadratsumme laufen in `R`; geschrieben wird `static_cast<double>(…)`. Im Kernel `RESIDUALS` liegt die vorläufige OPL zwischen den Durchläufen im Rechentyp auf dem Residuenplatz (`sizeof(df64) == sizeof(double)`), ohne Rundung nach double. Die zweite Stufe von `MERIT` und der Max-Reduktionen summiert bzw. vergleicht die FP64-Teilergebnisse wie bisher. Für `R = double` sind alle Umwandlungen ohne Wirkung: der FP64-Pfad ist unverändert (gleiche Register, gleiche Ergebnisse, alle bisherigen Tests unverändert grün).

**Reduktionen.** Warp-Summen und -Extrema rufen `shfl_down_sync`: für eingebaute Typen `__shfl_down_sync`, für df64 eine Überladung in `df64.hpp` aus zwei float-Shuffles (gefunden über ADL). Baumstruktur und Reihenfolge sind dieselben wie in FP64; zehn Läufe sind bitgleich (ADR-0007).

**Speicher.** df64 belegt 8 Byte wie double; Shared Memory (statisch und dynamisch: Geisterdaten, OPL im Kernel `MERIT`), Local Memory (`RaySurfaceRecord`, `SurfaceExtrema`) und alle Gerätepuffer sind gleich groß. Die Speicherabschätzung der TraceEngine (`max_variants`, Tranchen) hängt deshalb nicht vom Zahlentyp ab, und `merit_max_samples` liefert für beide denselben Wert (Test `CudaDf64.SettingsAndLimits`, aus den Kernelattributen der jeweiligen Variante).

**Register und Local Memory** (`cuobjdump --dump-resource-usage`, CUDA 12.9; FP64 zum Vergleich wie oben):

| Kernel | sm_60 | sm_86 |
|---|---|---|
| `residuals_kernel<df64, false>` | 96 Register, Stack 40 B | 96 Register, Stack 48 B |
| `residuals_kernel<df64, true>` | 96 Register, Stack 2.096 B | 96 Register, Stack 2.096 B |
| `merit_kernel<df64, false>` | 93 Register, Stack 32 B | 104 Register, Stack 32 B |
| `merit_kernel<df64, true>` | 96 Register, Stack 2.080 B | 104 Register, Stack 2.080 B |

Mit 96 Registern passen auf sm_86 5 Blöcke à 128 Threads pro SM (20 von 48 Warps), mit 104 Registern 4 (wie FP64 mit 110 Registern). Durchsatz: [benchmarks.md, „df64“](benchmarks.md#df64-m7-t1); auf der RTX 3090 rechnet der Kernel in df64 etwa 4-mal so schnell wie in FP64. Einsatz: GeForce-Karten (RTX 3090) mit `Precision::Df64`, die Tesla P100 (FP64 mit halber FP32-Rate) bleibt bei FP64 ([hardware.md](hardware.md#zielhardware)).

## FP32-Vorfilterung (M7-T2)

Schneller Pfad zum Aussortieren hoffnungsloser Multistart-Varianten; Fehlerabschätzung, Margen und Entscheidungsregel: [numerik.md, „FP32-Vorfilterung“](numerik.md#fp32-vorfilterung-m7-t2), Einsatz: [optimierung.md, Abschnitt 8](optimierung.md#8-kandidatenverwaltung). **Einziger Pfad mit R = float** (AGENTS.md §6.1); er ist kein Modus der Engine und kein Wert von `Precision`, damit seine Ergebnisse nicht versehentlich als Merit verwendet werden. Die Namen aller Bausteine enthalten „Prefilter“.

| Baustein | Ort | Inhalt |
|---|---|---|
| Pro Strahl | `include/mikroptim/merit/prefilter_ray.hpp` (header-only, `MKO_HD`) | `prefilter_ray` (Trace, Referenzsphäre, Grenzabstand), `PrefilterMarginObserver`, `prefilter_opl_offset` |
| Entscheidung, CPU-Referenz | `include/mikroptim/batch/batch_prefilter.hpp`, `src/batch/batch_prefilter.cpp` (`Mikroptim::batch`) | `PrefilterCriteria`, `PrefilterMargins`, `PrefilterStats`, `PrefilterResult`, `prefilter_decide`, `prefilter_decide_exact`, `prefilter_stats_cpu`, `prefilter_cpu`, `prefilter_rays_cpu` |
| Kernel | `src/cuda/prefilter.cu`, Header `include/mikroptim/cuda/prefilter.hpp` (reines C++) | `prefilter_kernel`, `prefilter_finalize_kernel`, `DevicePrefilter`, `prefilter_max_samples`, `make_prefilter_output_layout` |
| Engine | `TraceEngine::prefilter_variants` (Implementierung in `src/cuda/prefilter.cu`) | Prüfung, Tranchen, Upload, Kernel, Download der Rohdaten, Entscheidung auf dem Host |

```cpp
// include/mikroptim/cuda/trace_engine.hpp
struct PrefilterSettings { ResidualSettings quadrature; PrefilterOptions options;   // Kriterien, Margen
                           std::span<const double> extra_merit; };                 // leer oder [V], FP64
TraceRunInfo TraceEngine::prefilter_variants(const SystemBatchHost&, const PrefilterSettings&,
                                             PrefilterResult& out);
PrefilterResult TraceEngine::prefilter_variants(const SystemBatchHost&, const PrefilterSettings& = {});

// include/mikroptim/batch/batch_prefilter.hpp (Host, auch ohne CUDA)
struct PrefilterCriteria { double max_failed_fraction = 0.5, max_merit = inf, max_merit_factor = inf; };
struct PrefilterMargins  { double boundary = 0.15, opl_error_factor = 32, merit_relative = 0.01; };
struct PrefilterStats    { int variants; int64_t rays_per_variant; vector<double> merit, optical_length;
                           vector<int64_t> failed, uncertain_ok, uncertain_failed; };
struct PrefilterResult   { PrefilterStats stats; vector<double> merit_lower, merit_upper;
                           vector<uint8_t> reject;            // Bitmaske PrefilterReason, 0 = behalten
                           double reference_merit; int reference_variant, kept; bool keep(int v) const; };
```

**Ablauf im Block** (b = (v·F + f)·L + l, 128 Threads, wie MERIT): Flächentabelle, Quadratur und Geisterdaten in float in den Shared Memory (`load_block_table<float>`, `load_block_ghosts<float>`); `opl_offset` = axiale optische Länge des Tupels (jeder Thread rechnet sie aus dem Shared Memory, keine weitere Synchronisation). Durchlauf 1: `prefilter_ray` pro Stützstelle, Klasse (sicher gültig, sicher versagt, unsicher gültig, unsicher versagt) und OPL in den dynamischen Shared Memory, Teilsummen (W, W·OPL) und größte optische Länge der sicher gültigen Strahlen, Zähler der Klassen. Kolben `block_piston<float>` über die sicher gültigen Strahlen; Durchlauf 2: Σ r² der sicher gültigen Strahlen. Reduktionen wie MERIT: `block_sum` (float bzw. int32), `block_max` für die optische Länge; Thread 0 schreibt pro Block Σ r², Länge (als double) und drei int32-Zähler. Zweite Stufe `prefilter_finalize_kernel` (ein Thread pro Variante, Blöcke in der Reihenfolge f, l): Summe in double, Maximum, Zähler als int64. Keine Gleitkomma-Atomics; Ergebnisse bitgleich über Läufe und Tranchen.

**Speicher.** Ausgaben pro Variante 5 × 8 Byte, pro Block 2 × 8 + 3 × 4 Byte, dazu die Feldablage (`make_prefilter_output_layout`, Bereiche auf 256 Byte ausgerichtet); die Engine bestimmt die Tranche per Bisektion über `prefilter_device_bytes` (Batch + Ausgaben) gegen den nutzbaren Speicher. Die Puffer von RESIDUALS und MERIT bleiben erhalten, solange sie die Tranche nicht unter V drücken; der Puffer der Vorfilterung wird beim nächsten `run` freigegeben (er zählt in `available_bytes` als verfügbar). Shared Memory: statisch 1.104 Byte, dynamisch Geisterdaten (G·K + 2G)·4 Byte (auf 8 aufgerundet) und 5 Byte pro Stützstelle; höchstens 48 KB, also bis 9.609 Stützstellen pro Feld ohne und 7.868 im ungünstigsten Fall mit Geisterplatten (G = 64, K = 32, `prefilter_max_samples`).

**Register** (`cuobjdump --dump-resource-usage`, CUDA 12.9): `prefilter_kernel` 72 Register, Stack 32 B auf sm_60 und sm_86 (sm_86: 7 Blöcke à 128 Threads pro SM, 28 von 48 Warps). Durchsatz: [benchmarks.md, „FP32-Vorfilterung“](benchmarks.md#fp32-vorfilterung-m7-t2).

## Modus EVAL (M8-T1)

Auswertung auf einem dichten kartesischen Pupillenraster: OPD-Karten, Spotdiagramme, Verzeichnung und Feldkrümmung pro Variante, Feld und Wellenlänge; jedes Versagen hart. Formeln (Raster, Referenzsphäre durch die Austrittspupille, Kennzahlen, Feldstrahlen, Rasterfehler) und Abnahme: [numerik.md, „Auswertung auf dem Raster“](numerik.md#auswertung-auf-dem-raster-m8-t1). Wie die Vorfilterung ist EVAL eine eigene Funktion der Engine (`TraceEngine::evaluate`) und kein Wert von `TraceMode`, weil Ausgabe und Ablage andere sind als bei RESIDUALS und MERIT.

| Baustein | Ort | Inhalt |
|---|---|---|
| Pro Strahl | `include/mikroptim/eval/eval_ray.hpp` (header-only, `MKO_HD`) | `EvalReference`, `eval_source_ray`, `exit_pupil_sphere_step`, `eval_ray`, Feldstrahlen `eval_field_ray_point` |
| Host, CPU-Referenz, Kennzahlen | `include/mikroptim/eval/eval.hpp`, `src/eval/eval.cpp` (Bibliothek `Mikroptim::eval`, linkt `batch` und privat `optim` für `batch_paraxial`) | `EvalSettings`, `EvalGrid`, `EvalResult`, `eval_reference_geometry`, `eval_cpu`, `eval_statistics`, `eval_opd_map`, `EvalOutputLayout`, `eval_device_bytes` |
| Kernel | `src/cuda/eval.cu`, Header `include/mikroptim/cuda/eval.hpp` (reines C++) | `eval_kernel`, `DeviceEval`, `TraceEngine::evaluate` |
| CLI | `src/cli/eval_command.cpp` | `mikroptim eval` ([cli.md](cli.md#mikroptim-eval)) |

```cpp
// include/mikroptim/eval/eval.hpp (Host, auch ohne CUDA)
struct EvalSettings { int grid = 64; bool half_pupil = false;
                      EvalReference reference = EvalReference::ExitPupil;   // oder Vertex
                      double field_ray_offset = 1e-3; };                    // δ der Feldstrahlen
struct EvalPoint { double x, y, weight; };
struct EvalGrid { int size, full_count; bool half; vector<EvalPoint> points; vector<int32_t> column, row; };
EvalGrid make_eval_grid(int size, bool half_pupil);                        // N ∈ [1, kMaxEvalGrid = 2048]
struct EvalReferenceGeometry { vector<double> curvature /*[V·F] κ = σ/R*/, exit_pupil_position /*[V]*/; };
EvalReferenceGeometry eval_reference_geometry(const SystemBatchHost&, EvalReference);
struct EvalResult { int variants, fields, wavelengths; EvalSettings settings; EvalGrid grid; double reference_nm;
    vector<double> image_height, reference_curvature /*[V·F]*/, exit_pupil_position /*[V]*/;
    vector<double> opd, spot_x, spot_y; vector<uint8_t> status;       // [V·F·L·n], NaN bei Versagen
    vector<double> field_rays /*[V·F·L·4·5]*/; vector<uint8_t> field_status /*[V·F·L·4]*/; ... };
EvalResult eval_cpu(const SystemBatchHost&, const EvalSettings& = {});
struct EvalTupleStats { rays, failed, valid_weight, rms_opd, pv_opd, rms_opd_tilt, rms_opd_defocus,
    tilt_x, tilt_y, defocus, centroid_x, centroid_y, rms_spot, image_height, chief_x, chief_y,
    distortion_chief, distortion_centroid /*[%]*/, focus_tangential, focus_sagittal; astigmatism(); };
std::vector<EvalTupleStats> eval_statistics(const EvalResult&, int v);     // Ablage [f·L + l]
EvalMap eval_opd_map(const EvalResult&, int v, int f, int l, OpdRemoval = Piston); // N×N, NaN außen
// include/mikroptim/cuda/trace_engine.hpp
TraceRunInfo TraceEngine::evaluate(const SystemBatchHost&, const EvalSettings&, EvalResult& out);
EvalResult   TraceEngine::evaluate(const SystemBatchHost&, const EvalSettings& = {});
```

**Ablage.** Rasterpunkte zeilenweise (Zeile j aufsteigend in y, darin Spalte i aufsteigend in x), nur die Punkte in der Pupille (n = `grid.samples()`, für alle Felder dasselbe Raster); Strahl k des Tupels (v, f, l) unter ((v·F + f)·L + l)·n + k, Feldstrahl j (0 Hauptstrahl, 1/2 tangential ±δ, 3 sagittal) unter ((v·F + f)·L + l)·4 + j mit fünf Werten (x, y relativ zum idealen Bildpunkt, d_x, d_y, d_z). `eval_opd_map` legt die Karte wieder auf das volle N×N-Raster (halbe Pupille gespiegelt).

**Ablauf im Block** (b = (v·F + f)·L + l, 128 Threads wie RESIDUALS, Rechentyp immer FP64):

1. Flächentabelle mit den Brechzahlen der Wellenlänge l und die Geisterdaten in den Shared Memory (`load_block_table`, `load_block_ghosts` aus `wavefront_block.hpp`), `__syncthreads`; Thread 0 rechnet den Hauptstrahl als Bezugswert O₀ der OPL, `__syncthreads`.
2. Durchlauf 1: Thread t bearbeitet die Rasterpunkte k = t, t + 128, … (Punkte aus einem Gerätefeld `EvalPoint[n]`), `eval_ray` (Startstrahl aus (x, y), `trace_ray`, d_z-Prüfung, Schnitt mit der Bildebene, Schritt auf die Referenzsphäre), schreibt Status und Spot (NaN bei Versagen) und OPL − O₀ vorläufig auf den OPD-Platz, Teilsummen (W, W·(OPL − O₀)) der gültigen Strahlen.
3. Kolben mit `block_piston` (Baumreduktion im Warp, Warps in fester Reihenfolge, keine Atomics).
4. Durchlauf 2: OPD = (OPL − O₀) − Kolben, NaN bei Versagen oder ohne gültigen Strahl.
5. Threads 0 … 3 rechnen die vier Feldstrahlen.

Versagte Strahlen laufen maskiert weiter. Die Krümmungen κ der Referenzsphären [V·F] rechnet der Host vor dem Lauf (`eval_reference_geometry`: Austrittspupille aus `batch_paraxial` bei λ_ref, κ = 0 im Unendlichen) und lädt sie mit dem Raster hoch. Kennzahlen (Fits, Schwerpunkt, Verzeichnung, Brennlinien) entstehen auf dem Host mit `eval_statistics` aus den heruntergeladenen Rohdaten, für CPU und GPU mit derselben Funktion. Shared Memory: statisch 1.968 Byte (Tabelle, Kolben, Bezugswert), dynamisch nur die Geisterdaten (≤ 17 KiB); die Rasterpunkte liegen im globalen Speicher, die Zahl der Rasterpunkte ist deshalb nicht durch das Shared Memory begrenzt. Register (`cuobjdump --dump-resource-usage`, CUDA 12.9): `eval_kernel` 114 Register, Stack 72 B auf sm_60; 96 Register, Stack 88 B auf sm_86.

**Engine.** `evaluate` prüft Einstellungen und Werte (`check_eval_settings`, `validate_values`), legt die Host-Ausgabe an (`prepare_eval_result`) und rechnet die Referenzgeometrie, bevor das Gerät angefasst wird. Tranche T = größtes V mit `eval_device_bytes(V)` ≤ nutzbarer Speicher (Bisektion, V·F·L ≤ 2³¹ − 1); die Puffer von RESIDUALS, MERIT und der Vorfilterung bleiben erhalten, solange die Tranche dadurch nicht kleiner als V wird (wie bei der Vorfilterung), sonst werden sie freigegeben. Der Ausgabepuffer (`DeviceEval`) lebt nur während des Aufrufs: EVAL ist selten und seine Puffer sind groß, er soll den Optimierungsmodi danach nicht im Weg liegen. Pro Tranche Upload, Kernel, Download der Rohdaten an die Stelle `first` von `out`. Die Ergebnisse jeder Variante sind unabhängig von der Aufteilung bitgleich (Test `CudaEval.RepeatedRunsAndTranchesAreBitIdentical`: 5 Läufe und 50 Varianten in 5 Tranchen bitgleich zum Lauf in einer Tranche).

**Speicher** (`make_eval_output_layout`, `eval_device_bytes`; Bereiche auf 256 Byte ausgerichtet): pro Rasterstrahl 25 Byte (OPD, Spot x, Spot y je 8, Status 1), pro Tupel 4 × 5 × 8 + 4 Byte Feldstrahlen, pro Variante und Feld 8 Byte κ, einmal pro Lauf das Raster (24 Byte pro Punkt), dazu der Batch. Beispiel Zielsystem-Topologie (S = 30, L = 7, F = 5, G = 6, K = 8) mit N = 128 (12.892 Punkte): ca. 11,3 MB pro Variante, also ca. 1.500 Varianten pro Tranche in 16 GiB (Test `EvalMemory.DeviceBytesFollowLayout`). Die Host-Ausgabe hat dieselbe Größe.

**Durchsatz** (RTX 3090, Test `CudaEval.LargeGridThroughput`, Information, über die Engine mit Upload, Download und Puffern; Mehrlinser F = 3, L = 4): 1 Variante mit N = 512 (12 Blöcke, 2,5·10⁶ Strahlen) 64 ms, 10 Varianten mit N = 256 (120 Blöcke, 6,2·10⁶ Strahlen) 170–230 ms, 400 Varianten mit N = 64 (4.800 Blöcke, 1,5·10⁷ Strahlen, 388 MB Ausgabe) 284 ms, also 2,7–5,5·10⁷ Strahlen/s; der Download der Rohdaten (25 Byte pro Strahl) ist ein wesentlicher Teil. Bei wenigen Tupeln rechnen nur V·F·L Blöcke (ein Block pro Tupel, ADR-0004), also nur so viele SMs; die Pupille eines Tupels auf mehrere Blöcke zu verteilen (Kolben dann zweistufig) wäre eine spätere Optimierung (M7).

## PSF, MTF, Strehl (M8-T2)

Skalare Beugung aus den OPD-Karten des Modus EVAL über eine zweidimensionale FFT; Formeln, Abtastung, Vorzeichen, Grenzen bei hoher NA und Abnahme: [numerik.md, „Beugung: PSF, MTF, Strehl“](numerik.md#beugung-psf-mtf-strehl-m8-t2). Die PSF ist eine Nachbearbeitung der heruntergeladenen EVAL-Ergebnisse (Host → Gerät: nur die N×N-Karten), kein weiterer Modus der TraceEngine.

| Baustein | Ort | Inhalt |
|---|---|---|
| FFT (CPU) | `include/mikroptim/eval/fft.hpp`, `src/eval/fft.cpp` (`Mikroptim::eval`) | `fft`, `fft2d`: Radix-2 in FP64, Konvention wie cuFFT (vorwärts e^{−}, ohne Normierung) |
| Elementweise Schritte | `include/mikroptim/eval/psf_ops.hpp` (header-only, `MKO_HD`) | `psf_pupil_value`, `psf_fft_index`/`psf_output_index` (Zentrierung, Spiegelung), `psf_norm2`, `PsfPupilParams` |
| Host, CPU-Referenz, Kennzahlen | `include/mikroptim/eval/psf.hpp`, `src/eval/psf.cpp` | `PsfSettings`, `psf_fft_size`, `PsfPupil`, `make_psf_pupil(s)`, `PsfResult`, `prepare_psf_result`, `finish_psf`, `psf_cpu`, `PsfDeviceLayout` |
| GPU | `include/mikroptim/cuda/psf.hpp` (reines C++), `src/cuda/psf.cu` (linkt `CUDA::cufft` privat) | `PsfEngine`, `PsfRunInfo` |
| CLI | `src/cli/eval_command.cpp` | `mikroptim eval --psf` ([cli.md](cli.md#mikroptim-eval)) |

```cpp
// include/mikroptim/eval/psf.hpp (Host, auch ohne CUDA)
struct PsfSettings { double padding = 4.0;              // Q, M = Zweierpotenz >= max(2N, Q·N)
                     double numerical_aperture = 0.0; }; // 0: NA aus den Feldstrahlen
int psf_fft_size(int grid, double padding);              // <= kMaxPsfFft = 8192
struct PsfPupil { int field, wavelength, grid, full_count; double wavelength_nm, na_x, na_y;
                  bool flip_x, flip_y; vector<double> opd; /*[N·N] mm, NaN = undurchlässig*/ };
PsfPupil make_psf_pupil(const SystemBatchHost&, const EvalResult&, int v, int f, int l, const PsfSettings& = {});
std::vector<PsfPupil> make_psf_pupils(const SystemBatchHost&, const EvalResult&, int v, const PsfSettings& = {});
struct PsfResult { int field, wavelength, grid, fft_size; double wavelength_nm, na_x, na_y, dx, dy, airy_radius;
    vector<double> psf;                                  // [M·M] zentriert, Maximum der perfekten PSF = 1
    vector<double> mtf_sagittal, mtf_tangential;         // [N + 1], f = k·f_c/N
    int64_t transmitted; double energy, strehl_fft, strehl_origin, strehl, reference_x, reference_y,
    peak, peak_x, peak_y; cutoff_x(), cutoff_y(), mtf_*_at(ν) };
PsfResult psf_cpu(const PsfPupil&, int fft_size);
std::vector<PsfResult> psf_cpu(const SystemBatchHost&, const EvalResult&, int v, const PsfSettings& = {});
// include/mikroptim/cuda/psf.hpp
class PsfEngine { explicit PsfEngine(int device, const EngineOptions& = {});
    std::vector<PsfResult> compute(std::span<const PsfPupil>, int fft_size);
    std::vector<PsfResult> compute(const SystemBatchHost&, const EvalResult&, int v, const PsfSettings& = {});
    const PsfRunInfo& last_run() const; size_t device_bytes() const; void release(); };
```

**Ablauf auf der GPU** (pro Tranche von T Pupillen, alle mit derselben Rasterkante N und FFT-Größe M):

1. Upload der OPD-Karten [T·N²] und der Parameter pro Pupille (λ, 1/n_voll², Spiegelung).
2. `psf_pupil_kernel`: Pupillenfunktion in die Ecke jedes M×M-Feldes, sonst 0 (Grid-Stride-Schleife über T·M², 256 Threads).
3. `cufftExecZ2Z` vorwärts, ein Plan mit Batch T (`cufftMakePlanMany64`, Rang 2, dichte Ablage, Abstand M²).
4. `psf_intensity_kernel`: I = |U|²/n_voll² an die zentrierte (ggf. gespiegelte) Stelle des PSF-Puffers und reell zurück in das Spektrum.
5. `cufftExecZ2Z` rückwärts (OTF), `psf_mtf_kernel`: MTF-Schnitte (k, 0) und (0, k), k = 0 … N, normiert auf OTF(0).
6. Download von PSF [T·M²] und MTF [T·2(N + 1)], Kennzahlen mit `finish_psf` auf dem Host (dieselbe Funktion wie die CPU-Referenz).

Jeder CUDA-Aufruf ist mit `MKO_CUDA_CHECK`, jeder cuFFT-Aufruf mit `MKO_CUFFT_CHECK` geprüft (`cufftResult` → `CudaError` mit Namen des Fehlers), nach jedem Kernelstart `cudaGetLastError()`. Keine Reduktionen auf dem Gerät (Kennzahlen auf dem Host, sequenziell), cuFFT ist für denselben Plan deterministisch.

**Speicher und Tranchen** (`make_psf_device_layout`, Bereiche auf 256 Byte ausgerichtet): pro Pupille 24 Byte pro FFT-Punkt (Spektrum komplex 16, PSF 8), 8 Byte pro Rasterpunkt, 16(N + 1) Byte MTF, 24 Byte Parameter, dazu der Arbeitsspeicher von cuFFT (automatische Allokation abgeschaltet, `cufftSetAutoAllocation(plan, 0)`, Bereich hinter den Puffern in derselben Allokation, `cufftSetWorkArea`). Die Tranche T ist die größte Zahl Pupillen, deren Puffer und Arbeitsspeicher in den nutzbaren Speicher passen (freier Speicher nach `cudaMemGetInfo` plus eigene Puffer, ggf. `EngineOptions::memory_limit_bytes`, abzüglich `MemoryReserve`): erst aus der Abschätzung pro Pupille (`cufftGetSizeMany64`), dann mit dem exakten Arbeitsspeicher des Plans geprüft und bei Bedarf um ein Viertel verkleinert. Die letzte, unvollständige Tranche läuft mit demselben Plan (leere Plätze sind 0), damit jede Pupille unabhängig von ihrer Lage mit demselben Plan gerechnet wird. Plan und Puffer bleiben zwischen den Aufrufen erhalten, solange M und T gleich bleiben. Beispiel: N = 128, M = 512 (Q = 4) belegt 6,0 MiB pro Pupille plus Arbeitsspeicher; das Zielsystem mit F·L = 35 Tupeln passt in einer Tranche. Höchstens M = 8192 (1,5 GiB pro Pupille).

**Host-Speicher.** `PsfResult::psf` hat M² Werte pro Tupel; `mikroptim eval --psf` rechnet die Tupel deshalb einzeln und behält die PSF nur für das Tupel der Ausgabedateien.

## Vektorielle PSF (M8-T5)

Vektorielle Beugung nach Richards-Wolf aus denselben OPD-Karten (Nachbearbeitung wie die skalare PSF); Modell, Apodisation, Polarisationsdrehung, Normierung, Gültigkeit und Abnahme: [numerik.md, „Vektorielle PSF“](numerik.md#vektorielle-psf-m8-t5).

| Baustein | Ort | Inhalt |
|---|---|---|
| Elementweise Schritte | `include/mikroptim/eval/vector_psf_ops.hpp` (header-only, `MKO_HD`) | `VectorPsfParams` (88 Byte pro Pupille), `VectorPsfJones`, `vector_psf_basis` (Richtung, 1/√s_z, Drehung **u** → **s**, Defokus-OPD), `vector_psf_pupil_value` (drei Feldkomponenten eines Rasterpunkts) |
| Host, CPU-Referenz, Kennzahlen | `include/mikroptim/eval/vector_psf.hpp`, `src/eval/vector_psf.cpp` (`Mikroptim::eval`) | `Polarization`, `VectorPsfSettings`, `VectorPsfPupil`, `make_vector_psf_pupil(s)`, `vector_psf_reference_intensity`, `VectorPsfResult` (erbt `PsfResult`), `finish_vector_psf`, `vector_psf_intensity_at` (direkte Summe), `vector_psf_cpu`, `VectorPsfDeviceLayout` |
| GPU | `include/mikroptim/cuda/vector_psf.hpp` (reines C++), `src/cuda/vector_psf.cu`; Prüfung der cuFFT-Aufrufe gemeinsam mit M8-T2 in `src/cuda/cufft_check.hpp` | `VectorPsfEngine` (Laufinfo `PsfRunInfo`) |
| CLI | `src/cli/eval_command.cpp` | `mikroptim eval --vector [--polarization P] [--defocus Z] [--component-files P]` ([cli.md](cli.md#mikroptim-eval)) |

```cpp
// include/mikroptim/eval/vector_psf.hpp (Host, auch ohne CUDA)
enum class Polarization { X, Y, Circular, Unpolarized };
struct VectorPsfSettings { Polarization polarization = Polarization::X; double defocus = 0.0; /*mm, > 0: in +z*/ };
struct VectorPsfPupil { PsfPupil pupil; double index, chief_x, chief_y, input_y, input_z; };
VectorPsfPupil make_vector_psf_pupil(const SystemBatchHost&, const EvalResult&, int v, int f, int l, const PsfSettings& = {});
struct VectorPsfResult : PsfResult {                    // psf = |E|²/I_0, MTF, Strehl, Energie, Maximum
    Polarization polarization; double defocus, reference_intensity;   // I_0
    std::array<std::vector<double>, 3> component;       // |E_x|², |E_y|², |E_z|² [M·M]
    std::array<double, 3> energy_fraction, component_peak; double fwhm_x, fwhm_y; int64_t evanescent; };
VectorPsfResult vector_psf_cpu(const VectorPsfPupil&, int fft_size, const VectorPsfSettings& = {});
double vector_psf_intensity_at(const VectorPsfPupil&, const VectorPsfSettings&, double I0, double x, double y,
                               std::array<double, 3>* components = nullptr);
// include/mikroptim/cuda/vector_psf.hpp
class VectorPsfEngine { explicit VectorPsfEngine(int device, const EngineOptions& = {});
    std::vector<VectorPsfResult> compute(std::span<const VectorPsfPupil>, int fft_size, const VectorPsfSettings& = {});
    std::vector<VectorPsfResult> compute(const SystemBatchHost&, const EvalResult&, int v,
                                         const PsfSettings& = {}, const VectorPsfSettings& = {}); };
```

**Ablauf auf der GPU** (pro Tranche von T Pupillen mit K = 1 oder 2 Eingangsfeldern, gleiche Rasterkante N und FFT-Größe M):

1. Host: I_0 jeder Pupille (`prepare_vector_psf_result`), Upload der OPD-Karten [T·N²] und der `VectorPsfParams` [T]; die Jones-Vektoren stehen in den Kernel-Argumenten.
2. `vector_psf_pupil_kernel`: ein Thread pro (Pupille, Eingangsfeld, Pixel) schreibt die drei Feldkomponenten G_c in die Ecke der Felder (t, k, c), sonst 0 (Grid-Stride über T·K·M², 256 Threads).
3. `cufftExecZ2Z` vorwärts mit einem Plan für alle T·K·3 Felder (Abstand M²).
4. `vector_psf_intensity_kernel`: pro Pixel Σ_k w_k |E_c|² · (1/I_0) für c = x, y, z, Intensität als Summe, beide an die zentrierte (ggf. gespiegelte) Stelle; die Intensität reell in das Feld (t, 0, 0) (derselbe Thread liest und schreibt das Pixel).
5. `cufftExecZ2Z` rückwärts mit einem zweiten Plan (Batch T, Abstand K·3·M², `inembed`/`onembed` = {M, M}) nur über die Felder (t, 0, 0), `vector_psf_mtf_kernel` für die MTF-Schnitte.
6. Download von PSF, Komponenten [T·3·M²] und MTF, Kennzahlen mit `finish_vector_psf` auf dem Host (direkte Summen für Strehl und Halbwertsbreiten, dieselbe Funktion wie die CPU-Referenz).

**Speicher und Tranchen** (`make_vector_psf_device_layout`): pro Pupille 16·3K + 8 + 24 Byte pro FFT-Punkt (Feldkomponenten, PSF, Komponenten; K = 2: 128 Byte, also das 5,3-Fache der skalaren PSF), 8 Byte pro Rasterpunkt, 88 Byte Parameter, MTF; dazu der Arbeitsspeicher von cuFFT, den sich beide Pläne teilen (sie laufen nacheinander im selben Strom). Tranchenwahl wie bei der `PsfEngine` (Abschätzung mit `cufftGetSizeMany64` für beide Pläne, dann exakt, bei Bedarf um ein Viertel kleiner). Beispiel N = 128, M = 512, unpolarisiert: 32 MiB pro Pupille plus Arbeitsspeicher.

**Host-Speicher.** `VectorPsfResult` hält 4·M² Werte pro Tupel; `mikroptim eval --vector` behält PSF und Komponenten nur für das Tupel der Ausgabedateien.

## Modus ADJOINT: Vorwärtsdurchlauf mit Aufzeichnung (M5-T1)

Grundlage des adjungierten Modus (ADR-0009): Pro Strahl speichert der Vorwärtsdurchlauf die Zwischenzustände, aus denen der Rückwärtsdurchlauf (M5-T2) die lokalen Ableitungen jeder Flächenoperation bildet. Pro-Strahl-Teil header-only in `include/mikroptim/trace/adjoint_record.hpp` (`MKO_HD`, über R templatisiert, feste Größe), Grundgerüst des Kernels in `src/cuda/adjoint.cu`, Host-API `include/mikroptim/cuda/adjoint.hpp` (reines C++). Welche Größen der Rückwärtsdurchlauf nutzt: [numerik.md, Abschnitt 5](numerik.md#zwischengrößen-für-den-rückwärtsdurchlauf-vorschau-m5-t1).

### Minimaler Satz

Gespeichert werden pro Strahl und Fläche s nur der Zustand nach der Fläche – Schnittpunkt **p**_s und Richtung **d**_s (6 Zahlen) – und der genommene Zweig des weichen Versagens (1 Byte, `kBranchRegular`, `kBranchSoftMissed`, `kBranchSoftTIR`), dazu einmal pro Strahl s_ref und Zweig der Referenzsphäre. Alle lokalen Zwischengrößen (s₀, s, H, B, √D, Normale, cos I, k, √k; an Geisterplatten u, t(u), dt/du, Φ, ∇Φ, Radikand; an der Referenzsphäre b, g, √(b² − g), σ) folgen in O(1) aus dem Eingangszustand (**p**_{s−1}, **d**_{s−1}) und der Flächentabelle: `recompute_surface_step` und `recompute_reference_step` liefern sie als `SurfaceStepState` bzw. `ReferenceStepState`, mit Ausgang, Status und Zweig aus derselben `propagate_surface` bzw. `reference_sphere_step` (bitgleich zur Aufzeichnung). Gründe gegen das Speichern der ca. 20 Zwischengrößen pro Fläche:

- **Speicher und Bandbreite:** 49 statt ca. 170 Byte pro Strahl und Fläche (FP64). Der Rückwärtsdurchlauf rechnet ohnehin pro Fläche; die Neuberechnung kostet etwa einen zusätzlichen Flächenschritt, also einen weiteren Vorwärtsdurchlauf, und ändert die Größenordnung „zwei bis drei Traces“ der Adjungierten nicht.
- **Numerik:** Zwischengrößen im Vorwärtsdurchlauf abzugreifen hieße, Produkte wie Φ = (n_G − n)·t oder ∇Φ zusätzlich zu verwenden; das kann ihre FMA-Kontraktion in OPL += Φ bzw. n d_⊥ + ∇Φ verhindern und damit die OPL ändern. Die Aufzeichnung liest nur den Strahlzustand, den der Trace ohnehin hält.
- **Nicht gespeichert** wird die OPL (geht linear ein, die Adjungierte ist konstant) und die Normale (aus **p**_s und c).

### Recorder und Flächenschleife

`trace_segment(sys, l, ray, at, end, recorder, failed, observer, soft)` ist die einzige Schleife über die Flächen (`trace.hpp`): Sie setzt den Trace vom Zustand nach Fläche `at.surface` (Strahl, `TraceProgress` mit Zahl der passierten Geisterplatten und Fläche des ersten weichen Ereignisses, Akkumulator `SoftFailure`) bis vor `end` fort und ruft nach jeder Fläche den Recorder mit Strahl, Fortschritt, Zweig und Akkumulator auf. `trace_ray`, `trace_ray_recorded`, `trace_to_reference_sphere` (Recorder als letzter optionaler Parameter, zusätzlich `image(s_ref, Zweig, Strahl)`) und `batch_ray_opl` bauen darauf auf; ohne Recorder (`NoTraceRecorder`) entfallen die Aufrufe zur Übersetzungszeit. Den Zweig liefert `propagate_surface` über den neuen optionalen Ausgabeparameter `branch`; dafür melden `refract_sphere` und `phase_surface_step` die Fortsetzung jetzt mit `SoftTIR` statt `Ok`.

### Datenstrukturen

```cpp
// include/mikroptim/trace/adjoint_record.hpp
inline constexpr int kRecordCapacitySmall = 16, kRecordCapacityMedium = 32,
                     kRecordCapacityLarge = kMaxSurfaces;          // Kapazitätsklassen
constexpr int record_capacity_class(int surfaces);                // kleinste Klasse >= S
inline constexpr int kCheckpointInterval = 8;
template <class R, int C = kMaxSurfaces> struct AdjointRecord {   // alle Zustände
    int count; bool soft_mode;                 // gültige Zustände; Trace mit Akkumulator
    Vec3<R> p[C], d[C]; unsigned char branch[C];
    R image_s; unsigned char image_branch;     // Referenzsphäre
};
template <class R> struct TraceCheckpoint { Ray<R> ray; TraceProgress at; SoftFailure<R> soft; };
template <class R, int C = kMaxSurfaces, int k = kCheckpointInterval> struct CheckpointRecord {
    int count; bool soft_mode;
    TraceCheckpoint<R> checkpoint[C / k];      // Zustand nach Fläche c·k
    unsigned char branch[C]; R image_s; unsigned char image_branch;
};
template <class R, int k> struct SegmentStates { int first, count; Vec3<R> p[k], d[k]; };
TraceStatus replay_segment(sys, l, const CheckpointRecord&, int segment, SegmentStates&,
                           TraceCheckpoint<R>* end_state = nullptr);   // bitgleich
void for_each_state_reverse(sys, l, const Record&, fn);   // fn(s, p, d, branch), s = count−1 … 0
int ghost_index_before(sys, s);
SurfaceStepState<R> recompute_surface_step(sys, l, s, ghost, p_in, d_in, soft_mode);
ReferenceStepState<R> recompute_reference_step(p_in, d_in, center, n_image, soft_mode);
```

Eintrag s ist der Zustand nach Fläche s (Eintrag 0 der Startzustand, Richtung vor der Brechung an Fläche s ist `d[s − 1]`); bei hartem Versagen an Fläche s ist count = s, der Rückwärtsdurchlauf braucht dann nur die Zustände davor (Gradient der Strafen weicher Ereignisse). Flächen ab der Kapazität werden nicht aufgezeichnet; der Host wählt C ≥ S.

**Speicher pro Strahl** (`sizeof`, R = double; df64 gleich groß; Test `AdjointRecord.MemoryPerRay`):

| Form | C = 16 | C = 32 | C = 64 |
|---|---:|---:|---:|
| `AdjointRecord` (49 Byte pro Fläche + 24) | 808 B | 1.592 B | 3.160 B |
| `CheckpointRecord`, k = 8 (88 Byte pro Checkpoint, 1 Byte pro Fläche) | 216 B | 408 B | 792 B |
| dazu beim Lesen `SegmentStates<R, 8>` | 392 B | 392 B | 392 B |

### Wo die Aufzeichnung liegt

CUDA kennt kein zur Laufzeit bemessenes Local Memory. Erwogen wurden:

1. **Local Memory mit fester Kapazität** (Felder mit Index zur Laufzeit liegen dort): Local Memory ist ein Bereich des globalen Speichers, verschränkt pro Thread, sodass gleiche Indizes eines Warps koaleszieren; Zugriffe laufen über L1/L2. Der Treiber reserviert die Größe pro Thread für alle gleichzeitig möglichen Threads (RTX 3090: 82 SMs × 1.536 Threads, P100: 56 × 2.048), unabhängig von S und V.
2. **Kapazitätsklassen** 16, 32, 64 als Template-Instanzen (gewählt): Die Reservierung folgt der tatsächlichen Flächenzahl; der Verkehr ohnehin (nur geschriebene Einträge).
3. **Checkpoints** (gewählt als Option): Zustand nur alle k = 8 Flächen, beim Lesen jeder Abschnitt einmal neu vorwärts; spart Speicher und Bandbreite gegen einen zusätzlichen Vorwärtsdurchlauf.
4. **Globaler Arbeitspuffer pro Strahl** (verworfen): wächst mit V·F·L·N statt mit der Zahl gleichzeitig laufender Threads (z. B. 1.000 Varianten × 2.310 Strahlen × 30 Flächen × 49 B = 3,4 GB) und braucht eine eigene Ablage; Local Memory ist derselbe Speicher, nur automatisch verschränkt.
5. **Shared Memory** (verworfen): 48 KB / 128 Threads = 384 B pro Strahl, nicht einmal 8 Flächen.

**Register und Local Memory** (`cuobjdump --dump-resource-usage`, CUDA 12.9; Stack = Local Memory pro Thread; Test `CudaAdjoint.KernelResources` liest dasselbe über `adjoint_kernel_info`):

| Kernel `adjoint_forward_kernel<R, Form, C>` | sm_60 | sm_86 |
|---|---|---|
| double, ohne Aufzeichnung | 122 Register, Stack 88 B | 96 Register, Stack 104 B |
| double, vollständig, C = 16 / 32 / 64 | 122 Register, Stack 896 / 1.680 / 3.248 B | 96 Register, Stack 936 / 1.720 / 3.288 B |
| double, Checkpoints, C = 16 / 32 / 64 | 128 Register, Stack 816 / 1.008 / 1.392 B | 120 Register, Stack 784 / 976 / 1.360 B |
| df64, ohne Aufzeichnung | 121 Register, Stack 48 B | 96 Register, Stack 48 B |
| df64, vollständig, C = 16 / 32 / 64 | 126 Register, Stack 848 / 1.632 / 3.200 B | 117 Register, Stack 848 / 1.632 / 3.200 B |
| df64, Checkpoints, C = 16 / 32 / 64 | 119 Register, Stack 712 / 888 / 1.240 B | 96 Register, Stack 736 / 912 / 1.264 B |

Die vollständige Aufzeichnung kostet auf sm_86 keine Register (96 wie ohne; 5 Blöcke à 128 Threads pro SM), die Checkpoints wegen der Neuberechnung 120 (4 Blöcke). **Laufzeitreserve** des Local Memory: Stack × Threads pro SM × SMs, für C = 64 vollständig ca. 3,3 KB × 125.952 = 414 MB (RTX 3090) bzw. 3,3 KB × 114.688 = 373 MB (P100), für C = 16 ca. 118 bzw. 103 MB, Checkpoints C = 64 ca. 171 bzw. 160 MB. Sie liegt innerhalb der Standardreserve (max(256 MiB, 5 % des freien Speichers), auf der RTX 3090 ca. 1,1 GiB, auf der P100 ca. 0,8 GiB; [hardware.md](hardware.md#speicherreserve-und-tranchen)).

### Bitgleichheit und FMA-Kontraktion

Auf der CPU (keine Kontraktion) sind Trace mit und ohne Aufzeichnung und das Neu-Vorwärtsrechnen aus Checkpoints bitgleich, weil alle denselben `trace_segment`-Aufruf mit denselben Werten ausführen. Auf dem Gerät reicht das nicht: nvcc verschmilzt mit `--fmad=true` (Standard) Multiplikation und Addition zu FMA abhängig vom umgebenden Code (Grundblöcke, weitere Verwendungen eines Produkts). Gemessen mit einer ersten Fassung: Mit Aufzeichnung wichen 91 von 37.440 Strahlen des gestörten Mehrlinsers im harten Modus um 1 ulp der OPL vom Lauf ohne Aufzeichnung ab (im weichen Modus 1.222 Strahlen, tief im Versagensbereich bis 7·10⁻¹⁰ relativ), und das Neu-Vorwärtsrechnen aus Checkpoints wich vom ursprünglichen Durchlauf ab; ein gezieltes `__dmul_rn` für reine Produkte im Strahlzustand (Startstrahlen, normierte Richtungen) änderte daran nichts. df64 war immer bitgleich (dessen Operationen sind explizite FP32-Intrinsics und werden nie kontrahiert). Deshalb wird `src/cuda/adjoint.cu` mit **`--fmad=false`** übersetzt (CMake, nur diese Datei): Jede Operation ist dann einzeln IEEE-gerundet (Division und Wurzel ohnehin korrekt gerundet), das Ergebnis hängt nur von der Reihenfolge im Quelltext ab, und alle Speicherformen sowie das Neu-Vorwärtsrechnen sind bitgleich (Tests `CudaAdjoint.RecordingLeavesResultsBitIdentical`, `CudaAdjoint.CheckpointsAndCapacityClassesBitIdentical`). Das ist strenger als der Standard (keine fast-math-Option, AGENTS.md §6). Folgen:

- ADJOINT rechnet in FP64 nicht bitgleich zu RESIDUALS (mit FMA), sondern innerhalb der Toleranzen CPU/GPU: OPD gültiger Strahlen ≤ 0,1·`kCpuGpuFp64`-Schranke, weich versagter ≤ 0,003·`kCpuGpuSoftTuple`, Strafresiduen relativ ≤ 1,5·10⁻¹²; Status gleich. In df64 bitgleich zu RESIDUALS. Gegen die CPU-Referenz liegt FP64 ohne Kontraktion näher als mit (Abweichung der Zustände ≤ 0,008 der Toleranz).
- Für M5-T3 (LM mit adjungierter Jacobi) heißt das: Residuen und Jacobi-Zeilen kommen aus demselben Kernel ADJOINT; ob RESIDUALS und MERIT ebenfalls ohne Kontraktion rechnen sollen (dann alle Optimierungsmodi untereinander bitgleich), ist eine Frage an den Maintainer (PR M5-T1).

### Kernel-Grundgerüst

`adjoint_forward_kernel<R, Form, C>` (R = double oder df64, Form = keine, vollständig, Checkpoints, C = 16, 32, 64; 14 Instanzen): Block-Abbildung, Flächentabelle und Geisterdaten im Shared Memory, Kolben per Baumreduktion und Residuen wie RESIDUALS (dieselben Bausteine aus `wavefront_block.hpp`, deterministisch nach ADR-0007). Pro Stützstelle: `batch_ray_opl` mit der Aufzeichnung im Local Memory, danach `for_each_state_reverse` – genau das Zugriffsmuster des Rückwärtsdurchlaufs, bei Checkpoints mit dem Neu-Vorwärtsrechnen jedes Abschnitts –, daraus eine Prüfsumme pro Strahl (FNV-1a über die Bitmuster, `record_checksum_step`) und auf Wunsch die Ablage der Aufzeichnung im Gerätespeicher (Tests). In M5-T2 tritt an die Stelle der Prüfsumme der Rückwärtsdurchlauf (Gradient der OPL und der Strafe pro Strahl), danach die Blockreduktion der Gradienten für den Kolbenabzug; M5-T3 bindet den Modus in die TraceEngine ein.

```cpp
// include/mikroptim/cuda/adjoint.hpp (reines C++)
enum class RecordStorage { None = 0, Full = 1, Checkpoint = 2 };
struct AdjointForwardOptions { RecordStorage storage = Full; int capacity = 0;   // 0: automatisch
                               Precision precision = Fp64; bool opd = false; bool states = false; };
struct AdjointForwardResult { ResidualResult residuals; RecordStorage storage; int capacity, surfaces;
    std::vector<double> opl; std::vector<uint64_t> checksum;           // [V · rays]
    std::vector<int32_t> count; std::vector<double> states;           // nur mit states: [ray][S][6]
    std::vector<uint8_t> branch; std::vector<double> image_s; std::vector<uint8_t> image_branch; };
class DeviceAdjointForward { void run(const DeviceSystemBatch&, const ResidualLayout&,
                                      const AdjointForwardOptions& = {});
                             AdjointForwardResult download() const; void release() noexcept; };
AdjointForwardResult run_adjoint_forward(const DeviceSystemBatch&, const ResidualLayout&,
                                         const AdjointForwardOptions& = {});
AdjointKernelInfo adjoint_kernel_info(RecordStorage, int capacity, Precision = Fp64, int device = -1);
constexpr uint64_t record_checksum_step(uint64_t h, uint64_t word);
```

Ein `cudaMalloc` pro Puffergröße (wiederverwendet): Residuen, Status, OPD wie `make_output_layout`, dahinter Feldablage, OPL und Prüfsumme (je 8 Byte pro Strahl), mit `states` zusätzlich 4 + 49·S + 9 Byte pro Strahl. Jeder CUDA-Aufruf geprüft, nach dem Kernelstart `cudaGetLastError()`.

### Durchsatz und Wahl der Speicherform

`trace_throughput --mode residuals,adjoint --type fp64,df64 --surfaces 16,30,60 --variants 10000` (RTX 3090, 510 Strahlen pro Variante, Kernelzeit, Median aus 20 Läufen; die GPU wurde während der Messungen zeitweise von Tests anderer Arbeitsbäume mitbenutzt, die Absolutwerte streuen deshalb um bis zu 30 %, die Verhältnisse innerhalb eines Laufs sind belastbar; Messreihe und Wiederholung in [benchmarks.md](benchmarks.md#adjoint-grundgerüst-m5-t1)):

| S | Zahlentyp | RESIDUALS | ADJOINT ohne Aufzeichnung | vollständig | Checkpoints (k = 8) |
|---:|---|---:|---:|---:|---:|
| 16 | FP64 | 59,2 ms | 56,6 ms | 55,6 ms (−2 %) | 105,8 ms (+87 %) |
| 30 | FP64 | 77,3 ms | 94,2 ms | 100,6 ms (+7 %) | 158,9 ms (+69 %) |
| 60 | FP64 | 158,3 ms | 158,7 ms | 159,6 ms (+1 %) | 304,5 ms (+92 %) |
| 16 | df64 | 12,9 ms | 13,2 ms | 22,5 ms (+71 %) | 36,4 ms (+177 %) |
| 30 | df64 | 21,3 ms | 17,7 ms | 49,3 ms (+179 %) | 81,2 ms (+360 %) |
| 60 | df64 | 34,2 ms | 34,2 ms | 57,0 ms (+67 %) | 90,1 ms (+163 %) |

(Prozent gegen ADJOINT ohne Aufzeichnung, in dem die Aufzeichnung samt Lesen in absteigender Reihenfolge der einzige Unterschied ist.)

- **FP64 auf der RTX 3090** ist durch die FP64-Rechenleistung begrenzt: Schreiben und Lesen von 49 Byte pro Strahl und Fläche kosten 0–7 %; die Checkpoints verdoppeln dagegen fast die Zeit (das Neu-Vorwärtsrechnen ist ein zweiter Trace). Ohne FMA-Kontraktion (`--fmad=false`) ist ADJOINT ohne Aufzeichnung im Rahmen der Streuung so schnell wie RESIDUALS (S = 16 und 60 gleich; der Ausreißer bei S = 30 ist Messrauschen der geteilten GPU).
- **df64** rechnet viermal so schnell und wird mit vollständiger Aufzeichnung durch den Speicherverkehr begrenzt (+67 … +180 %); Checkpoints sind trotzdem langsamer, weil das Neu-Vorwärtsrechnen mehr kostet als der gesparte Verkehr.
- **Tesla P100** (FP64 mit halber FP32-Rate, ca. 9-mal so schnell wie die RTX 3090 in FP64, 732 GB/s): dort wird FP64 ähnlich wie df64 auf der 3090 verkehrsbegrenzt sein; nicht gemessen (keine P100 am Arbeitsplatz).

**Wahl:** vollständige Aufzeichnung (`AdjointRecord`) mit der kleinsten Kapazitätsklasse als Standard (`AdjointForwardOptions::storage = Full`, `capacity = 0`). Sie ist in allen Messungen die schnellere Form, braucht keine zusätzlichen Register und kostet höchstens ca. 0,4 GB Reserve. Checkpoints bleiben als Option für den Fall, dass Local Memory knapp wird (z. B. wenn der Rückwärtsdurchlauf in M5-T2 selbst viel Local Memory für P Gradientenkomponenten braucht) – sie senken den Stack von 3,3 KB auf 1,4 KB bei C = 64. Die Neuberechnung der Zwischengrößen im Rückwärtsdurchlauf (M5-T2) kommt in beiden Formen hinzu.

### Abnahmetests (M5-T1)

Im Plan hat M5-T1 kein eigenes Kriterium; festgelegt wurden:

| Kriterium | Test | Ergebnis |
|---|---|---|
| Aufzeichnung ändert den Trace nicht (Status, versagende Fläche, Ort, Richtung, OPL, Strafe), alle Formen und Kapazitäten, hart und weich | `AdjointRecord.RecordingLeavesTraceBitIdentical` | 46.068 Strahlen (623 weich, 5.135 hart versagt) bitgleich |
| Jeder gespeicherte Zustand gegen einen unabhängigen Trace Fläche für Fläche (eigene Schleife über `propagate_surface`, dann `reference_sphere_step`) | `AdjointRecord.StatesMatchIndependentStepByStepTrace` | 304.863 Zustände bitgleich (Plane, Sphere, GhostPlate; Zweige SoftMissed 2.284, SoftTIR 2.859, beide 39; Referenzsphäre weich verfehlt 155) |
| Zweige an bekannten Flächen (verfehlte Sphäre, Totalreflexion, beides, Radikand der Geisterplatte, Referenzsphäre) | `AdjointRecord.SoftBranchesAtKnownSurfaces` | erfüllt |
| Neu-Vorwärtsrechnen aus Checkpoints (k = 4, 8, 64; C = 32, 64) bitgleich, Endzustand eines Abschnitts setzt bitgleich bis zum nächsten Checkpoint fort | `AdjointRecord.CheckpointReplayIsBitIdentical` | 248.117 Abschnitte bitgleich |
| Lokale Zwischengrößen aus dem minimalen Satz | `AdjointRecord.RecomputedLocalQuantitiesFromMinimalSet` | Ausgänge bitgleich; Sphärengleichung ≤ 1,8·10⁻¹⁴ mm, Snellius ≤ 4,4·10⁻¹⁶, \|p − M\| − R ≤ 3·10⁻¹⁶ relativ |
| GPU: mit und ohne Aufzeichnung bitgleich (FP64, df64), gegen RESIDUALS innerhalb der Toleranzen (df64 bitgleich) | `CudaAdjoint.RecordingLeavesResultsBitIdentical` | 202.404 Strahlen |
| GPU-Aufzeichnung gegen CPU | `CudaAdjoint.RecordMatchesCpu` | 1,84·10⁶ Zustände, ≤ 0,008 der Toleranz `kCpuGpuFp64` |
| GPU: Checkpoints und Kapazitätsklassen untereinander bitgleich | `CudaAdjoint.CheckpointsAndCapacityClassesBitIdentical` | erfüllt |
| Determinismus, Prüfsumme gegen heruntergeladene Aufzeichnung | `CudaAdjoint.RepeatedRunsBitIdenticalAndChecksumMatchesRecord` | erfüllt |

## Modus ADJOINT: Jacobi-Matrix (M5-T2)

Rückwärtsdurchlauf pro Strahl und adjungierte Jacobi-Matrix in der Ablage der FD-Jacobi. Formeln, Zweige, Kolbenprojektion und der hybride Ansatz für Zusatzzeilen: [numerik.md, „Rückwärtsdurchlauf (M5-T2)“](numerik.md#rückwärtsdurchlauf-m5-t2).

```cpp
// include/mikroptim/trace/adjoint_backward.hpp (header-only, MKO_HD, über R templatisiert)
template <class R> struct AdjointSeed { R opl, penalty; };          // Spur Φ = ō OPL + π̄ π
template <class R> struct RayAdjoint { Vec3<R> p, d; };
template <class R> struct SurfaceAdjoint { R thickness, curvature, n_before, n_after; };
void refracting_step_backward(p, d, t, c, n_vor, n_nach, branch, seed, RayAdjoint&, SurfaceAdjoint&);
void ghost_step_backward(p, d, t, GhostPlateParams, n_vor, n_nach, branch, seed, adj, out, coeff_sink);
void reference_step_backward(p, d, M, n_B, branch, seed, adj, Vec3& M_adj, R& n_B_adj);
R point_source_backward(d0, n0, adj);
void backward_ray(sys, l, const Record&, bool reached_image, M, bool point_source, seed, Sink&);
//   Sink: curvature(s, v), thickness(s, v), index(s, v), ghost(g, k, v) – jeder Parameter ≤ 1-mal
// include/mikroptim/merit/ray_jacobian.hpp (header-only, MKO_HD)
struct AdjointParameterMap { int variables, surfaces, wavelengths, ghost_plates, ghost_terms;
    const int *curvature /*[S]*/, *thickness /*[S]*/, *glass /*[S·3]*/, *ghost /*[G·K]*/;
    const double* glass_slope; /*[L·3] = (1, ω_l, ω_l²)*/ };          // Spalte p oder −1
template <class R> struct JacobianRowWriter { AdjointParameterMap map; int wavelength;
    double* column0; int64_t rows, row; R factor; /* Sink: J[p·rows + row] = factor·v */ };
void ray_gradient_rows(table, table_l, rec, status, soft_first, soft_mode, M, point_source,
                       opl_row, penalty_row);   // OPL-Spur (in_piston), Strafspur (weiches Ereignis)
// include/mikroptim/optim/adjoint_jacobian.hpp (Host, Mikroptim::optim)
class AdjointParameters { AdjointParameters(const VariableSet&, const SystemBatchHost&);
                          AdjointParameterMap view() const; /* Vektoren für den Upload */ };
struct AdjointJacobianOptions { ResidualSettings quadrature; ExtraResiduals extra;
                                TraceExtraResiduals trace_extra; int extra_threads; };
struct AdjointJacobian { JacobianShape shape; std::vector<double> jacobian /*[runs][P][R]*/,
    residuals /*[runs][R]*/; std::vector<uint8_t> status /*[runs][rays]*/; /* at, column, … */ };
SystemBatchHost make_adjoint_batch(const VariableSet&, const SystemBatchHost& base, x0);
ExtraRowsJacobian extra_rows_cpu(set, base, x0, options);           // Zusatzzeilen per FD
ExtraRowsJacobian extra_rows_from_fd(const FdJacobianBatch&, extra, E);
void insert_extra_rows(const ExtraRowsJacobian&, AdjointJacobian&);
FdJacobianOptions fd_options_of(const AdjointJacobianOptions&);
AdjointJacobian adjoint_jacobian_cpu(set, base, x0, options = {});  // CPU-Referenz
// include/mikroptim/cuda/adjoint_jacobian.hpp (reines C++, Mikroptim::cuda)
class DeviceAdjointJacobian {   // RAII, nur verschiebbar; J bleibt auf dem Gerät
    void compute(TraceEngine&, const VariableSet&, const SystemBatchHost& base, x0,
                 const AdjointJacobianOptions& = {}, Precision = Fp64);
    AdjointJacobian download() const; void release() noexcept; bool ready() const;
    const double* device_jacobian() const; const double* device_residuals() const;
    const uint8_t* device_status() const; const JacobianShape& shape() const; };
AdjointJacobian adjoint_jacobian_gpu(TraceEngine&, set, base, x0, options = {}, Precision = Fp64);
AdjointKernelInfo adjoint_jacobian_kernel_info(int capacity, Precision = Fp64, int device = -1);
```

**Ablage** wie die FD-Jacobi (optimierung.md, „Finite-Differenzen-Jacobi“): pro Lauf i eine R × P-Matrix spaltenweise, J_i(r, p) = `jacobian[(i·P + p)·R + r]`, r₀_i(r) = `residuals[i·R + r]`, R = R_v + P_v + E (`JacobianShape`), auf dem Gerät in derselben Ablage (`device_jacobian`, `device_residuals`). Ein Lauf ist eine Variante (am Punkt x₀ des Laufs, `make_adjoint_batch`: Vorlage + `VariableSet::apply`, bitgleich zur Variante x₀ des FD-Batches); die Zeilen r₀ sind auf der CPU bitgleich zu denen der FD-Jacobi (`residuals_cpu`). M5-T3 tauscht `DeviceFdJacobian` gegen `DeviceAdjointJacobian` (gleiche Methode `compute(engine, set, base, x0, …)`, gleiche Gerätezeiger); `DeviceNormalEquations::compute` nimmt bisher nur `DeviceFdJacobian` und braucht dafür eine Überladung (nur J, r₀ und die Form).

**CPU-Referenz** `adjoint_jacobian_cpu`: `residuals_cpu` für r₀ und Status, dann pro Strahl `batch_ray_opl` mit `AdjointRecord<double>` (Status bitgleich, sonst `std::logic_error`) und `ray_gradient_rows` in eine mit 0 vorbelegte Matrix, pro Tupel und Variable die Projektion (Summen aufsteigend in k), zuletzt die Zusatzzeilen (`extra_rows_cpu`). Läuft in einem Thread (CI-Referenz; Parallelisierung über Läufe wäre trivial, M5-T3).

**Kernel `adjoint_jacobian_kernel<R, C>`** (`src/cuda/adjoint_jacobian.cu`, mit `--fmad=false` wie `adjoint.cu`; R = double oder df64, C = 16, 32, 64, sechs Instanzen): ein Block pro (Lauf, Feld, Wellenlänge), 128 Threads über die Stützstellen, Flächentabelle und Geisterdaten im Shared Memory (dieselben Bausteine aus `wavefront_block.hpp`).

1. Pro Stützstelle: `batch_ray_opl` mit `AdjointRecord<R, C>` im Local Memory, OPL vorläufig auf dem Residuenplatz, Status, Strafresiduum; danach sofort `ray_gradient_rows`: OPL-Spur (Strahl im Kolben) schreibt ∂OPL/∂x roh in die Wellenfrontzeile, Strafspur (weiches Ereignis) schreibt s_p √W ∂π/∂x in die Strafzeile. Jede Spalte erhält pro Strahl höchstens einen Schreibzugriff; J ist vorher per `cudaMemset` 0. Threads eines Warps schreiben benachbarte Zeilen derselben Spalte (koalesziert).
2. Kolben und W_ok (Baumreduktion wie RESIDUALS), Wellenfrontresiduen.
3. Projektion: für p = 0 … P − 1 Teilsumme Σ W_k J(k, p) der eigenen Strahlen (aufsteigend), Warp-Shuffle, Warps in fester Reihenfolge, Mittel per Shared Memory, dann J(k, p) = s √W_k (J(k, p) − ḡ_p) bzw. 0 (in FP64, auch für R = df64). Keine Atomics: zwei Läufe sind bitgleich, ein Lauf ist unabhängig von seiner Lage im Batch bitgleich (Test `CudaAdjointJacobian.RepeatedRunsAndBatchPositionBitIdentical`).

Die Zusatzzeilen rechnet der Host (bei trace-abhängigen Residuen mit den Extrema eines MERIT-Laufs der Engine über den FD-Batch) und kopiert sie mit `cudaMemcpy2D` in die Zeilen R_v + P_v … R − 1 jeder Spalte und von r₀. Ein `cudaMalloc` für J, r₀, Status, Feldablage und Spaltenzuordnung (wiederverwendet), der Batch der Läufe in einem eigenen `DeviceSystemBatch` auf dem Gerät der Engine; jeder CUDA-Aufruf geprüft, nach dem Start `cudaGetLastError()`. J muss ganz in den Gerätespeicher passen (Läufe pro Aufruf wählt M5-T3); Tranchen und die Einbindung als `TraceMode` der Engine folgen in M5-T3.

**Ressourcen** (`adjoint_jacobian_kernel_info`, RTX 3090, sm_86, CUDA 12.9, Test `CudaAdjointJacobian.KernelResources`):

| Instanz | Register | Stack (Local Memory) | statischer Shared Memory |
|---|---:|---:|---:|
| double, C = 16 / 32 / 64 | 168 | 976 / 1.760 / 3.328 B | 2.000 B |
| df64, C = 16 / 32 / 64 | 158 | 848 / 1.632 / 3.200 B | 2.000 B |

Die Aufzeichnung dominiert den Stack (wie im Grundgerüst, M5-T1); der Rückwärtsdurchlauf selbst braucht keine Felder der Länge P, weil jede Ableitung sofort in ihre Spalte geschrieben wird. Die 168 Register (Grundgerüst 96) begrenzen die Belegung auf drei Blöcke pro SM; Tuning ist Sache von M5-T3/M7.

**Erste Laufzeitzahl** (kein Benchmark; Test `CudaAdjointJacobian.FirstTimingAgainstFd`, RTX 3090, FP64): Mehrlinser, P = 13, 200 Läufe (680 Strahlen, J = 28 MB) adjungiert 3,3 ms gegen 24,4 ms für die FD-Jacobi (27 Traces pro Lauf), Faktor 7,3. Die FD-Kosten wachsen mit 2P + 1, die adjungierten mit P nur über Projektion und Schreiben von J; die Abnahme „mindestens 10× für P ≥ 60“ ist M5-T3.

**Abnahme** (Tests `AdjointLocal.*`, `AdjointRay.*`, `AdjointJacobian.*`, `CudaAdjointJacobian.*`; Messwerte und Toleranzen in [numerik.md](numerik.md#rückwärtsdurchlauf-m5-t2), Abschnitt „Ergebnisse“).

## Kernel-Modi

| Modus | Ausgabe | Zweck | Meilenstein |
|---|---|---|---|
| `MERIT` | gewichtete Quadratsumme pro Variante | Liniensuche, Multistart-Screening | M2 |
| `RESIDUALS` | Residuenvektor pro Variante | finite Differenzen, LM | M2 |
| `ADJOINT` | Residuen plus Gradientenzeilen; Grundgerüst seit M5-T1: Residuen wie RESIDUALS plus Aufzeichnung pro Strahl (`DeviceAdjointForward`, oben); seit M5-T2 die Jacobi-Matrix in der Ablage der FD-Jacobi (`DeviceAdjointJacobian`, oben) | analytische Jacobi-Matrix | M5 |
| `EVAL` | OPD, Spot und Status pro Rasterpunkt, Feldstrahlen; Kennzahlen auf dem Host (`TraceEngine::evaluate`, oben) | Endbewertung, Eingabe der PSF und MTF (M8-T2, `PsfEngine`; vektoriell M8-T5, `VectorPsfEngine`), später Tolerierung | M8-T1 |

Zusätzlich liefern die Optimierungsmodi optional pro Variante und Fläche die maximale Strahlhöhe und den maximalen Einfallswinkel (Max-Reduktion) für mechanische Nebenbedingungen. Die FP32-Vorfilterung (M7-T2) und EVAL (M8-T1) sind bewusst keine Werte von `TraceMode`, sondern eigene Funktionen der Engine.

## Reduktionen

Deterministisch: Baumreduktion im Block mit `__shfl_down_sync` und Shared Memory, Ergebnis pro Block in den globalen Speicher, danach Summation in fester Reihenfolge (zweiter Kernel oder Host). Keine Gleitkomma-Atomics für Ergebniswerte (ADR-0007).

## Speicherbedarf

Residuenmatrix für finite Differenzen: Läufe × (P+1) × Residuen × 8 Byte. Beispiel 1.000 × 91 × 2.500 × 8 B ≈ 1,8 GB. Die Batchgröße wird zur Laufzeit aus `cudaMemGetInfo` bestimmt; reicht der Speicher nicht, werden Läufe in Tranchen verarbeitet.

### Abschätzung (M2-T1, `include/mikroptim/batch/batch_memory.hpp`)

Jedes Feld wird auf A = `kBatchAlignment` = 256 Byte aufgerundet, ⌈x⌉_A = A·⌈x/A⌉. Mit B = F·L Blöcken pro Variante und R_v = Strahlen pro Variante:

- **Batch** (`make_batch_layout(topology, V).total_bytes`):
  ⌈S⌉ + ⌈4G⌉ + ⌈8G⌉ + 2·⌈8L⌉ + ⌈8F⌉ + 2·⌈8VS⌉ + ⌈8VSL⌉ + ⌈8V⌉ + 2·⌈8VF⌉ + ⌈8VGK⌉ + ⌈8VGL⌉ + ⌈8VS⌉ + ⌈8VG⌉ (jeweils ⌈·⌉_A, leere Felder 0; die letzten beiden sind die Brechzahlen bei λ_ref, ab M3-T3).
- **Ausgaben** (`make_output_layout(topology, V, OutputRequest, extra)`), jede einzeln abschaltbar:
  Residuen ⌈8V(R_v + P_v + E)⌉ (P_v = R_v Strafresiduen im weichen Modus, M4-T2, sonst 0), Status pro Strahl ⌈V·R_v⌉ (1 Byte), rohe OPD pro Strahl ⌈8V·R_v⌉ (nur auf Anforderung, `opd`, ab M2-T2), Merit ⌈8V⌉ + Teilsummen pro Block ⌈8VB⌉ + versagte Strahlen ⌈8V⌉ (int64) + pro Block ⌈4VB⌉ (int32, ab M2-T3, `failed_rays`, `failed_rays_partial`), dasselbe für die hart versagten (ab M4-T2, `hard_failed_rays`, `hard_failed_rays_partial`), Max-Reduktionen (Strahlhöhe, Einfallswinkel) ⌈16VS⌉ + pro Block ⌈16VBS⌉, Arbeitsspeicher des Aufrufers ⌈V·extra⌉. E = weitere Residuen pro Variante (Paraxial, Strafen).
- **Strahlen pro Variante:** `rays_per_variant(F, L, N·M, F_Achse, N)` = L·(F_Achse·N + (F − F_Achse)·N·M), wenn die F_Achse Achsfelder (an beliebiger Stelle) die Achsquadratur (M = 1) nutzen (`sample_count` in `include/mikroptim/pupil/quadrature.hpp`); gleich `ResidualLayout::rays_per_variant()`.
- `estimate_device_bytes` = Batch + Ausgaben. Der Kernel `RESIDUALS` legt seine Ausgaben nach diesem Schema an (Residuen, Status, optional OPD und Max-Reduktionen), der Kernel `MERIT` ebenso (Merit, Teilsummen, versagte Strahlen, optional Max-Reduktionen; keine Residuen, kein Status); `extra_merit` von `MERIT` belegt 8 Byte pro Variante (in der TraceEngine als `extra_bytes_per_variant`).

Größte Batchgröße: `max_variants(free, topology, outputs, reserve, extra)` sucht (Bisektion, die Abschätzung ist monoton in V) das größte V mit Abschätzung ≤ free − max(`min_bytes`, `fraction`·free); Standardreserve max(256 MiB, 5 %) für Kontext, Kernel-Stacks, Local Memory (`TraceRecord`) und Fragmentierung. Zusätzlich gilt V·F·L ≤ 2³¹ − 1 (größtes Grid). Auf dem Gerät: `max_variants_on_device` mit `device_memory_info().free_bytes`; nie fest verdrahten (AGENTS.md, Abschnitt 7). Die TraceEngine bestimmt daraus die Tranchengröße (zusätzlich eigene Puffer, künstliche Grenze und Feldablage, Abschnitt [Host-API TraceEngine](#host-api-traceengine-m2-t4)).

Beispiel (Test `BatchMemory.MaxVariantsRespectsLimit`): Zielsystem-Topologie S = 30, L = 7, F = 5 (ein Achsfeld), G = 6, K = 8, N = 10, M = 8 ergibt 2.310 Strahlen und ca. 42,0 kB pro Variante mit allen Ausgaben, also ca. 391.000 Varianten bei 16 GiB und 586.000 bei 24 GiB freiem Speicher (ohne Strafresiduen; im weichen Modus kommen im Modus RESIDUALS 8·R_v = 18,5 kB pro Variante dazu, MERIT braucht keine). Der Batch selbst ist klein (ca. 3,3 kB pro Variante); die Ausgaben dominieren.

## Host-API TraceEngine (M2-T4)

Einziger Einstieg für Programme, die auf der GPU tracen (CLI ab M2-T5, Optimierer ab M3, Python ab M9, GPU-Tests). Header `include/mikroptim/cuda/trace_engine.hpp` (reines C++, ohne nvcc einbindbar; der CPU-Build kompiliert ihn im Test `TraceEngineHeader.*`), Implementierung `src/cuda/trace_engine.cu` in `Mikroptim::cuda`. Die Engine gibt es nur in Builds mit `MIKROPTIM_ENABLE_CUDA`; Aufrufer schützen ihren GPU-Pfad mit `#ifdef MIKROPTIM_HAS_CUDA` (wie `mikroptim --devices`). Eine Attrappe, die ohne CUDA wirft, gibt es bewusst nicht: Ein fehlender GPU-Pfad fällt so schon beim Bauen auf.

```cpp
namespace mko::cuda {
enum class TraceMode { Residuals, Merit /* M2-T3 */ };
struct EngineOptions { std::optional<size_t> memory_limit_bytes; MemoryReserve reserve; };
struct TraceSettings { ResidualSettings quadrature;   // N × M, Strahlversagen und w_fail (M4-T2)
                       bool opd = false, status = true; int64_t extra_residuals = 0;   // nur RESIDUALS
                       bool extrema = false;                     // beide Modi (M2-T3)
                       std::span<const double> extra_merit;     // nur MERIT: leer oder [V]
                       Precision precision = Precision::Fp64; }; // beide Modi: Fp64 | Df64 (M7-T1)
struct TraceRunInfo { int variants, tranches, tranche_variants; size_t budget_bytes, device_bytes; };
struct DeviceTranche { int first, count; int64_t rays_per_variant, penalty_rows, stride;   // M3-T2, M4-T2
                       const double* residuals; const uint8_t* status; const double *opd, *extrema; };
using DeviceTrancheFn = std::function<void(const DeviceTranche&)>;
struct TraceOutput { TraceMode mode; ResidualLayout layout;
                     ResidualResult residuals;   // Modus Residuals (sonst leer), Extrema in .extrema
                     MeritResult merit;          // Modus Merit (sonst leer), Extrema in .extrema
                     TraceRunInfo info; };
class TraceEngine {                                   // nur verschiebbar
public:
    explicit TraceEngine(int device, const EngineOptions& = {});
    void run(const SystemBatchHost&, TraceMode, const TraceSettings&, TraceOutput& out);
    TraceOutput run(const SystemBatchHost&, TraceMode, const TraceSettings& = {});
    int max_variants(const BatchTopology&, TraceMode, const TraceSettings& = {}) const;
    // Modus Residuals ohne Download (M3-T2): Rückruf pro Tranche mit Gerätezeigern,
    // Tranchen als Vielfache von granularity
    TraceRunInfo run_on_device(const SystemBatchHost&, const TraceSettings&, int granularity,
                               const DeviceTrancheFn& consume);
    // FP32-Vorfilterung (M7-T2), Abschnitt "FP32-Vorfilterung"
    TraceRunInfo prefilter_variants(const SystemBatchHost&, const PrefilterSettings&, PrefilterResult&);
    // Modus EVAL (M8-T1), Abschnitt "Modus EVAL"
    TraceRunInfo evaluate(const SystemBatchHost&, const EvalSettings&, EvalResult&);
    void release() noexcept;
    int device() const; const std::string& device_name() const; size_t device_bytes() const;
};
}
// batch_residuals.hpp (Host, auch ohne CUDA): RMS pro Tupel aus OPD und Status
std::vector<TupleWavefront> tuple_wavefronts(const ResidualResult&, const ResidualLayout&, int v);
```

Gegenüber der ursprünglichen Skizze (`upload(batch)`, dann `run(mode, settings, out)`) nimmt `run` den Host-Batch direkt: Die Tranchengröße hängt von den Einstellungen ab (OPD, weitere Residuen), ein getrennter Upload müsste also entweder den ganzen Batch auf dem Gerät halten oder eine Kopie auf dem Host. Der Upload ist gegen den Kernel klein (10.000 Varianten des Mehrlinsers: ca. 33 ms pro Lauf mit Upload und Download gegen 26 ms für den Kernel allein, Test `CudaResiduals.LargeBatch`), und der Optimierer baut ohnehin in jeder Iteration einen neuen Batch.

**Ablauf von `run`:**

1. **Prüfen** (Host, vor jeder Gerätearbeit): Modus bekannt, `extra_residuals` ≥ 0, `ResidualLayout(topology, quadrature)` (verlangt festgelegte Pupillenarten und eine gültige Quadratur), `validate_values()` (endliche Werte, Brechzahlen, Gewichte, Punktquellen-NA, `check_pupil_modes`); im Modus MERIT außerdem `extra_merit` leer oder V Werte lang und höchstens `merit_max_samples(extrema)` Stützstellen pro Feld (fragt nur Kernelattribute ab, legt keinen Speicher an).
2. **Host-Ausgabe** anlegen: RESIDUALS: `residuals` [V·(R_v + E)], `opd` [V·R_v] nur mit `opd`, `status` [V·R_v] nur mit `status`, `residuals.extrema` [V·S·2] nur mit `extrema`; Ablage wie `DeviceResiduals::download` ([Kernel RESIDUALS](#kernel-residuals-m2-t2)). MERIT: `merit.merit`, `merit.failed` [V], `merit.extrema` [V·S·2] nur mit `extrema` ([Kernel MERIT](#kernel-merit-m2-t3)). Das Ergebnis des jeweils anderen Modus ist leer. `out` ist wiederverwendbar (die Vektoren behalten ihre Kapazität).
3. **Tranchengröße** T = `max_variants(topology, mode, settings)`: verfügbar A = min(frei + eigene Puffer, `memory_limit_bytes`), nutzbar U = `usable_bytes(A, reserve)`, dann das größte V mit `estimate_device_bytes(V)` + ⌈16·F⌉_A ≤ U (Bisektion `max_variants` aus batch_memory mit Reserve 0; ⌈16·F⌉_A ist die Feldablage `ResidualFieldPlan[F]`, die pro Lauf unabhängig von V anfällt). Die Ausgaben sind im Modus RESIDUALS die von `DeviceResiduals` (Residuen, Status – der Kernel braucht ihn immer –, OPD und Max-Reduktionen nur auf Anforderung; keine Merit-Felder), im Modus MERIT die von `DeviceMerit` (Merit, Teilsummen, versagte Strahlen, Max-Reduktionen auf Anforderung, `extra_merit` als 8 Byte pro Variante). Eigene Puffer zählen als verfügbar, weil ein Lauf sie wiederverwendet oder freigibt. Der Ausgabepuffer des anderen Modus bleibt beim Wechsel RESIDUALS ↔ MERIT erhalten (Liniensuche im LM) und zählt dann nicht als verfügbar; nur wenn die Tranche dadurch kleiner als V und kleiner als ohne ihn wäre, wird er freigegeben. T ≥ 1, sonst `CudaError` mit verfügbarem, nutzbarem und benötigtem Speicher. Die Gridgrenze V·F·L ≤ 2³¹ − 1 steckt schon in `max_variants`.
4. **Puffer:** Reichen die vorhandenen Puffer (`DeviceSystemBatch` und `DeviceResiduals` bzw. `DeviceMerit`) nicht für T Varianten, werden beide vorher freigegeben, damit alter und neuer Puffer nie gleichzeitig belegt sind. Danach werden sie über alle Tranchen und folgende Läufe wiederverwendet (die letzte, kleinere Tranche nutzt denselben Puffer).
5. **Tranchen** [first, first + T): `DeviceSystemBatch::upload(host, first, count)`, `DeviceResiduals::run` bzw. `DeviceMerit::run` (mit `extra_merit.subspan(first, count)`), dann je ein `cudaMemcpy` pro Ausgabe (Residuen, Status, OPD bzw. Merit, versagte Strahlen; Max-Reduktionen) von den Gerätezeigern an die Stelle `first` der Host-Vektoren. Synchron; bei der Rückkehr liegen alle Daten auf dem Host.

**Bitgleichheit:** Der Kernel rechnet jede Variante unabhängig von ihrer Lage im Batch (Summationsreihenfolge nur abhängig von `kResidualThreads`), daher sind die Ergebnisse jeder Variante unabhängig von der Aufteilung in Tranchen bitgleich (Test `TraceEngine.TranchesMatchSingleRun`: 700 Varianten mit einer Grenze für genau 200, also 4 Tranchen, bitgleich zum Lauf in einer Tranche; `TraceEngine.OneVariantPerTranche`; für MERIT mit Max-Reduktionen und `extra_merit` sowie RESIDUALS mit Max-Reduktionen `TraceEngine.MeritTranchesMatchSingleRun`). Dasselbe gilt für die Verteilung auf mehrere GPUs ([Mehrere GPUs (M7-T3)](#mehrere-gpus-m7-t3)).

**Künstliche Grenze** (`EngineOptions::memory_limit_bytes`): begrenzt den Speicher, den die Engine als verfügbar ansieht (Tests der Tranchen; Teilen einer GPU mit anderen Programmen). Die Reserve gilt zusätzlich; für exakte Tests `MemoryReserve{0, 0.0}`. Mit Grenze G und Reserve 0 belegt eine volle Tranche höchstens G Byte (`device_bytes()`), im Test genau G.

**Gerätewahl:** `TraceEngine(device)` verlangt 0 ≤ device < `device_count()`; ohne Gerät `CudaError`, ungültiger Index oder ein Gerät unter sm_60 (keine passende Kernel-Version, ADR-0002) `std::invalid_argument`. Der Konstruktor legt noch keinen Kontext und keinen Speicher an. Jeder Aufruf setzt das Gerät der Engine und stellt danach das aktuelle Gerät des Aufrufers wieder her (`DeviceGuard`).

**Fehler:** CUDA-Fehler und Speichermangel `CudaError`, ungültige Argumente (Index, Einstellungen, Werte im Batch, im Modus MERIT Länge von `extra_merit` und zu viele Stützstellen pro Feld) `std::invalid_argument`, bevor das Gerät angefasst wird. Nach einer Ausnahme bleibt die Engine benutzbar, der Inhalt von `out` ist unbestimmt. Ein leerer Batch (V = 0) liefert leere Vektoren und `tranches = 0` ohne Gerätearbeit.

**Thread-Sicherheit:** Kein globaler Zustand; mehrere Engines (auch auf demselben Gerät, Test `TraceEngine.EnginesAreIndependent`) sind unabhängig. Eine Engine darf nicht gleichzeitig aus mehreren Threads benutzt werden. Für mehrere GPUs bekommt jede GPU eine eigene Engine in einem eigenen Host-Thread (`cudaSetDevice` gilt pro Thread; der Default-Stream und `cudaDeviceSynchronize` blockieren nur das eigene Gerät); das übernimmt `MultiTraceEngine` ([Mehrere GPUs (M7-T3)](#mehrere-gpus-m7-t3)).

**Ohne Download (`run_on_device`, M3-T2):** Für Folgekernel, deren Eingabe der Residuenpuffer ist (Differenzen der FD-Jacobi), läuft der Modus RESIDUALS wie oben, aber nach dem Kernel jeder Tranche ruft die Engine einen Rückruf mit den Gerätezeigern der Tranche (`DeviceTranche`: erste Variante, Anzahl, Residuen mit Stride R_v + E, Status, optional OPD und Max-Reduktionen) auf, statt herunterzuladen. Der Rückruf läuft mit dem Gerät der Engine als aktuellem Gerät; Arbeit, die er im Default-Stream startet, ist vor dem Upload der nächsten Tranche beendet. Die Tranchengröße wird auf ein Vielfaches von `granularity` abgerundet (zusammengehörige Varianten, z. B. die 2P + 1 eines FD-Laufs, liegen in derselben Tranche; nur die letzte Tranche ist kleiner, wenn V kein Vielfaches ist); passt nicht einmal `granularity` Varianten, `CudaError`. Tranchenrechnung und Puffer teilen sich `run` und `run_on_device` (`plan_tranches`). Speicher, den der Aufrufer vorher selbst belegt (z. B. J), sieht die Engine über `cudaMemGetInfo` als belegt. Test `TraceEngineOnDevice.TranchesAreMultiplesOfGranularity`.

**Modus MERIT (M2-T3):** eigener Ausgabepuffer `DeviceMerit` neben `DeviceResiduals`, eigene `OutputRequest` in der Tranchenrechnung, Ergebnis in `TraceOutput::merit`; der Ablauf (Prüfen, Tranchen, Download an die Stelle `first`) ist derselbe wie bei RESIDUALS.

**Auswertung für die CLI (M2-T5):** Mit `opd = true` liefert `tuple_wavefronts(out.residuals, out.layout, v)` pro Feld und Wellenlänge RMS [mm], Σ W der gültigen Strahlen und die Zahl versagter Strahlen, Ablage [f·L + l] wie `WavefrontResult::maps`. Für das Ergebnis von `residuals_cpu` ist das bitgleich zu `evaluate_wavefront` (Test `ResidualsCpu.TupleWavefrontsMatchEvaluateWavefront`), für die GPU innerhalb 1,6·10⁻¹⁵ mm (Test `TraceEngine.TupleWavefrontsMatchEvaluateWavefront`). Die Umrechnung in Wellen (λ_ref) und die Paraxialdaten bleiben in der CLI.

**Abgrenzung der Tests:** Alle GPU-Tests, die die Kernel `RESIDUALS` und `MERIT` fachlich prüfen (CPU-Abgleich, Determinismus, Grenzfälle, Durchsatz), laufen über die Engine. Direkt auf die Bausteine greifen nur Tests zu, die deren eigenen Vertrag prüfen: `CudaResiduals.DeviceBufferReuse` und `CudaResiduals.InvalidUseThrows` (`DeviceResiduals`), `CudaMerit.DeviceBufferReuse` und `CudaMerit.InvalidUseThrows` (`DeviceMerit`), `test_cuda_system_batch.cpp` (Upload/Download des Batches) und die Prüfkernel aus M1 (`test_cuda_trace.cpp`, `test_cuda_quadrature.cpp`, `test_cuda_dispersion.cpp`, `test_cuda_smoke.cpp`).

## Optimierer: Variablen und Variantengenerator (M3-T1)

Statische Host-Bibliothek `mikroptim_optim` (Alias `Mikroptim::optim`), linkt `model` und `batch`; ohne CUDA. Formeln und Festlegungen: [optimierung.md](optimierung.md#variablenabbildung-ab-m3-t1).

```cpp
// variable_set.hpp
struct Variable { VariableType type; int file_surface, batch_surface; bool split; double offset;
                  double lower, upper, typical; std::string name; }; // name: "c[3]", "t[2]", "w"
SurfaceMap checked_surface_map(const SystemDescription&, const BatchTopology&); // make_surface_map + Prüfung
class VariableSet {
    VariableSet(const SystemDescription&, const BatchTopology&);
    int size() const; const Variable& operator[](int) const;
    std::span<const double> lower() const, upper() const, typical() const;
    std::vector<double> extract(const SystemBatchHost&, int v) const;
    void apply(SystemBatchHost&, int v, std::span<const double> x) const;
    std::vector<double> extract(const SystemDescription&) const;
    void apply_to_description(SystemDescription&, std::span<const double> x) const;
    bool feasible(std::span<const double>) const; void clamp_to_bounds(std::span<double>) const;
};
// variant_generator.hpp
std::vector<double> perturbed_vector(const VariableSet&, std::span<const double> x0,
                                     const PerturbationOptions&, std::uint64_t j);
void write_perturbed_variants(set, base, base_variant, x0, options, SystemBatchHost& out, int first, int count);
SystemBatchHost make_perturbed_batch(set, base, base_variant, x0, int variants, options);
FdPattern fd_pattern(const VariableSet&, std::span<const double> x0, FdScheme);
FdPattern write_fd_variants(set, base, base_variant, x0, FdScheme, SystemBatchHost& out, int first);
FdBatch make_fd_batch(set, base, base_variant, x0, FdScheme = FdScheme::Central);
```

Der Generator arbeitet nur auf dem Host-Batch (`SystemBatchHost`); die TraceEngine lädt ihn wie jeden anderen Batch hoch. Für den gebündelten LM (M3-T4) liegen die FD-Muster mehrerer Läufe hintereinander in einem Batch (`write_fd_variants` mit `first` = Lauf · (2P + 1)).

## Optimierer: Zusatzresiduen und paraxiale Nebenbedingungen (M3-T3)

Ebenfalls in `Mikroptim::optim` (linkt zusätzlich `paraxial` und privat `Threads::Threads`). Formeln und Festlegungen: [optimierung.md, Abschnitte 2.1 und 2.2](optimierung.md#21-zusatzresiduen-schnittstelle-ab-m3-t3); Schema: [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3); Modell `mko::Constraints` in `Mikroptim::model` (`include/mikroptim/model/constraints.hpp`).

```cpp
// extra_residuals.hpp: Schnittstelle zu FD-Jacobi (M3-T2) und LM (M3-T4)
using ExtraResidualFn = std::function<void(const SystemBatchHost&, int v, std::span<double> out)>;
struct ExtraResiduals { int count = 0; ExtraResidualFn evaluate; };
// paraxial_residuals.hpp
ParaxialData batch_paraxial(const SystemBatchHost&, int v);              // bei λ_ref, Geisterplatten
std::vector<ParaxialData> batch_paraxial_all(const SystemBatchHost&, int threads = 0);
ParaxialConstraints make_paraxial_constraints(const SystemDescription&);
ExtraResiduals make_paraxial_residuals(ParaxialConstraints);
void evaluate_extra_residuals(const ExtraResiduals&, const SystemBatchHost&, std::span<double> out,
                              int first = 0, int count = -1, int threads = 0);
```

Der Paraxialtrace läuft auf dem Host (pro Variante ein y-nu-Trace über höchstens 64 Flächen, Felder auf dem Stack, keine Allokation), verteilt auf `std::thread`s in fester Aufteilung (intern `src/optim/parallel_variants.hpp`); Ergebnisse sind für jede Threadzahl bitgleich. Ein Paraxialtrace auf der GPU lohnt erst, wenn er neben dem Strahltrace ins Gewicht fällt (der Kern `ynu_trace.hpp` ist dafür schon `MKO_HD`).

## Optimierer: Trace-abhängige Zusatzresiduen und mechanische Nebenbedingungen (M4-T3)

Formeln, Vorzeichen und Glattheit: [optimierung.md, Abschnitte 2.1 und 2.5](optimierung.md#25-mechanische-nebenbedingungen-ab-m4-t3); Schema: [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3). Beide Header in `Mikroptim::optim`.

`ExtraResiduals` sieht nur den Batch einer Variante, keinen Trace. Strahlhöhen und Einfallswinkel entstehen aber im Trace; sie kommen aus den Max-Reduktionen (`SurfaceExtremaResult`, [Max-Reduktionen](#max-reduktionen-m2-t3)) **derselben** Auswertung, die auch die Strahlresiduen liefert. Dafür gibt es eine zweite Art Zusatzresiduen:

```cpp
// trace_extra_residuals.hpp
struct VariantExtrema { int surfaces; const double* values;          // [S][2] einer Variante
                        double max_height(int s) const; double min_cos_incidence(int s) const; };
using TraceExtraResidualFn = std::function<void(const SystemBatchHost&, int v,
                                                const VariantExtrema&, std::span<double> out)>;
struct TraceExtraResiduals { int count = 0; TraceExtraResidualFn evaluate; };
void evaluate_trace_extra_residuals(const TraceExtraResiduals&, const SystemBatchHost&,
                                    const SurfaceExtremaResult&, int extrema_first,
                                    std::span<double> out, int first = 0, int count = -1, int threads = 0);
std::vector<double> trace_extra_merit_values(const TraceExtraResiduals&, const SystemBatchHost&,
                                             const SurfaceExtremaResult&, int threads = 0);
// mechanical_constraints.hpp
double sag(double c, double h); double edge_thickness(double t, double c1, double c2, double h);
MechanicalConstraints make_mechanical_constraints(const SystemDescription&);
struct MechanicalResiduals { ExtraResiduals geometric; TraceExtraResiduals traced; … };
MechanicalResiduals make_mechanical_residuals(const MechanicalConstraints&);
```

Im Residuenvektor einer Variante stehen sie hinter allen übrigen Zusatzresiduen: R = R_v + E + E_t (`FdJacobianOptions::trace_extra`, `extra_count()` = E + E_t; `JacobianShape::extra_rows` zählt beide). Wege der Extrema:

| Pfad | Extrema | Auswertung |
|---|---|---|
| FD-Jacobi CPU (`fd_jacobian_cpu`, `CpuLmBackend::normal_equations`) | `surface_extrema_cpu` jeder FD-Variante (zweiter Trace auf dem Host) | `fd_extra_residuals`: Zusatzresiduen, dahinter die trace-abhängigen, `join_extra_rows` |
| FD-Jacobi GPU (`DeviceFdJacobian::compute`) | Engine im Modus RESIDUALS mit `extrema = true` (`run_on_device`); pro Tranche Download [count][S][2] | Host wertet die Varianten der Tranche aus, lädt ihre Zeilen des Zusatzpuffers [V·E] neu hoch, danach der Differenzkernel (alle Varianten eines Laufs liegen in einer Tranche) |
| Merit CPU (`CpuLmBackend::merit`, `make_cpu_merit`) | `merit_cpu(…, extrema = true)` | F = Strahl- und Strafresiduen + Σ e² + Σ t² (in dieser Reihenfolge) |
| Merit GPU (`GpuLmBackend::merit`) | Engine im Modus MERIT mit `extrema = true` | F = Kernel (Strahlen, Strafen, `extra_merit` = Σ e²) + Σ t² auf dem Host |

Damit gilt F = ‖r‖² auch mit diesen Residuen (bis auf die Summationsreihenfolge, Test `MechanicalLm.MeritIsSumOfSquares`), und jede FD-Variante bekommt die Residuen ihres eigenen Traces. Determinismus: Download und Auswertung pro Variante, keine Reduktion über Varianten; das Ergebnis hängt nicht von der Aufteilung in Tranchen ab (Test `CudaMechanical.FdJacobianMatchesCpuAndTranches`). Kosten: auf der GPU ein Download von 16·S Byte pro Variante und Tranche und die Host-Auswertung; auf der CPU ein zweiter Trace pro Variante (wie `merit_cpu` mit Extrema). Ohne trace-abhängige Residuen (E_t = 0) laufen alle Pfade unverändert ohne Extrema. Die Zuordnung Datei- zu Tabellenflächen (Geisterplatten) übernimmt `make_mechanical_constraints` mit der `SurfaceMap`.

## Optimierer: Finite-Differenzen-Jacobi (M3-T2)

Schablonen, Formeln, Genauigkeit und Ablage: [optimierung.md](optimierung.md#finite-differenzen-jacobi-ab-m3-t2). Host-Teil in `Mikroptim::optim` (`src/optim/fd_jacobian.cpp`), GPU-Teil in `Mikroptim::cuda` (`src/cuda/fd_jacobian.cu`; `Mikroptim::cuda` linkt dafür `Mikroptim::optim`).

```cpp
// Zusatzresiduen: ExtraResiduals und evaluate_extra_residuals aus M3-T3 (oben)
// fd_formula.hpp (header-only, MKO_HD, CPU und Kernel)
enum class FdStencil : uint8_t { Central, OneSided, Truncated, Forward };
template <class R> R fd_derivative(FdStencil, R r0, R ra, R rb, R da, R db);
// fd_jacobian.hpp (Host)
struct JacobianShape { int runs, variables; int64_t ray_rows; int extra_rows; int64_t rows() const; };
struct FdJacobianOptions { ResidualSettings quadrature; FdScheme scheme; ExtraResiduals extra;
                           TraceExtraResiduals trace_extra;   // aus den Extrema (M4-T3)
                           int extra_threads = 0; };   // Host-Threads für die Zusatzresiduen
struct FdJacobianBatch { SystemBatchHost batch; std::vector<FdPattern> patterns; int runs, variants_per_run; };
FdJacobianBatch make_fd_jacobian_batch(const VariableSet&, const SystemBatchHost& base,
                                       std::span<const double> x0 /*[runs·P]*/, FdScheme = Central);
struct FdJacobian { JacobianShape shape; FdScheme scheme;
                    std::vector<double> jacobian /*[runs][P][R]*/, residuals /*[runs][R]*/;
                    std::vector<int64_t> status_changes /*[runs][P]*/; std::vector<FdPattern> patterns;
                    double at(run, row, p); std::span<const double> column(run, p), residual(run); };
void fd_jacobian_from_residuals(const FdJacobianBatch&, const ResidualResult&, std::span<const double> extra,
                                int extra_count, FdJacobian& out);
FdJacobian fd_jacobian_cpu(const VariableSet&, const SystemBatchHost& base, std::span<const double> x0,
                           const FdJacobianOptions& = {});
// cuda/fd_jacobian.hpp (reines C++)
class DeviceFdJacobian {                    // RAII, nur verschiebbar, J bleibt auf dem Gerät
    void compute(TraceEngine&, const VariableSet&, const SystemBatchHost& base,
                 std::span<const double> x0, const FdJacobianOptions& = {});
    FdJacobian download() const; void release() noexcept; bool ready() const;
    const double* device_jacobian() const; const double* device_residuals() const;
    const int64_t* device_status_changes() const; const TraceRunInfo& run_info() const;
    static size_t buffer_bytes(const JacobianShape&, int variants);
};
FdJacobian fd_jacobian_gpu(TraceEngine&, const VariableSet&, const SystemBatchHost& base,
                           std::span<const double> x0, const FdJacobianOptions& = {});
```

**Kernel `fd_difference_kernel`** (pro Tranche der Engine): ein Block pro (Lauf, Spalte p) und ein Block pro Lauf, der r₀ kopiert, also Läufe·(P + 1) Blöcke mit `kFdJacobianThreads` = 128 Threads über die R Zeilen. Jedes Element J_i(r, p) = `fd_derivative`(Schablone, r₀, r_a, r_b, d_a, d_b) hängt nur von drei Residuen ab (Strahlzeilen aus dem Ausgabepuffer der Engine, Zusatzzeilen aus einem eigenen Puffer [V·E]; die trace-abhängigen Zeilen darin ersetzt der Host vor dem Kernel jeder Tranche, M4-T3); keine Gleitkomma-Reduktion. Die Status-Wechsel pro Spalte sind eine ganzzahlige Baumreduktion im Shared Memory (128 · 8 Byte). Schablonen und Schritte liegen als Felder [Läufe·P] im Gerätespeicher.

**Speicher:** `DeviceFdJacobian` belegt einen Puffer (J 8·Läufe·P·R, r₀ 8·Läufe·R, Status-Wechsel 8·Läufe·P, Zusatzresiduen 8·V·E, Schablonen und Schritte 17·Läufe·P Byte, jeweils auf 256 Byte ausgerichtet; `buffer_bytes`). Er wird vor dem Lauf der Engine angelegt, damit deren Tranchenrechnung ihn als belegt sieht, und über Aufrufe wiederverwendet. J muss ganz in den Gerätespeicher passen (M3-T4 wählt die Zahl der Läufe pro Batch); der Residuenpuffer der Engine wird bei Bedarf in Tranchen aus ganzen Läufen aufgeteilt. Beispiel: 1.000 Läufe des Mehrlinsers, J = 70,7 MB.

## Optimierer: Gebündelter Levenberg-Marquardt (M3-T4)

Verfahren, Kriterien und Messungen: [optimierung.md](optimierung.md#verfahren-ab-m3-t4); Abweichung vom Plan (kein cuBLAS/cuSOLVER): [ADR-0010](adr/0010-normalgleichungen-ohne-cublas.md).

- **Steuerlogik** `levenberg_marquardt` (`src/optim/levenberg_marquardt.cpp`, `Mikroptim::optim`): Zustand pro Lauf auf dem Host, Schritte (Cholesky mit aktiver Menge) pro Lauf unabhängig auf Host-Threads. Rechnungen an Punkten über die Schnittstelle `LmBackend` (`normal_equations`, `merit` mit versagten und hart versagten Strahlen seit M4-T2; virtuell, nur Host-Code).
- **Optische Treiber** `optimize_lm_cpu`, `cuda::optimize_lm_gpu`, `cuda::optimize_lm_multi_gpu`: gemeinsamer Ablauf `optimize_lm_rounds` mit einer Fabrik für das Backend einer Runde; im weichen Modus das äußere Strafverfahren (M4-T2): Läufe mit versagten Strahlen starten an ihrem Endpunkt mit w_fail · `fail_weight_factor` neu (eigener Backend-Aufbau pro Runde, Vorlagen per `lm_template_batch`), bis kein Strahl mehr versagt oder `max_fail_weight` erreicht ist; die GPU-Backends summieren ihre Statistik (pro Slot) über die Runden.
- **CPU-Backend** `CpuLmBackend`: FD-Batch wie `fd_jacobian_cpu` (Varianten auf Host-Threads, bitgleich), `normal_equations_host`, `merit_cpu` plus Σ e² (plus Σ t² der trace-abhängigen Zusatzresiduen aus den Extrema desselben Laufs, M4-T3).
- **GPU-Backend** `cuda::GpuLmBackend` (`src/cuda/levenberg_marquardt.cpp`, reines Host-C++ in `Mikroptim::cuda`): Teilstücke aus ganzen Läufen nach dem Speicher (J höchstens die Hälfte des nutzbaren Speichers), pro Teilstück `DeviceFdJacobian::compute` → `DeviceNormalEquations::compute` → Download von A und g; Merit über `TraceEngine::run` im Modus MERIT mit `extra_merit` (mit trace-abhängigen Zusatzresiduen zusätzlich `extrema = true` und Σ t² auf dem Host, M4-T3).
- **Kernel `gram_kernel`** (`src/cuda/normal_equations.cu`): G = [J r₀]ᵀ[J r₀] pro Lauf, (P + 1) × (P + 1) spaltenweise, Abstand (P + 1)². Grid: Läufe · T(T + 1)/2 Blöcke (T = ⌈(P + 1)/16⌉ Kacheln, nur Kachelpaare tp ≤ tq), Block 16 × 16 Threads, Shared Memory 2 · 16 · 17 · 8 = 4.352 Byte. Thread (ty, tx) lädt Zeile r₀ + tx der Spalten 16·tp + ty und 16·tq + ty (zusammenhängend über tx) und führt die FMA-Kette für G(16·tp + ty, 16·tq + tx) über die Zeilen in aufsteigender Reihenfolge fort; die letzte Kachel endet bei R − 1 (kein Polster in der Summe). Keine Reduktion über Threads, keine Atomics; bitgleich zu `normal_equations_host`. Puffer in `DeviceNormalEquations` (8·Läufe·(P + 1)² Byte), wiederverwendet.
- **Host-Threads** (`lm_host_threads`): nur, wenn die Arbeit eines Aufrufs es lohnt (Zusatzresiduen ab 4.096 Varianten pro Thread, CPU-Strahlen ab 2·10⁵ Strahl-Flächen, Schritte ab 2·10⁶ Gleitkommaoperationen); sonst im aufrufenden Thread.

## Systemsynthese: Homotopie-Controller (M6-T3)

Statische Host-Bibliothek `mikroptim_synth` (Alias `Mikroptim::synth`, `src/synth/`), linkt `model`, `batch`, `cpu`, `optim` und privat `glass` und nlohmann/json; ohne CUDA. Verfahren und Festlegungen: [optimierung.md, Abschnitt 7](optimierung.md#festlegungen-der-implementierung-m6-t3).

| Header | Inhalt |
|---|---|
| `include/mikroptim/synth/homotopy.hpp` | `HomotopyPoint` (NA, Feldanteil, Spektrumstufe, Faktoren auf μ, ε und die einseitigen Gewichte), `HomotopyStepControl`, Schnittstelle `HomotopyModel`, `HomotopyOptions`, `run_homotopy`, Protokoll `HomotopyStep`/`HomotopyResult` als Text und JSON |
| `include/mikroptim/synth/optical_homotopy.hpp` | `OpticalHomotopyModel` (realisiert einen Punkt auf der Systembeschreibung, Schritt = ein LM-Lauf, Diagnose, Operatoren A/B mit innerer Homotopie, Zielprüfung am realen System), `OpticalHomotopyOptions`, `HomotopyOptimizer`, `is_one_sided`, `glass_elements` |

Schichtung: Die Steuerlogik (`run_homotopy`, `src/synth/homotopy.cpp`) kennt nur die Schnittstelle `HomotopyModel` und rechnet nichts Optisches; Tests prüfen sie mit synthetischen Modellen. Das optische Modell (`src/synth/optical_homotopy.cpp`) benutzt die Bausteine aus `Mikroptim::optim` (Variablen, Zusatzresiduen, Diagnose, Einfüge-Operatoren, LM). Der LM-Lauf eines Schritts geht über den austauschbaren `HomotopyOptimizer` (Standard `optimize_lm_cpu`); die GPU bindet der Aufrufer ein (`mikroptim synth --gpu`, GPU-Tests: Lambda um `cuda::optimize_lm_gpu` mit einer `TraceEngine`), sodass `Mikroptim::synth` ohne CUDA baut. Diagnose, Zielprüfung und RMS-Bericht rechnen auf der CPU (ein Trace pro Variante bzw. Platte). Der Zustand ist eine `SystemDescription`; jedes `solve` baut Brechzahltabelle, Batch und `VariableSet` neu (Kataloge aus dem Start, `rebuild_loaded_system`, ohne Dateizugriff), weil Apertur, Spektrum und Topologie sich zwischen den Schritten ändern. CLI: `src/cli/synth_command.cpp` ([cli.md](cli.md#mikroptim-synth)).

## Mehrere GPUs (M7-T3)

Varianten werden nach gemessenem Durchsatz auf die Geräte verteilt (P100 und RTX 3090 im selben Rechner). Ergebnisse pro Variante sind unabhängig von der Verteilung bitgleich, solange alle Geräte denselben Code bitgleich rechnen (gemischte Architekturen: unten). Getrennte Rechner sind nicht Teil von M7-T3; dafür gibt es den Fingerabdruck-Test (unten).

Header `include/mikroptim/cuda/multi_trace_engine.hpp` (reines C++, der CPU-Build kompiliert ihn im Test `MultiGpuSplit.*`), Implementierung `src/cuda/multi_trace_engine.cpp` (Host-C++ in `Mikroptim::cuda`, linkt privat `Threads::Threads`); LM auf mehreren GPUs in `multi_levenberg_marquardt.hpp`/`.cpp`.

```cpp
namespace mko::cuda {
std::vector<int> usable_devices();                         // device_info.hpp: alle Geräte ab sm_60
struct DeviceRange { int first, count; };
enum class WorkKind { Residuals, Merit, Prefilter, Jacobian };
std::vector<DeviceRange> split_variants(int variants, int granularity, std::span<const double> shares);
bool update_shares(std::vector<double>& shares, std::span<const DeviceRange>, std::span<const double> seconds,
                   double smoothing, double min_seconds);  // beide inline, ohne CUDA testbar
struct MultiEngineOptions {
    std::vector<int> devices;               // leer: usable_devices(); Mehrfachnennung = logische Geräte
    EngineOptions engine; std::vector<EngineOptions> slot_engines;   // pro Slot optional
    std::vector<double> weights;            // leer: gemessen; sonst feste Anteile (Tests)
    int calibration_variants = 256; double calibration_seconds = 0.02;
    double smoothing = 0.3; double min_measure_seconds = 0.005;
};
struct DeviceRunInfo { int device; DeviceRange range; double seconds, throughput; TraceRunInfo info; };
struct MultiRunInfo { WorkKind kind; Precision precision; int variants; double seconds;
                      bool calibrated, shares_updated; std::vector<double> shares;
                      std::vector<DeviceRunInfo> devices; };
using MultiDeviceTrancheFn = std::function<void(int slot, const DeviceTranche&)>;
class MultiTraceEngine {                    // nur verschiebbar
public:
    explicit MultiTraceEngine(const MultiEngineOptions& = {});
    void run(const SystemBatchHost&, TraceMode, const TraceSettings&, TraceOutput& out);  // wie TraceEngine
    TraceRunInfo run_on_device(const SystemBatchHost&, const TraceSettings&, int granularity,
                               const MultiDeviceTrancheFn&);
    TraceRunInfo prefilter_variants(const SystemBatchHost&, const PrefilterSettings&, PrefilterResult&);
    void calibrate(const SystemBatchHost& sample, TraceMode, const TraceSettings& = {});
    int size() const; TraceEngine& engine(int slot);
    std::vector<double> shares(WorkKind, Precision = Fp64) const; bool has_estimate(WorkKind, Precision) const;
    void set_shares(WorkKind, Precision, std::span<const double>);
    // Bausteine für weitere Verfahren (MultiGpuLmBackend)
    std::vector<DeviceRange> plan(int units, WorkKind, Precision) const;
    std::vector<double> run_parallel(std::span<const DeviceRange>, const SlotWorkFn&);   // Zeiten [s]
    bool record(WorkKind, Precision, std::span<const DeviceRange>, std::span<const double> seconds);
    const MultiRunInfo& last_run() const; void release() noexcept;
};
// multi_levenberg_marquardt.hpp
class MultiGpuLmBackend final : public LmBackend { /* ein GpuLmBackend pro Slot */ };
LmResult optimize_lm_multi_gpu(MultiTraceEngine&, const VariableSet&, const SystemBatchHost& base,
                               std::span<const double> x0, const LmOptions& = {},
                               std::vector<LmGpuStats>* stats = nullptr);
}
```

**Slots:** Jeder Eintrag in `devices` ist ein Slot mit eigener `TraceEngine` (Puffer, Tranchen, Reserve wie bisher) und eigenem Teil-Batch auf dem Host. Ein Gerät darf mehrfach vorkommen („logische Geräte“): So lassen sich Aufteilung, Nebenläufigkeit und Bitgleichheit mit einer GPU testen. Ohne `memory_limit_bytes` begrenzt die Engine dann jeden der k Slots eines Geräts auf 1/k des beim Erzeugen freien Speichers, damit sich die Slots nicht gegenseitig den Speicher wegnehmen. Logische Geräte teilen sich die GPU (gemeinsamer Default-Stream des Kontexts, `cudaDeviceSynchronize` wartet auf alle Slots des Geräts) und sind nur zum Testen gedacht, nicht schneller. Mit `slot_engines` erhält ein Slot eigene Engine-Einstellungen (z. B. eine Grenze nur für die GPU, an der der Bildschirm hängt).

**Ablauf von `run`** (ebenso `run_on_device` und `prefilter_variants`):

1. **Prüfen** im aufrufenden Thread wie `TraceEngine::run` (Modus, Einstellungen, `ResidualLayout`, `validate_values`, Länge von `extra_merit`), damit Meldungen die Variantennummer des ganzen Batches tragen.
2. **Kalibrieren**, wenn für die Art (`WorkKind`, `Precision`) noch keine Schätzung vorliegt, keine festen Gewichte gelten, es mehr als einen Slot gibt und der Batch mindestens c = `calibration_variants` Varianten hat (kleinere Batches laufen mit gleichen Anteilen; ihre Aufteilung ist unwichtig und eine Messung vom Startaufwand bestimmt): Jeder Slot rechnet die ersten c Varianten (bei `run_on_device` auf ein Vielfaches der Granularität aufgerundet) gleichzeitig mit den anderen, einmal zum Aufwärmen (Kontext, Puffer, Laden der Kernel), dann wiederholt, bis `calibration_seconds` vergangen sind (höchstens 64-mal). Die Anteile werden proportional zum Durchsatz n·c/t gesetzt; die Ergebnisse der Kalibrierung werden verworfen. Für `run_on_device` kalibriert `run` im Modus RESIDUALS.
3. **Aufteilen** mit `split_variants`: U = ⌈V/g⌉ Blöcke aus g Varianten (g = `granularity` bei `run_on_device`, sonst 1), Grenzen b_k = round(U · Σ_{i<k} w_i / Σ w_i), Slot k bearbeitet [b_k·g, min(b_{k+1}·g, V)). Zusammengehörige Varianten (z. B. die 2P + 1 eines FD-Laufs) liegen so immer auf demselben Gerät; innerhalb des Slots sorgt die Engine dafür, dass sie in derselben Tranche liegen.
4. **Rechnen:** ein Host-Thread pro Slot mit Arbeit (der erste davon ist der aufrufende Thread; `std::thread` pro Lauf). Jeder Slot kopiert seinen Bereich mit `SystemBatchHost::assign_variants` in seinen Teil-Batch (keine Kopie, wenn der Bereich der ganze Batch ist), ruft seine Engine auf und kopiert die Ergebnisse an die Stelle `first` der vorher im aufrufenden Thread angelegten Ausgabe (disjunkte Bereiche, keine Sperren). Bei `run_on_device` ruft er den Rückruf mit seinem Slot und `first` bezogen auf den ganzen Batch auf (gleichzeitig aus mehreren Threads, der Rückruf muss thread-sicher sein). Bei der Vorfilterung sammelt er die Rohdaten (`PrefilterStats`); die Entscheidung `prefilter_decide` fällt danach einmal über den ganzen Batch, weil Kriterium (c) die beste Variante aller Geräte braucht. Hat nur ein Slot Arbeit, rechnet seine Engine direkt in `out`.
5. **Statistik und Schätzung:** `out.info` fasst die Slots zusammen (Tranchen, Budget und belegter Speicher summiert, `tranche_variants` als Maximum), `last_run()` enthält pro Slot Bereich, Wandzeit, Durchsatz und die Statistik seiner Engine. Danach geht der Lauf in den gleitenden Mittelwert ein (`update_shares`): Mit dem Durchsatz r_i = n_i/t_i der beteiligten Slots P und ihrem bisherigen Gesamtanteil S_P gilt s_i ← (1 − α)·s_i + α·S_P·r_i/Σ_{j∈P} r_j (α = `smoothing`); nicht beteiligte Slots behalten ihren Anteil, die Summe bleibt 1. Läufe mit weniger als zwei beteiligten Slots oder mit einem Slot unter `min_measure_seconds` ändern nichts (dort bestimmt der Startaufwand die Zeit).

**Schätzungen pro Art:** Das Verhältnis der Geräte hängt von Modus und Zahlentyp ab (RTX 3090 in df64 etwa 4-mal so schnell wie in FP64, die P100 in FP64 schneller als in df64, [hardware.md](hardware.md)), deshalb gibt es eine Schätzung pro (`WorkKind`, `Precision`). Ohne Schätzung gelten gleiche Anteile (gespeichert wird eine Schätzung erst nach einer Kalibrierung oder einem Lauf, der sie tatsächlich aktualisiert); `Jacobian` (Jacobi-Matrizen und Normalgleichungen im LM, ohne eigene Kalibrierung) übernimmt dann die Schätzung von `Residuals`. Die Schätzungen leben so lange wie die Engine; `set_shares` setzt sie z. B. aus einer früheren Sitzung.

**LM auf mehreren GPUs:** `MultiGpuLmBackend` besitzt ein `GpuLmBackend` pro Slot. `normal_equations` und `merit` teilen die Läufe des Aufrufs nach `plan(·, Jacobian bzw. Merit, Fp64)` in zusammenhängende Bereiche, rechnen sie mit `run_parallel` und melden die Zeiten an `record`. Da die Ergebnisse eines Laufs nur von seinem Index und seinem Punkt abhängen (Vertrag von `LmBackend`), ist `optimize_lm_multi_gpu` bitgleich zu `optimize_lm_gpu` (Test `MultiTraceEngine.LmBitIdenticalToSingleGpu`); auch mit dem äußeren Strafverfahren des weichen Versagens (M4-T2): `optimize_lm_multi_gpu` läuft über `optimize_lm_rounds` mit einem `MultiGpuLmBackend` pro Runde, Statistik pro Slot über die Runden summiert (Test `CudaSoftFailure.MultiGpuRoundsBitIdenticalToSingleGpu`). Die Zusatzresiduen werden dabei aus mehreren Threads gleichzeitig ausgewertet (`make_paraxial_residuals` ist dafür geeignet). Eine eigenständige mehrgerätige FD-Jacobi (`fd_jacobian_gpu` mit Download) gibt es nicht; der LM ist ihr einziger Verbraucher.

**Fehler:** Ungültige Argumente werfen vor jeder Gerätearbeit wie bei `TraceEngine`. Scheitert ein Slot, rechnen die anderen ihren Bereich zu Ende (bei `run_on_device` ohne weitere Rückrufe: Der nächste Rückruf bricht den Slot ab), alle Threads werden zusammengeführt, dann wirft die Engine die Ausnahme des kleinsten fehlerhaften Slots mit gleichem Typ (`CudaError`, `std::invalid_argument`, …) und vorangestelltem „Gerät d (Name, Slot k): “. Ausnahmen aus Rückrufen des Aufrufers werden unverändert weitergereicht. Danach bleibt die Engine benutzbar.

**Bitgleichheit:** Jede Variante läuft vollständig auf einem Gerät, der Kernel rechnet sie unabhängig von ihrer Lage im Batch, und Tranchen ändern nichts ([Host-API TraceEngine](#host-api-traceengine-m2-t4)). Daher sind die Ergebnisse jeder Variante unabhängig von der Verteilung bitgleich zum Lauf auf einer Engine, sofern alle Geräte denselben Zahlentyp verwenden und denselben Code bitgleich rechnen. Tests mit logischen Geräten ({0, 0} und {0, 0, 0}, gemessene, gleiche, ungleiche Anteile und ein Anteil 0, V = 1, 2, 7, 700, 701): `MultiTraceEngine.ResidualsBitIdenticalOnLogicalDevices`, `MeritAndDf64BitIdenticalOnLogicalDevices` (MERIT mit `extra_merit` und Max-Reduktionen, beide Modi in df64), `TranchesInsideSlots`, `RunOnDeviceKeepsBlocksTogether`, `PrefilterBitIdenticalOnLogicalDevices`, `DistributionFollowsMeasuredThroughput`, `LmBitIdenticalToSingleGpu`.

### Gemischte Architekturen (sm_60 und sm_86)

Die P100 führt SASS für sm_60 aus, die RTX 3090 SASS für sm_86; beide erzeugt ptxas aus dem PTX derselben Quelle. „Bitgleich unabhängig von der Verteilung“ gilt auf 3090 + P100 nur, wenn beide Architekturen bitgleich rechnen. Untersucht mit nvcc 12.9 (Oktober 2026, ohne P100):

1. **PTX:** Das PTX von `residuals.cu` und `merit.cu` für `compute_60` und `compute_86` enthält dieselben Gleitkommabefehle; es unterscheidet sich nur in 92 Zeilen ganzzahliger Bitoperationen in `__internal_trig_reduction_slowpathd` (`bfe.u32` gegen `shr` + `and`, gleichwertig). In `prefilter.cu` ist nur die Reihenfolge einiger Lade- und Umwandlungsbefehle anders.
2. **Keine Näherungsbefehle:** Kein `.approx`/`.full` im PTX. Division, Wurzel und Kehrwert sind `div.rn`, `sqrt.rn`, `rcp.rn` (auf jeder Architektur korrekt gerundet nach IEEE 754); die `MUFU.RCP64H`/`MUFU.RSQ64H` im SASS sind nur Startwerte dieser korrekt gerundeten Folgen. Sinus und Kosinus (Feldwinkel) kommen aus libdevice und stehen damit vollständig im PTX.
3. **Einziger Spielraum ist die FMA-Kontraktion durch ptxas:** NVVM kontrahiert mit `-fmad=true` (Standard) selbst zu `fma.rn.f64`; übrig bleiben im PTX von `residuals.cu` 190 `mul.f64`, 102 `add.f64` und 228 `sub.f64` *ohne* Rundungsmodifikator, die ptxas laut PTX-ISA zu FMA zusammenfassen darf – grundsätzlich architekturabhängig. Die float-Arithmetik des df64-Pfads verwendet ausschließlich `__fadd_rn`/`__fmul_rn`/`__fmaf_rn` (`.rn`, nicht kontrahierbar); seine wenigen FP64-Befehle (u. a. `residual_scale`) fallen unter Punkt 4.
4. **SASS (cuobjdump des Builds):** In allen Kerneln (`residuals_kernel` und `merit_kernel` für double und df64, je mit und ohne Max-Reduktionen, `extrema_finalize_kernel`, `merit_finalize_kernel`, `fd_difference_kernel`, `gram_kernel`) sind die Mengen der rundenden FP64-Befehle einschließlich Modifikatoren für sm_60 und sm_86 gleich (z. B. `residuals_kernel<double, false>`: je 464 DFMA, 176 DMUL, 100 DADD), nur ihre Reihenfolge (Planung) unterscheidet sich; ptxas hat also auf beiden Architekturen gleich viele Paare kontrahiert. In den df64-Kerneln sind die FFMA gleich (832 bzw. 880), sm_86 hat 235 FADD und 44 FMUL mehr – alle aus `.rn`-Befehlen, also zusätzliche Kopien (z. B. entrollte Schleifen), keine andere Rundung.
5. **JIT auf der RTX 3090:** Ein lokaler Prüfbau nur mit `60-virtual` (PTX für `compute_60`, auf der 3090 mit `CUDA_CACHE_DISABLE=1` per JIT zu sm_86-SASS übersetzt, ca. 8 s) liefert dieselben Fingerabdrücke (FP64, df64, Vorfilterung) wie der native sm_86-Build (Test `MultiTraceEngine.ResultFingerprint`), und die Tests mit logischen Geräten sind dort ebenfalls grün. Das zeigt, dass das `compute_60`-PTX auf sm_86 dieselben Bits ergibt; das sm_60-Backend von ptxas prüft es nicht.

**Ergebnis:** Nach Bau und Analyse sind bitgleiche Ergebnisse auf sm_60 und sm_86 zu erwarten (gleiches PTX, korrekt gerundete Grundoperationen, gleiche Zahl kontrahierter FMA), bewiesen ist es ohne P100 nicht. Die Prüfung auf dem Rechner mit beiden Karten: `MultiTraceEngine.RealDevicesBitIdenticalToSingle` (jedes Gerät einzeln gegen Gerät 0 und die Verteilung auf alle Geräte, FP64 und df64, beide Modi, Vorfilterung) und `MultiTraceEngine.ResultFingerprint` (gleiche Zahlen auf getrennten Rechnern = gleiche Bits). Sollte sich ein Unterschied zeigen, wären Gegenmaßnahmen Änderungen an Numerik- bzw. Toolchain-Regeln und brauchen eine Entscheidung des Maintainers (z. B. `-fmad=false` für die Kernel-Übersetzungseinheiten, explizite `__fma_rn`/`__dmul_rn`/`__dadd_rn` im Mathe-Kern, nur gleichartige Geräte in einer Verteilung oder eine Toleranz statt Bitgleichheit in ADR-0007).

## Verhältnis zu Raytatouille

Mikroptim ist ein eigenständiges Projekt mit eigenem Dateiformat. Ein Adapter zwischen `*.rtt.json` (Raytatouille) und `mikroptim.system/1` ist für M9 vorgesehen (ADR-0008).
