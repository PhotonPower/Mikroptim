#pragma once

// Modus EVAL (M8-T1): Auswertung eines SystemBatch auf einem dichten kartesischen
// Pupillenraster. OPD-Karten, Spotdiagramme, Verzeichnung und Feldkrümmung pro Variante, Feld
// und Wellenlänge. Reines C++ ohne CUDA-Typen (Bibliothek Mikroptim::eval): Raster, Ablage und
// Speicherabschätzung, CPU-Referenz (eval_cpu) und die Kennzahlen (eval_statistics), die für
// CPU- und GPU-Ergebnisse dieselben sind. Der Kernel EVAL ist über
// mko::cuda::TraceEngine::evaluate erreichbar. Festlegungen: docs/numerik.md, Abschnitt 7
// ("Auswertung auf dem Raster, M8-T1"); Ablage und Ablauf: docs/architektur.md, Abschnitt
// "Modus EVAL".

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/eval/eval_ray.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mko {

// Größte Rasterkantenlänge N (N² ≈ 4,2·10⁶ Rasterpunkte, 3,3·10⁶ in der Pupille).
inline constexpr int kMaxEvalGrid = 2048;

// Einstellungen eines Laufs im Modus EVAL (für alle Varianten, Felder und Wellenlängen gleich).
struct EvalSettings {
    // Kantenlänge N des Rasters über das Quadrat [−1, 1]² der normierten Pupille, Zellmitten
    // x_i = (2i + 1 − N)/N; Punkte mit x² + y² > 1 gehören nicht zur Pupille. 1 … kMaxEvalGrid.
    int grid = 64;
    // Nur die Hälfte x >= 0 (Symmetrie x → −x, Feld in der y-z-Ebene); Punkte mit x > 0 zählen
    // doppelt. Nicht erlaubt bei einer Punktquelle mit x ≠ 0 (keine Symmetrie).
    bool half_pupil = false;
    // Lage der Referenzsphäre (Standard: durch die paraxiale Austrittspupille).
    EvalReference reference = EvalReference::ExitPupil;
    // Abstand δ der Nachbarstrahlen für die Fokuslagen in normierten Pupillenkoordinaten,
    // 0 < δ <= 0,5 (Abbruchfehler O(δ²), Rundung ~ ε/δ).
    double field_ray_offset = 1e-3;
};

// Prüft die Einstellungen (ohne Batch); wirft std::invalid_argument.
void check_eval_settings(const EvalSettings& settings);

// Prüft die Einstellungen gegen einen Batch: check_eval_settings, Halbpupille nur bei
// Symmetrie x → −x (Punktquelle mit source_point[0] = 0, koaxialer Batch; M8-T3). Wirft
// std::invalid_argument.
void check_eval_settings(const EvalSettings& settings, const SystemBatchHost& batch);

// Rasterpunkt in normierten Pupillenkoordinaten mit Gewicht (Summe über das Raster 1).
struct EvalPoint {
    double x;
    double y;
    double weight;
};

// Raster der Pupille: die Zellmitten mit x² + y² <= 1 (ganzzahlig geprüft:
// (2i + 1 − N)² + (2j + 1 − N)² <= N²), zeilenweise (Zeile j aufsteigend in y, darin Spalte i
// aufsteigend in x). Volle Pupille: alle Gewichte 1/n_voll; halbe Pupille: nur x >= 0, Gewicht
// 2/n_voll für x > 0 und 1/n_voll auf der Spalte x = 0 (nur bei ungeradem N).
struct EvalGrid {
    int size = 0;       // N
    bool half = false;  // nur x >= 0
    int full_count = 0; // Punkte der vollen Pupille n_voll
    std::vector<EvalPoint> points;
    std::vector<std::int32_t> column; // Spalte i des Punktes (x)
    std::vector<std::int32_t> row;    // Zeile j des Punktes (y)

    std::int64_t samples() const { return static_cast<std::int64_t>(points.size()); }
};

// Wirft std::invalid_argument bei size ∉ [1, kMaxEvalGrid].
EvalGrid make_eval_grid(int size, bool half_pupil);

// Krümmung κ = σ/R der Referenzsphäre pro Variante und Feld [V·F] (Ablage v·F + f) und, nur
// im Modus ExitPupil, die Lage der Austrittspupille z_AP [V] relativ zur letzten Fläche
// (batch_paraxial bei λ_ref; NaN im Modus Vertex und bei einer Austrittspupille im
// Unendlichen, dann κ = 0). Mit M_f = (0, h_f, w): ExitPupil R = |M_f − E|, σ = +1 für
// w >= z_AP, sonst −1; Vertex R = |M_f|, σ = +1 für w >= 0 (nur zur Information; der Schritt
// nutzt dann reference_sphere_step). Wirft std::invalid_argument, wenn die Paraxialrechnung
// scheitert (z. B. paraxial_index nicht gesetzt) oder R = 0 ist.
struct EvalReferenceGeometry {
    std::vector<double> curvature;           // [V·F]
    std::vector<double> exit_pupil_position; // [V]
};
EvalReferenceGeometry eval_reference_geometry(const SystemBatchHost& batch,
                                              EvalReference reference);

// Ergebnis eines Laufs (CPU-Referenz oder Download vom Gerät). Strahlen des Rasters liegen
// unter ((v·F + f)·L + l)·n + k (n = grid.samples()), Feldstrahlen unter
// ((v·F + f)·L + l)·kEvalFieldRays + j. Versagte Strahlen haben NaN in OPD, Spot und
// Feldstrahl-Werten; ihr Status ist gesetzt (hartes Versagen).
struct EvalResult {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    int variants = 0;
    int fields = 0;
    int wavelengths = 0;
    EvalSettings settings{};
    EvalGrid grid;
    double reference_nm = 0.0;               // λ_ref des Batches (OPD in Wellen)
    std::vector<double> image_height;        // [V·F] ideale Bildhöhe h_f (field_target)
    std::vector<double> reference_curvature; // [V·F] κ = σ/R der Referenzsphäre
    std::vector<double> exit_pupil_position; // [V] z_AP (nur ExitPupil, sonst NaN)
    std::vector<double> opd;                 // [V·F·L·n] OPD nach Kolbenabzug [mm]
    std::vector<double> spot_x;              // [V·F·L·n] Bildebene, relativ zu M_f [mm]
    std::vector<double> spot_y;              // [V·F·L·n]
    std::vector<std::uint8_t> status;        // [V·F·L·n] TraceStatus
    std::vector<double> field_rays;          // [V·F·L·kEvalFieldRays·kEvalFieldRayValues]
    std::vector<std::uint8_t> field_status;  // [V·F·L·kEvalFieldRays]

    std::int64_t samples() const { return grid.samples(); }
    std::size_t tuple(int v, int f, int l) const {
        return (static_cast<std::size_t>(v) * static_cast<std::size_t>(fields) +
                static_cast<std::size_t>(f)) *
                   static_cast<std::size_t>(wavelengths) +
               static_cast<std::size_t>(l);
    }
    std::size_t ray_index(int v, int f, int l, std::int64_t k) const {
        return tuple(v, f, l) * static_cast<std::size_t>(samples()) + static_cast<std::size_t>(k);
    }
    // Wert `value` (0 … kEvalFieldRayValues − 1) des Feldstrahls j.
    double field_ray(int v, int f, int l, int j, int value) const {
        return field_rays[(tuple(v, f, l) * kEvalFieldRays + static_cast<std::size_t>(j)) *
                              kEvalFieldRayValues +
                          static_cast<std::size_t>(value)];
    }
    TraceStatus field_ray_status(int v, int f, int l, int j) const {
        return static_cast<TraceStatus>(
            field_status[tuple(v, f, l) * kEvalFieldRays + static_cast<std::size_t>(j)]);
    }
    // Zahl der Rasterstrahlen mit Status ≠ Ok (alle Tupel).
    std::size_t failed_count() const;
};

// Legt die Vektoren eines Ergebnisses für `variants` Varianten an (Werte unbestimmt) und
// übernimmt Raster, Einstellungen, Bildhöhen und die Geometrie der Referenzsphäre.
void prepare_eval_result(EvalResult& out, const SystemBatchHost& batch,
                         const EvalSettings& settings, EvalGrid grid,
                         EvalReferenceGeometry geometry);

// CPU-Referenz des Kernels EVAL für alle Varianten: pro Tupel (v, f, l) und Rasterpunkt
// eval_ray über surface_table(v), Kolben mit opd_statistics über OPL − OPL_Hauptstrahl
// (Gewichte des Rasters, versagte Strahlen ausgeschlossen, Summation in aufsteigender
// Reihenfolge k), OPD = (OPL − OPL_Hauptstrahl) − Kolben; dazu die vier Feldstrahlen. Wirft
// std::invalid_argument (Einstellungen, Werte im Batch: validate_values, Referenzgeometrie).
EvalResult eval_cpu(const SystemBatchHost& batch, const EvalSettings& settings = {});

// Kennzahlen eines Tupels (Variante v, Feld f, Wellenlänge l), berechnet auf dem Host aus einem
// EvalResult (CPU oder GPU). Mittelwerte über die gültigen Rasterpunkte mit den Gewichten des
// Rasters (renormiert); bei halber Pupille zählt jeder Punkt als Paar (±x, y) mit gespiegelten
// Werten (Spot x → −x). NaN, wo kein gültiger Strahl vorliegt.
struct EvalTupleStats {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    int field = 0;
    int wavelength = 0;
    std::int64_t rays = 0;     // Rasterpunkte n
    std::int64_t failed = 0;   // davon versagt
    double valid_weight = 0.0; // Σ W der gültigen Rasterpunkte (1 ohne Versagen)
    // Wellenfront [mm]: RMS und PV der OPD nach Kolbenabzug (wie MERIT), RMS nach Abzug von
    // Kolben und Kippung bzw. Kolben, Kippung und Defokus (gewichtete kleinste Quadrate mit der
    // Basis 1, x, y, ρ² auf den gültigen Rasterpunkten).
    double rms_opd = kNaN;
    double pv_opd = kNaN;
    double rms_opd_tilt = kNaN;
    double rms_opd_defocus = kNaN;
    // Koeffizienten des Fits OPD ≈ c₀ + t_x x + t_y y + c_d ρ² [mm] (x, y normiert).
    double tilt_x = kNaN;
    double tilt_y = kNaN;
    double defocus = kNaN;
    // Spotdiagramm in der Bildebene, relativ zum idealen Bildpunkt M_f [mm]: Schwerpunkt und
    // RMS-Radius um den Schwerpunkt.
    double centroid_x = kNaN;
    double centroid_y = kNaN;
    double rms_spot = kNaN;
    // Feldgrößen aus den Feldstrahlen: ideale Bildhöhe h_f, Hauptstrahl in der Bildebene
    // relativ zu M_f, Verzeichnung 100·(y − h_f)/h_f [%] über den Hauptstrahl bzw. den
    // Schwerpunkt (NaN für h_f = 0), Fokuslagen tangential und sagittal als z relativ zur
    // Bildebene [mm] (> 0: Fokus hinter der Bildebene).
    double image_height = 0.0;
    double chief_x = kNaN;
    double chief_y = kNaN;
    double distortion_chief = kNaN;
    double distortion_centroid = kNaN;
    double focus_tangential = kNaN;
    double focus_sagittal = kNaN;

    bool valid() const { return valid_weight > 0.0; }
    // Astigmatische Differenz z_t − z_s [mm].
    double astigmatism() const { return focus_tangential - focus_sagittal; }
};

// Kennzahlen aller Tupel der Variante v, Ablage [f·L + l]. Summiert sequenziell in
// aufsteigender Reihenfolge (deterministisch). Wirft std::out_of_range (v).
std::vector<EvalTupleStats> eval_statistics(const EvalResult& result, int v);

// Was vor der Darstellung einer OPD-Karte abgezogen wird (Standard nur Kolben, wie MERIT).
enum class OpdRemoval { Piston, Tilt, Defocus };

// OPD-Karte eines Tupels auf dem vollen N×N-Raster, values[j·N + i] (Zeile j = y, Spalte i =
// x) [mm]; NaN außerhalb der Pupille und bei versagten Strahlen. Bei halber Pupille wird die
// Hälfte x < 0 gespiegelt. Tilt/Defocus ziehen zusätzlich den Fit aus eval_statistics ab.
struct EvalMap {
    int size = 0;
    std::vector<double> values;
    double at(int i, int j) const {
        return values[static_cast<std::size_t>(j) * static_cast<std::size_t>(size) +
                      static_cast<std::size_t>(i)];
    }
};
EvalMap eval_opd_map(const EvalResult& result, int v, int f, int l,
                     OpdRemoval removal = OpdRemoval::Piston);

// ---- Speicher auf dem Gerät --------------------------------------------------------------

// Ablage der Ausgaben des Kernels EVAL im Gerätepuffer (Bereiche auf kBatchAlignment
// ausgerichtet), n = Rasterpunkte pro Tupel, B = F·L Tupel pro Variante.
struct EvalOutputLayout {
    BufferRegion opd;                 // V·B·n · 8 (vorläufig OPL, dann OPD)
    BufferRegion spot_x;              // V·B·n · 8
    BufferRegion spot_y;              // V·B·n · 8
    BufferRegion status;              // V·B·n · 1
    BufferRegion field_rays;          // V·B·kEvalFieldRays·kEvalFieldRayValues · 8
    BufferRegion field_status;        // V·B·kEvalFieldRays · 1
    BufferRegion reference_curvature; // V·F · 8
    BufferRegion points;              // n · sizeof(EvalPoint), unabhängig von V
    std::size_t total_bytes = 0;
};

// Wirft std::invalid_argument bei ungültiger Topologie, variants < 0 oder samples < 1.
EvalOutputLayout make_eval_output_layout(const BatchTopology& topology, int variants,
                                         std::int64_t samples);

// Gerätespeicher eines Laufs mit `variants` Varianten: Batch + Ausgaben.
std::size_t eval_device_bytes(const BatchTopology& topology, int variants, std::int64_t samples);

} // namespace mko
