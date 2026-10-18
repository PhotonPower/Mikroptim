#pragma once

// SystemBatch auf dem Host: V Varianten gleicher Topologie als Structure of Arrays in FP64
// (docs/architektur.md, Abschnitt "SystemBatch und Datenlayout"). Reines C++, ohne CUDA-Typen.

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace mko {

// Nur vorwärts deklariert, damit der Header (auch unter nvcc) leicht bleibt:
// mikroptim/model/system.hpp, mikroptim/glass/index_table.hpp, mikroptim/cpu/load_system.hpp.
struct SystemDescription;
struct IndexTable;
struct LoadedSystem;

// Topologie eines Batches: alles, was für alle Varianten gleich sein muss, damit sie in
// denselben Kernelaufruf passen (ADR-0004).
struct BatchTopology {
    int surfaces = 0;     // S, 2 … kMaxSurfaces
    int wavelengths = 0;  // L >= 1
    int fields = 0;       // F >= 1, bei der Punktquelle genau 1
    int ghost_plates = 0; // G >= 0, gleich der Zahl der Flächen mit Typ GhostPlate
    int ghost_terms = 0;  // K >= 0, >= 1 wenn G > 0
    SourceKind source = SourceKind::Collimated;
    std::vector<SurfaceKind> type; // [S]

    friend bool operator==(const BatchTopology&, const BatchTopology&) = default;
};

// Prüft die Regeln aus BatchTopology; wirft std::invalid_argument mit Beschreibung.
void validate_topology(const BatchTopology& topology);

// Besitzt alle Felder eines Batches in FP64 (std::vector, zusammenhängend wie im
// Gerätespeicher). Die Größen aller Felder folgen jederzeit aus Topologie und Variantenzahl
// (Klasseninvariante); die Werte setzt der Aufrufer über die Zugriffsfunktionen.
class SystemBatchHost {
public:
    SystemBatchHost() = default;
    // Leerer Batch mit `variants` Varianten; alle Werte 0 (Gewichte 0, λ_ref 0).
    // Wirft std::invalid_argument bei ungültiger Topologie oder variants < 0.
    explicit SystemBatchHost(BatchTopology topology, int variants = 0);

    const BatchTopology& topology() const { return topology_; }
    BatchShape shape() const;
    int variants() const { return variants_; }
    int surfaces() const { return topology_.surfaces; }
    int wavelengths() const { return topology_.wavelengths; }
    int fields() const { return topology_.fields; }
    int ghost_plates() const { return topology_.ghost_plates; }
    int ghost_terms() const { return topology_.ghost_terms; }
    SourceKind source() const { return topology_.source; }

    // ---- Varianten -------------------------------------------------------------------------
    // Neue Varianten am Ende sind mit 0 gefüllt; Verkleinern verwirft die letzten.
    void resize(int variants);
    // Hängt eine Kopie der Variante `from` an; Rückgabe: Nummer der neuen Variante.
    int add_variant_copy(int from);
    // Kopiert alle Felder pro Variante von `from` nach `to` (beide in diesem Batch).
    void copy_variant(int from, int to);
    // Kopiert Variante `from` aus `other` nach Variante `to` dieses Batches. Verlangt gleiche
    // Topologie (sonst std::invalid_argument); gemeinsame Felder werden nicht kopiert.
    void copy_variant_from(const SystemBatchHost& other, int from, int to);

    // ---- Gemeinsame Felder ----------------------------------------------------------------
    // Alle Zugriffsfunktionen prüfen jeden Index einzeln und werfen std::out_of_range.
    SurfaceKind type(int s) const;
    int ghost_surface(int g) const; // Fläche der Geisterplatte g (aus der Topologie abgeleitet)
    double& ghost_norm_radius(int g);
    double ghost_norm_radius(int g) const;
    double& wavelength_nm(int l);
    double wavelength_nm(int l) const;
    double& wavelength_weight(int l);
    double wavelength_weight(int l) const;
    double& field_weight(int f);
    double field_weight(int f) const;
    double reference_wavelength_nm = 0.0; // λ_ref [nm]
    std::array<double, 3> source_point{}; // nur Punktquelle
    double source_na = 0.0;               // nur Punktquelle

    // ---- Felder pro Variante ---------------------------------------------------------------
    double& curvature(int v, int s);
    double curvature(int v, int s) const;
    double& thickness(int v, int s);
    double thickness(int v, int s) const;
    double& index(int v, int s, int l);
    double index(int v, int s, int l) const;
    double& pupil_radius(int v);
    double pupil_radius(int v) const;
    double& field_angle(int v, int f);
    double field_angle(int v, int f) const;
    double& field_target(int v, int f);
    double field_target(int v, int f) const;
    double& ghost_coeff(int v, int g, int k);
    double ghost_coeff(int v, int g, int k) const;
    double& ghost_index(int v, int g, int l);
    double ghost_index(int v, int g, int l) const;
    // Arbeitsabstand = thickness(v, S−1) (einzige Quelle, kein eigenes Feld).
    double& image_distance(int v) { return thickness(v, surfaces() - 1); }
    double image_distance(int v) const { return thickness(v, surfaces() - 1); }

    // ---- Rohfelder (für Upload/Download und schnelle Schleifen) ---------------------------
    std::span<const SurfaceKind> type_data() const { return topology_.type; }
    std::span<const int> ghost_surface_data() const { return ghost_surface_; }
    std::span<double> ghost_norm_radius_data() { return ghost_norm_radius_; }
    std::span<const double> ghost_norm_radius_data() const { return ghost_norm_radius_; }
    std::span<double> wavelength_nm_data() { return wavelength_nm_; }
    std::span<const double> wavelength_nm_data() const { return wavelength_nm_; }
    std::span<double> wavelength_weight_data() { return wavelength_weight_; }
    std::span<const double> wavelength_weight_data() const { return wavelength_weight_; }
    std::span<double> field_weight_data() { return field_weight_; }
    std::span<const double> field_weight_data() const { return field_weight_; }
    std::span<double> curvature_data() { return curvature_; }
    std::span<const double> curvature_data() const { return curvature_; }
    std::span<double> thickness_data() { return thickness_; }
    std::span<const double> thickness_data() const { return thickness_; }
    std::span<double> index_data() { return index_; }
    std::span<const double> index_data() const { return index_; }
    std::span<double> pupil_radius_data() { return pupil_radius_; }
    std::span<const double> pupil_radius_data() const { return pupil_radius_; }
    std::span<double> field_angle_data() { return field_angle_; }
    std::span<const double> field_angle_data() const { return field_angle_; }
    std::span<double> field_target_data() { return field_target_; }
    std::span<const double> field_target_data() const { return field_target_; }
    std::span<double> ghost_coeff_data() { return ghost_coeff_; }
    std::span<const double> ghost_coeff_data() const { return ghost_coeff_; }
    std::span<double> ghost_index_data() { return ghost_index_; }
    std::span<const double> ghost_index_data() const { return ghost_index_; }

    // ---- Sichten ---------------------------------------------------------------------------
    // Sicht auf die Host-Vektoren; gültig, bis der Batch verändert (resize, Zuweisung) oder
    // zerstört wird.
    SystemBatchView<double> view() const;
    // Flächentabelle der Variante v für trace_ray (wirft std::out_of_range bei ungültigem v).
    SurfaceTableView<double> surface_table(int v) const;

    // Prüft die Werte: alle endlich, Pupillenradius > 0, Brechzahlen > 0, Gewichte >= 0,
    // Normierungsradien > 0, λ_ref > 0, bei der Punktquelle 0 < NA < index[v][0][l].
    // Wirft std::invalid_argument mit Variante und Feld der ersten Verletzung.
    void validate_values() const;

    // Bitweiser Vergleich aller Felder (gemeinsam und pro Variante) und der Topologie.
    friend bool operator==(const SystemBatchHost&, const SystemBatchHost&);

private:
    // Prüfen einen Index gegen [0, count) und werfen std::out_of_range mit Feldname.
    static void check(int i, int count, const char* what);

    BatchTopology topology_;
    int variants_ = 0;
    std::vector<int> ghost_surface_;
    std::vector<double> ghost_norm_radius_;
    std::vector<double> wavelength_nm_;
    std::vector<double> wavelength_weight_;
    std::vector<double> field_weight_;
    std::vector<double> curvature_;
    std::vector<double> thickness_;
    std::vector<double> index_;
    std::vector<double> pupil_radius_;
    std::vector<double> field_angle_;
    std::vector<double> field_target_;
    std::vector<double> ghost_coeff_;
    std::vector<double> ghost_index_;
};

// Topologie eines Systems aus der Datei (G = K = 0, Geisterplatten folgen in M4).
BatchTopology topology_of(const SystemDescription& system, const IndexTable& index);

// Baut einen Batch mit `variants` gleichen Varianten aus Systembeschreibung und Brechzahltabelle
// (make_surface_table): Krümmungen (0 bei Stop und Plane), Dicken, Brechzahlen, Pupillenradius
// = semi_diameter der Fläche 0, Feldwinkel θ_f = field_angles_deg[f] · π/180, Gewichte und
// Wellenlängen aus spectrum bzw. source, λ_ref = spectrum.reference_nm.
// Feldziel h_f = f_soll · tan θ_f (docs/numerik.md, Abschnitt 7). f_soll ist der Reihe nach
// `focal_length`, target.focal_length oder die paraxiale Brennweite bei λ_ref (nur wenn λ_ref
// unter den Wellenlängen des Spektrums ist). Ist keine Brennweite bestimmbar (afokales System)
// oder θ_f = 0 oder die Quelle eine Punktquelle, ist h_f = 0.
// Wirft std::invalid_argument bei unpassender Tabelle (wie make_surface_table), variants < 1
// oder nicht endlicher `focal_length`.
SystemBatchHost make_system_batch(const SystemDescription& system, const IndexTable& index,
                                  int variants = 1,
                                  std::optional<double> focal_length = std::nullopt);

// Wie oben für ein geladenes System; f_soll = target.focal_length, sonst die paraxiale
// Brennweite bei spectrum.reference_nm (compute_paraxial(system, catalog)), afokal: h_f = 0.
SystemBatchHost make_system_batch(const LoadedSystem& loaded, int variants = 1);

} // namespace mko
