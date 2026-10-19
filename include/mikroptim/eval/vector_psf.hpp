#pragma once

// Vektorielle PSF für hohe NA (M8-T5): Debye-Wolf-Integral (Richards-Wolf) über die
// Richtungskosinus des Bildraums, berechnet als zweidimensionale FFT der drei Feldkomponenten
// (Leutenegger et al. 2006). Eingangsfeld: gleichmäßig beleuchtete Eintrittspupille, linear (x,
// y), zirkular polarisiert oder unpolarisiert (inkohärente Summe aus x und y); Aplanatfaktor
// √cos θ, Aberrationen als Phase aus der OPD-Karte des Modus EVAL, wahlweise defokussierte
// Ebene. Reines C++ ohne CUDA-Typen (Bibliothek Mikroptim::eval): Pupille aus einem EvalResult,
// CPU-Referenz mit der eigenen FFT, Kennzahlen (für CPU- und GPU-Ergebnisse dieselben) und die
// direkte Auswertung des Feldes an beliebigen Punkten. Auf der GPU über
// mko::cuda::VectorPsfEngine. Formeln, Apodisation, Abtastung und Gültigkeit: docs/numerik.md,
// Abschnitt "Vektorielle PSF (M8-T5)"; Ablauf: docs/architektur.md, Abschnitt "Vektorielle PSF".

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "mikroptim/eval/vector_psf_ops.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace mko {

// Polarisation des Eingangsfeldes in der Eintrittspupille.
enum class Polarization {
    X,          // linear, J = (1, 0)
    Y,          // linear, J = (0, 1)
    Circular,   // J = (1, i)/√2 (kohärent)
    Unpolarized // inkohärente Summe (|E(1, 0)|² + |E(0, 1)|²)/2
};

// "x", "y", "circular", "unpolarized".
const char* polarization_name(Polarization polarization);
// Umkehrung von polarization_name; wirft std::invalid_argument.
Polarization parse_polarization(std::string_view name);

// Eingangsfelder einer Polarisation (count Jones-Vektoren mit Gewichten, Summe der Gewichte · |J|²
// = 1).
struct VectorPsfInputs {
    int count = 0;
    VectorPsfJones jones[kMaxVectorPsfInputs] = {};
};
VectorPsfInputs vector_psf_inputs(Polarization polarization);

// Einstellungen der vektoriellen PSF (für alle Tupel gleich; Zero-Padding und NA wie bei der
// skalaren PSF über PsfSettings).
struct VectorPsfSettings {
    Polarization polarization = Polarization::X;
    // Lage der Auswerteebene relativ zur Bildebene entlang der Achse [mm] (> 0: in +z, also in
    // Tracerichtung hinter der Bildebene); Phase k n' z (cos θ − 1).
    double defocus = 0.0;
};

// Wirft std::invalid_argument (defocus nicht endlich).
void check_vector_psf_settings(const VectorPsfSettings& settings);

// Pupille eines Tupels: die skalare Pupille (OPD-Karte, NA, Spiegelung) und die Geometrie der
// Bildraumrichtungen.
struct VectorPsfPupil {
    PsfPupil pupil;
    double index = 1.0;   // Brechzahl n' des Bildraums bei der Wellenlänge des Tupels
    double chief_x = 0.0; // Richtungskosinus des Hauptstrahls im Bildraum
    double chief_y = 0.0;
    double input_y = 0.0; // Richtung des Eingangsbündels u = (0, sin θ_f, cos θ_f)
    double input_z = 1.0;
};

// Pupille des Tupels (v, f, l): make_psf_pupil (OPD-Karte, NA aus den Feldstrahlen oder
// vorgegeben), n' = index(v, S − 1, l), Hauptstrahlrichtung aus dem Feldstrahl (0 wenn er
// versagt und die NA vorgegeben ist), Eingangsbündel (0, sin θ_f, cos θ_f) bei kollimierter
// Quelle, (0, 0, 1) bei einer Punktquelle. Ausnahmen wie make_psf_pupil.
VectorPsfPupil make_vector_psf_pupil(const SystemBatchHost& batch, const EvalResult& result, int v,
                                     int f, int l, const PsfSettings& settings = {});

// Alle Tupel der Variante v, Ablage [f·L + l].
std::vector<VectorPsfPupil> make_vector_psf_pupils(const SystemBatchHost& batch,
                                                   const EvalResult& result, int v,
                                                   const PsfSettings& settings = {});

// Bezugsintensität I_0 = Σ_k w_k Σ_c |Σ_p G_c(p)|² der perfekten Pupille (alle Rasterpunkte der
// vollen Pupille, OPD = 0, keine Defokussierung) im Fokus: das Maximum der aberrationsfreien
// vektoriellen PSF derselben NA, auf das die Intensität normiert wird. Wirft
// std::invalid_argument (Pupille unvollständig, kein ausbreitungsfähiger Rasterpunkt).
double vector_psf_reference_intensity(const VectorPsfPupil& pupil, Polarization polarization);

// Kernel-Parameter einer Pupille (mit scale = 1/reference_intensity).
VectorPsfParams make_vector_psf_params(const VectorPsfPupil& pupil,
                                       const VectorPsfSettings& settings,
                                       double reference_intensity);

// Ergebnis eines Tupels (CPU oder GPU). Die Felder von PsfResult haben dieselbe Bedeutung wie bei
// der skalaren PSF, aber für die vektorielle Intensität |E|² = |E_x|² + |E_y|² + |E_z|² (bei
// unpolarisiert gemittelt), normiert auf I_0 (das Maximum der perfekten vektoriellen PSF ist 1):
// psf, MTF der Intensität, energy (Σ|E|² gegen Σ W² der Pupille), strehl_fft, strehl_origin
// (direkte Auswertung im idealen Bildpunkt), strehl (im Bezugspunkt der Kippung), peak.
// airy_radius bleibt der skalare Wert j₁,₁/(2π) λ/NA.
struct VectorPsfResult : PsfResult {
    Polarization polarization = Polarization::X;
    double defocus = 0.0;
    double reference_intensity = kNaN; // I_0
    // |E_x|², |E_y|², |E_z|² [M·M] wie psf (gleiche Ablage und Normierung, Summe = psf).
    std::array<std::vector<double>, 3> component;
    // Kennzahlen (finish_vector_psf):
    std::array<double, 3> energy_fraction = {kNaN, kNaN, kNaN}; // Σ|E_c|² / Σ|E|² (Bildebene)
    std::array<double, 3> component_peak = {kNaN, kNaN, kNaN};  // max |E_c|² (Strehl-Einheiten)
    // Halbwertsbreiten der Intensität entlang x und y durch das abgetastete Maximum [mm]
    // (Rand durch Bisektion auf dem direkt ausgewerteten Feld, unabhängig von der Abtastung).
    double fwhm_x = kNaN;
    double fwhm_y = kNaN;
    std::int64_t evanescent = 0; // Rasterpunkte der Pupille mit |s_⊥| >= 1 (ohne Transmission)

    double component_at(int c, int i, int j) const {
        return component[static_cast<std::size_t>(c)]
                        [static_cast<std::size_t>(j) * static_cast<std::size_t>(fft_size) +
                         static_cast<std::size_t>(i)];
    }
};

// Legt ein Ergebnis an (prepare_psf_result, Komponenten, I_0); Werte unbestimmt. Wirft
// std::invalid_argument wie prepare_psf_result und vector_psf_reference_intensity.
VectorPsfResult prepare_vector_psf_result(const VectorPsfPupil& pupil, int fft_size,
                                          const VectorPsfSettings& settings);

// Kennzahlen aus PSF, Komponenten und Pupille (Host, sequenziell, für CPU- und GPU-Ergebnisse
// dieselbe Funktion).
void finish_vector_psf(VectorPsfResult& result, const VectorPsfPupil& pupil);

// Direkte Auswertung des Debye-Integrals (Summe über die Rasterpunkte, keine FFT) im Punkt
// (x, y) [mm] relativ zum idealen Bildpunkt in der Ebene von settings.defocus: Intensität in
// Strehl-Einheiten (geteilt durch reference_intensity), optional die Komponenten.
double vector_psf_intensity_at(const VectorPsfPupil& pupil, const VectorPsfSettings& settings,
                               double reference_intensity, double x, double y,
                               std::array<double, 3>* components = nullptr);

// CPU-Referenz eines Tupels: pro Eingangsfeld drei Pupillenfelder G_c in die Ecke eines
// M×M-Feldes, FFT vorwärts, |E_c|² gewichtet aufsummiert und auf I_0 normiert, zentriert (und
// gespiegelt bei divergenter Welle), FFT⁻¹ der Intensität für die MTF, dann finish_vector_psf.
VectorPsfResult vector_psf_cpu(const VectorPsfPupil& pupil, int fft_size,
                               const VectorPsfSettings& settings = {});

// Alle Tupel der Variante v (M = psf_fft_size(N, psf_settings.padding)).
std::vector<VectorPsfResult> vector_psf_cpu(const SystemBatchHost& batch, const EvalResult& result,
                                            int v, const PsfSettings& psf_settings = {},
                                            const VectorPsfSettings& settings = {});

// ---- Speicher auf dem Gerät --------------------------------------------------------------

// Ablage der Puffer der VectorPsfEngine für eine Tranche von T Pupillen mit K Eingangsfeldern
// (Bereiche auf kBatchAlignment ausgerichtet); der Arbeitsspeicher von cuFFT kommt dazu.
struct VectorPsfDeviceLayout {
    BufferRegion spectrum;  // T·K·3·M² · 16 (Feldkomponenten, Spektren; Slot 0 danach OTF)
    BufferRegion psf;       // T·M² · 8
    BufferRegion component; // T·3·M² · 8
    BufferRegion opd;       // T·N² · 8
    BufferRegion params;    // T · sizeof(VectorPsfParams)
    BufferRegion mtf;       // T·2(N + 1) · 8
    std::size_t total_bytes = 0;
};

// Wirft std::invalid_argument bei grid < 1, fft_size < 2·grid, inputs ∉ [1, 2] oder pupils < 1.
VectorPsfDeviceLayout make_vector_psf_device_layout(int grid, int fft_size, int inputs, int pupils);

} // namespace mko
