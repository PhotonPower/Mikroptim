#pragma once

// Skalare Beugung (M8-T2): PSF, MTF und Strehl aus den OPD-Karten des Modus EVAL über eine
// zweidimensionale FFT der Pupillenfunktion P = A·exp(2πi·OPD/λ). Reines C++ ohne CUDA-Typen
// (Bibliothek Mikroptim::eval): Pupillenfunktion aus einem EvalResult, CPU-Referenz mit der
// eigenen Radix-2-FFT (mikroptim/eval/fft.hpp) und die Kennzahlen, die für CPU- und
// GPU-Ergebnisse dieselben sind. Auf der GPU über mko::cuda::PsfEngine (cuFFT). Formeln,
// Abtastung und Grenzen der skalaren Näherung bei hoher NA: docs/numerik.md, Abschnitt
// "Beugung: PSF, MTF, Strehl (M8-T2)"; Ablauf: docs/architektur.md, Abschnitt "PSF, MTF,
// Strehl".

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/eval/eval.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mko {

// Größte FFT-Kantenlänge M (M² komplexe Werte = 1 GiB pro Tupel).
inline constexpr int kMaxPsfFft = 8192;

// Erste Nullstelle j₁,₁ der Besselfunktion J₁: Airy-Radius r = j₁,₁/(2π) · λ/NA ≈ 0,61 λ/NA.
inline constexpr double kBesselJ1FirstZero = 3.8317059702075123;

// Einstellungen der PSF (für alle Tupel gleich).
struct PsfSettings {
    // Mindestfaktor Q des Zero-Paddings: FFT-Größe M = kleinste Zweierpotenz >= Q·N
    // (N Rasterkante von EVAL), Bildpunktabstand Δx = λ N / (2 M NA). Q >= 2, damit die
    // Autokorrelation der Pupille (MTF) nicht überlappt.
    double padding = 4.0;
    // NA der Pupillenabbildung in beiden Richtungen vorgeben (> 0). 0: aus den Feldstrahlen
    // des Tupels (Ableitung der Bildraumrichtung nach den Pupillenkoordinaten, numerik.md).
    double numerical_aperture = 0.0;
};

// Wirft std::invalid_argument (padding < 2 oder nicht endlich, numerical_aperture < 0).
void check_psf_settings(const PsfSettings& settings);

// FFT-Größe M = kleinste Zweierpotenz >= max(2N, ⌈Q·N⌉). Wirft std::invalid_argument bei
// ungültigem N, Q oder M > kMaxPsfFft.
int psf_fft_size(int grid, double padding);

// Pupillenfunktion eines Tupels auf dem vollen N×N-Raster von EVAL. Die Amplitude ist 1 in der
// Pupille und 0 außerhalb und bei versagten Strahlen (OPD NaN).
struct PsfPupil {
    int field = 0;
    int wavelength = 0;
    int grid = 0;               // N
    int full_count = 0;         // Rasterpunkte der vollen Pupille (Normierung der PSF)
    double wavelength_nm = 0.0; // Vakuumwellenlänge des Tupels
    // NA der Pupillenabbildung n'·|∂d/∂x| (sagittal, x) und n'·|∂d/∂y| (tangential, y).
    double na_x = 0.0;
    double na_y = 0.0;
    // Divergente Welle (virtueller Bildpunkt, ∂d/∂x > 0): Die Bildkoordinate ist gespiegelt.
    bool flip_x = false;
    bool flip_y = false;
    std::vector<double>
        opd; // [N·N] OPD [mm] nach Kolbenabzug, opd[j·N + i]; NaN: keine Transmission

    double wavelength_mm() const { return wavelength_nm * 1e-6; }
};

// Pupillenfunktion des Tupels (v, f, l): OPD-Karte aus eval_opd_map (halbe Pupille gespiegelt),
// NA aus den Feldstrahlen mit der Brechzahl n' = index(v, S − 1, l) des Bildraums oder aus
// settings.numerical_aperture. Wirft std::invalid_argument (Einstellungen; NA nicht bestimmbar,
// z. B. Feldstrahl versagt) und std::out_of_range (Tupel).
PsfPupil make_psf_pupil(const SystemBatchHost& batch, const EvalResult& result, int v, int f, int l,
                        const PsfSettings& settings = {});

// Alle Tupel der Variante v, Ablage [f·L + l].
std::vector<PsfPupil> make_psf_pupils(const SystemBatchHost& batch, const EvalResult& result, int v,
                                      const PsfSettings& settings = {});

// Ergebnis eines Tupels (CPU oder GPU).
struct PsfResult {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    int field = 0;
    int wavelength = 0;
    int grid = 0;     // N
    int fft_size = 0; // M
    double wavelength_nm = 0.0;
    double na_x = 0.0;
    double na_y = 0.0;
    // Bildpunktabstand Δx = λ N/(2 M NA_x), Δy entsprechend [mm].
    double dx = 0.0;
    double dy = 0.0;
    // Airy-Radius j₁,₁/(2π) · λ/NA mit NA = √(NA_x NA_y) [mm].
    double airy_radius = 0.0;
    // PSF [M·M], psf[j·M + i] im Bildpunkt ((i − M/2)·Δx, (j − M/2)·Δy) relativ zum idealen
    // Bildpunkt M_f; normiert auf das Maximum der PSF der ungestörten vollen Pupille (= 1).
    std::vector<double> psf;
    // MTF-Schnitte, normiert auf MTF(0) = 1, k = 0 … N bei den Frequenzen f = k · f_c/N mit der
    // Grenzfrequenz f_c = 2 NA/λ [Linienpaare/mm]: sagittal entlang f_x, tangential entlang f_y.
    std::vector<double> mtf_sagittal;
    std::vector<double> mtf_tangential;
    // Kennzahlen (finish_psf):
    std::int64_t transmitted = 0; // Rasterpunkte mit Transmission (gültige Strahlen)
    double energy = kNaN;         // Σ PSF · n_voll / M² (Parseval: = transmitted / n_voll)
    double strehl_fft = kNaN;     // PSF im idealen Bildpunkt (FFT)
    double strehl_origin = kNaN;  // dasselbe direkt: |Σ exp(iφ)|² / n_voll²
    double strehl = kNaN;         // direkt, nach Abzug der Kippung (Kleinste-Quadrate-Fit)
    double reference_x = kNaN;    // Bildpunkt, auf den sich `strehl` bezieht (Kippung) [mm]
    double reference_y = kNaN;
    double peak = kNaN;   // größter Abtastwert der PSF
    double peak_x = kNaN; // dessen Lage [mm]
    double peak_y = kNaN;

    double psf_at(int i, int j) const {
        return psf[static_cast<std::size_t>(j) * static_cast<std::size_t>(fft_size) +
                   static_cast<std::size_t>(i)];
    }
    double cutoff_x() const { return 2.0 * na_x / (wavelength_nm * 1e-6); }
    double cutoff_y() const { return 2.0 * na_y / (wavelength_nm * 1e-6); }
    // MTF bei der normierten Frequenz ν = f/f_c ∈ [0, 1] (linear zwischen den Stützstellen).
    double mtf_sagittal_at(double nu) const;
    double mtf_tangential_at(double nu) const;
};

// Legt ein Ergebnis für `pupil` mit FFT-Größe M an: Kennung, Geometrie (Δx, Δy, Airy-Radius),
// Vektoren in der richtigen Größe (Werte unbestimmt). Wirft std::invalid_argument (M keine
// Zweierpotenz, M < 2N, M > kMaxPsfFft, Pupille unvollständig).
PsfResult prepare_psf_result(const PsfPupil& pupil, int fft_size);

// Kennzahlen aus PSF und Pupille (Host, sequenziell, für CPU- und GPU-Ergebnisse dieselbe
// Funktion): transmitted, energy, strehl_fft, strehl_origin, strehl (mit Bezugspunkt), peak.
void finish_psf(PsfResult& result, const PsfPupil& pupil);

// Kleinste-Quadrate-Fit OPD ≈ c₀ + t_x x + t_y y über die transmittierenden Rasterpunkte (gleiche
// Gewichte, zeilenweise summiert; Kippung t in mm pro normierter Pupillenkoordinate). false bei
// weniger als drei Punkten oder singulärer Matrix. Bezugspunkt von PsfResult::strehl:
// ((flip_x ? −t_x : t_x)/NA_x, (flip_y ? −t_y : t_y)/NA_y).
bool psf_tilt_fit(const PsfPupil& pupil, double& tilt_x, double& tilt_y);

// CPU-Referenz eines Tupels: Pupillenfunktion in ein M×M-Feld (Ecke (0, 0)), FFT vorwärts,
// I = |U|²/n_voll², PSF zentriert (und gespiegelt bei divergenter Welle), FFT⁻¹ von I für die
// MTF-Schnitte, dann finish_psf.
PsfResult psf_cpu(const PsfPupil& pupil, int fft_size);

// Alle Tupel der Variante v (make_psf_pupils, M = psf_fft_size(N, settings.padding)).
std::vector<PsfResult> psf_cpu(const SystemBatchHost& batch, const EvalResult& result, int v,
                               const PsfSettings& settings = {});

// ---- Speicher auf dem Gerät --------------------------------------------------------------

// Ablage der Puffer der PsfEngine für eine Tranche von T Pupillen (Bereiche auf
// kBatchAlignment ausgerichtet); der Arbeitsspeicher von cuFFT kommt dahinter dazu.
struct PsfDeviceLayout {
    BufferRegion spectrum; // T·M² · 16 (komplex: Pupille, Spektrum, Intensität, OTF)
    BufferRegion psf;      // T·M² · 8 (zentrierte PSF)
    BufferRegion opd;      // T·N² · 8 (OPD-Karten)
    BufferRegion params;   // T · 24 (PsfPupilParams)
    BufferRegion mtf;      // T·2(N + 1) · 8 (MTF sagittal, tangential)
    std::size_t total_bytes = 0;
};

// Wirft std::invalid_argument bei grid < 1, fft_size < 2·grid oder pupils < 1.
PsfDeviceLayout make_psf_device_layout(int grid, int fft_size, int pupils);

} // namespace mko
