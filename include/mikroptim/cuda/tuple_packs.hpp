#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Tupelpakete der Kernel RESIDUALS und MERIT (M7-T4, docs/architektur.md, "Kernel-Abbildung"):
// Ein Block rechnet ein Paket aufeinanderfolgender Tupel t = f · L + l derselben Variante. Die
// Strahlen eines Pakets liegen in der Residuenablage hintereinander
// (offset(t + 1) = offset(t) + n_f), Durchlauf 1 verteilt sie dicht auf die Threads; Kolben und
// Quadratsumme bildet jedes Tupel unabhängig vom Paket. Hier nur die Planung auf dem Host.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::cuda {

// Höchstzahl der Tupel pro Paket (Felder im Shared Memory des Kernels).
inline constexpr int kMaxPackTuples = 16;

// Ein Paket: Tupel first_tuple … first_tuple + tuples − 1 (t = f · L + l) einer Variante.
// Gleich für alle Varianten eines Laufs; wird per cudaMemcpy auf das Gerät kopiert.
struct TuplePack {
    int first_tuple;        // t0 = f0 · L + l0
    int tuples;             // 1 … kMaxPackTuples
    int rays;               // Σ n_f der Tupel
    int first_wavelength;   // Fenster der Brechzahlen im Shared Memory: first_wavelength …
    int wavelengths;        // first_wavelength + wavelengths − 1
    std::int64_t first_row; // offset(f0, l0): Zeile des ersten Strahls in der Ablage
};

// Bytes der Tabellen eines Pakets im dynamischen Shared Memory (Rechentyp double oder df64,
// 8 Byte): Geisterkoeffizienten [G · K], Brechzahlen der Geistergläser [G][W], Normierungsradien
// [G], Brechzahlen der Flächen [S][W] für ein Fenster von W Wellenlängen und, nur bei nicht
// koaxialem Batch (transforms, M8-T3), die Transformationsdaten [S][4] mit ihren Sinus und
// Kosinus [S][4].
MKO_HD inline std::size_t pack_table_bytes(int surfaces, int ghost_plates, int ghost_terms,
                                           int wavelengths, bool transforms = false) {
    const auto S = static_cast<std::size_t>(surfaces);
    const auto G = static_cast<std::size_t>(ghost_plates);
    const auto K = static_cast<std::size_t>(ghost_terms);
    const auto W = static_cast<std::size_t>(wavelengths);
    const std::size_t X =
        transforms ? 2 * S * static_cast<std::size_t>(kSurfaceTransformValues) : 0;
    return (G * K + G * W + G + S * W + X) * sizeof(double);
}

// Pakete eines Laufs: Tupel in der Reihenfolge t = f · L + l, gierig zusammengefasst, solange
// das Paket höchstens max_rays Strahlen und kMaxPackTuples Tupel hat und seine Tabellen
// (pack_table_bytes) plus bytes_per_ray pro Strahl in dynamic_budget Byte passen. Ein Paket
// innerhalb eines Feldes lädt nur die Brechzahlen seiner Wellenlängen, ein Paket über eine
// Feldgrenze alle L. Jedes Paket hat mindestens ein Tupel, auch wenn dieses mehr als max_rays
// Strahlen hat. Wirft std::invalid_argument bei max_rays < 1 oder wenn schon ein einzelnes Tupel
// nicht in dynamic_budget passt. transforms: nicht koaxialer Batch (pack_table_bytes).
inline std::vector<TuplePack>
make_tuple_packs(const ResidualLayout& layout, const BatchShape& shape, int max_rays,
                 std::size_t bytes_per_ray, std::size_t dynamic_budget, bool transforms = false) {
    if (max_rays < 1) {
        throw std::invalid_argument("make_tuple_packs: max_rays muss >= 1 sein");
    }
    const int F = layout.fields();
    const int L = layout.wavelengths();
    const auto need = [&](int wavelengths, int rays) {
        return pack_table_bytes(shape.surfaces, shape.ghost_plates, shape.ghost_terms, wavelengths,
                                transforms) +
               static_cast<std::size_t>(rays) * bytes_per_ray;
    };
    std::vector<TuplePack> packs;
    TuplePack cur{};
    int cur_field = 0;
    for (int t = 0; t < F * L; ++t) {
        const int f = t / L;
        const int l = t % L;
        const int n = layout.samples(f);
        if (cur.tuples > 0) {
            const bool same_field = f == cur_field;
            const int lo = same_field ? cur.first_wavelength : 0;
            const int w = same_field ? l - cur.first_wavelength + 1 : L;
            if (cur.tuples < kMaxPackTuples && cur.rays + n <= max_rays &&
                need(w, cur.rays + n) <= dynamic_budget) {
                ++cur.tuples;
                cur.rays += n;
                cur.first_wavelength = lo;
                cur.wavelengths = w;
                continue;
            }
            packs.push_back(cur);
        }
        if (need(1, n) > dynamic_budget) {
            throw std::invalid_argument("make_tuple_packs: " + std::to_string(n) +
                                        " Stützstellen pro Feld passen nicht in das Shared Memory");
        }
        cur = TuplePack{t, 1, n, l, 1, layout.offset(f, l)};
        cur_field = f;
    }
    if (cur.tuples > 0) {
        packs.push_back(cur);
    }
    return packs;
}

// Größter Bedarf an dynamischem Shared Memory über die Pakete: Tabellen plus bytes_per_ray pro
// Strahl.
inline std::size_t pack_dynamic_bytes(const std::vector<TuplePack>& packs, const BatchShape& shape,
                                      std::size_t bytes_per_ray, bool transforms = false) {
    std::size_t bytes = 0;
    for (const TuplePack& p : packs) {
        bytes = std::max(bytes, pack_table_bytes(shape.surfaces, shape.ghost_plates,
                                                 shape.ghost_terms, p.wavelengths, transforms) +
                                    static_cast<std::size_t>(p.rays) * bytes_per_ray);
    }
    return bytes;
}

} // namespace mko::cuda
