// Tupelpakete der Kernel RESIDUALS und MERIT (M7-T4, CPU-Teil): Planung make_tuple_packs aus
// include/mikroptim/cuda/tuple_packs.hpp (reines C++). Die Bitgleichheit der Kernel für alle
// Paketgrößen prüft test_cuda_tuple_packs.cpp.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/cuda/tuple_packs.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using mko::BatchShape;
using mko::BatchTopology;
using mko::PupilMode;
using mko::ResidualLayout;
using mko::ResidualSettings;
using mko::SurfaceKind;
using mko::cuda::kMaxPackTuples;
using mko::cuda::make_tuple_packs;
using mko::cuda::pack_dynamic_bytes;
using mko::cuda::pack_table_bytes;
using mko::cuda::TuplePack;

namespace {

constexpr std::size_t kNoLimit = std::numeric_limits<std::size_t>::max();

// Topologie mit S Flächen, L Wellenlängen und F Feldern (Feld 0 auf der Achse, die übrigen
// halbe Pupille).
BatchTopology topology(int surfaces, int fields, int wavelengths) {
    BatchTopology t;
    t.surfaces = surfaces;
    t.wavelengths = wavelengths;
    t.fields = fields;
    t.type.assign(static_cast<std::size_t>(surfaces), SurfaceKind::Sphere);
    t.type[0] = SurfaceKind::Stop;
    for (int f = 0; f < fields; ++f) {
        t.pupil_mode.push_back(f == 0 ? PupilMode::Axis : PupilMode::Half);
    }
    return t;
}

// Form für die Planung; Geisterplatten gehen nur über ihre Zahl in den Speicherbedarf ein.
BatchShape shape_of(const BatchTopology& t, int ghost_plates = 0, int ghost_terms = 0) {
    BatchShape s{};
    s.surfaces = t.surfaces;
    s.wavelengths = t.wavelengths;
    s.fields = t.fields;
    s.ghost_plates = ghost_plates;
    s.ghost_terms = ghost_terms;
    return s;
}

// Invarianten jeder Planung: Pakete decken die Tupel t = 0 … F·L − 1 lückenlos in Reihenfolge
// ab, Strahlen und erste Zeile passen zur Ablage, Größen- und Tupelgrenzen, Wellenlängenfenster
// enthält alle Tupel des Pakets.
void check_packs(const std::vector<TuplePack>& packs, const ResidualLayout& layout, int max_rays,
                 const std::string& what) {
    SCOPED_TRACE(what);
    const int L = layout.wavelengths();
    int next = 0;
    std::int64_t rays = 0;
    for (const TuplePack& p : packs) {
        EXPECT_EQ(p.first_tuple, next);
        ASSERT_GE(p.tuples, 1);
        EXPECT_LE(p.tuples, kMaxPackTuples);
        const int f0 = p.first_tuple / L;
        EXPECT_EQ(p.first_row, layout.offset(f0, p.first_tuple % L));
        int n = 0;
        bool one_field = true;
        for (int j = 0; j < p.tuples; ++j) {
            const int t = p.first_tuple + j;
            const int f = t / L;
            const int l = t % L;
            n += layout.samples(f);
            one_field = one_field && f == f0;
            EXPECT_GE(l, p.first_wavelength);
            EXPECT_LT(l, p.first_wavelength + p.wavelengths);
        }
        if (!one_field) {
            EXPECT_EQ(p.first_wavelength, 0);
            EXPECT_EQ(p.wavelengths, L);
        }
        EXPECT_EQ(p.rays, n);
        if (p.tuples > 1) {
            EXPECT_LE(p.rays, max_rays);
        }
        rays += p.rays;
        next += p.tuples;
    }
    EXPECT_EQ(next, layout.fields() * L);
    EXPECT_EQ(rays, layout.rays_per_variant());
}

} // namespace

// Standardquadratur 10 × 8 mit 3 Feldern (Achse: 10 Strahlen, sonst 80) und 3 Wellenlängen
// (510 Strahlen pro Variante): Erwartete Pakete von Hand.
TEST(TuplePacks, StandardCaseByHand) {
    const BatchTopology t = topology(30, 3, 3);
    const ResidualLayout layout(t);
    ASSERT_EQ(layout.rays_per_variant(), 510);
    // 512: ein Paket mit allen 9 Tupeln (16 Warps für 510 Strahlen)
    const std::vector<TuplePack> one = make_tuple_packs(layout, shape_of(t), 512, 0, kNoLimit);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_EQ(one[0].tuples, 9);
    EXPECT_EQ(one[0].rays, 510);
    EXPECT_EQ(one[0].wavelengths, 3);
    // 256: 10 + 10 + 10 + 80 + 80 = 190 | 80 + 80 + 80 = 240 | 80
    const std::vector<TuplePack> three = make_tuple_packs(layout, shape_of(t), 256, 0, kNoLimit);
    ASSERT_EQ(three.size(), 3u);
    EXPECT_EQ(three[0].rays, 190);
    EXPECT_EQ(three[1].rays, 240);
    EXPECT_EQ(three[2].rays, 80);
    EXPECT_EQ(three[1].first_tuple, 5);      // (f = 1, l = 2), (f = 2, l = 0), (f = 2, l = 1)
    EXPECT_EQ(three[1].first_wavelength, 0); // über eine Feldgrenze: alle Wellenlängen
    EXPECT_EQ(three[1].wavelengths, 3);
    EXPECT_EQ(three[2].first_wavelength, 2);
    EXPECT_EQ(three[2].wavelengths, 1);
    // 1: ein Tupel pro Paket (Abbildung vor M7-T4)
    const std::vector<TuplePack> single = make_tuple_packs(layout, shape_of(t), 1, 0, kNoLimit);
    ASSERT_EQ(single.size(), 9u);
    for (std::size_t i = 0; i < single.size(); ++i) {
        EXPECT_EQ(single[i].tuples, 1);
        EXPECT_EQ(single[i].first_tuple, static_cast<int>(i));
        EXPECT_EQ(single[i].wavelengths, 1);
        EXPECT_EQ(single[i].first_wavelength, static_cast<int>(i % 3));
    }
    check_packs(one, layout, 512, "512");
    check_packs(three, layout, 256, "256");
    check_packs(single, layout, 1, "1");
}

// Invarianten für verschiedene Feld-/Wellenlängenzahlen, Quadraturen und Paketgrößen.
TEST(TuplePacks, CoverAllTuplesInOrder) {
    const int cases[][4] = {{8, 1, 1, 0},  {30, 3, 3, 0},  {40, 5, 7, 0},
                            {12, 4, 2, 3}, {64, 2, 20, 0}, {16, 30, 1, 0}};
    const ResidualSettings quads[] = {{10, 8}, {6, 5}, {12, 40}, {1, 1}};
    for (const auto& c : cases) {
        const BatchTopology t = topology(c[0], c[1], c[2]);
        for (const ResidualSettings& q : quads) {
            const ResidualLayout layout(t, q);
            for (const int max_rays : {1, 32, 128, 512, 4096}) {
                const std::string what =
                    "S=" + std::to_string(c[0]) + " F=" + std::to_string(c[1]) +
                    " L=" + std::to_string(c[2]) + " N×M=" + std::to_string(q.rings) + "×" +
                    std::to_string(q.spokes) + " P=" + std::to_string(max_rays);
                check_packs(make_tuple_packs(layout, shape_of(t, c[3], c[3] > 0 ? 4 : 0), max_rays,
                                             0, kNoLimit),
                            layout, max_rays, what);
            }
        }
    }
}

// Shared-Memory-Budget: Tabellen plus Bytes pro Strahl; ein einzelnes Tupel, das nicht passt,
// ist ein Fehler.
TEST(TuplePacks, SharedMemoryBudget) {
    EXPECT_EQ(pack_table_bytes(30, 0, 0, 3), 30u * 3u * 8u);
    EXPECT_EQ(pack_table_bytes(10, 2, 5, 4), (2u * 5u + 2u * 4u + 2u + 10u * 4u) * 8u);

    const BatchTopology t = topology(30, 3, 3);
    const ResidualLayout layout(t);
    const BatchShape s = shape_of(t, 2, 5);
    constexpr std::size_t kPerRay = 17;
    // Budget für genau ein Tupel mit 80 Strahlen und einer Wellenlänge
    const std::size_t tight = pack_table_bytes(30, 2, 5, 1) + 80 * kPerRay;
    const std::vector<TuplePack> packs = make_tuple_packs(layout, s, 4096, kPerRay, tight);
    check_packs(packs, layout, 4096, "knapp");
    for (const TuplePack& p : packs) {
        EXPECT_LE(pack_table_bytes(30, 2, 5, p.wavelengths) + p.rays * kPerRay, tight);
    }
    EXPECT_LE(pack_dynamic_bytes(packs, s, kPerRay), tight);
    EXPECT_EQ(packs.size(), 7u); // Achsfeld: drei Tupel à 10 Strahlen passen zusammen
    EXPECT_THROW((void)make_tuple_packs(layout, s, 4096, kPerRay, tight - 1),
                 std::invalid_argument);
    EXPECT_THROW((void)make_tuple_packs(layout, s, 0, kPerRay, kNoLimit), std::invalid_argument);
}

// Absicherung vor dem Kernelstart (check_tuple_packs): Jede Planung von make_tuple_packs wird
// angenommen, auch für die Form des Systems, bei dem in M6-T10 einmal "an illegal memory access"
// gemeldet wurde (39 Tabellenflächen, 3 Felder, 1 Wellenlänge, 6 Geisterplatten mit je 10
// Koeffizienten); jede einzelne Verfälschung eines Pakets, die im Kernel zu Zugriffen außerhalb
// der Puffer führte, wird mit std::logic_error abgelehnt.
TEST(TuplePacks, CheckAcceptsPlansAndRejectsCorruption) {
    using mko::cuda::check_tuple_packs;
    const int cases[][6] = {{8, 1, 1, 0, 0, 0},   {30, 3, 3, 0, 0, 0},  {40, 5, 7, 2, 4, 0},
                            {39, 3, 1, 6, 10, 0}, {39, 3, 1, 6, 10, 1}, {64, 2, 20, 0, 0, 0}};
    constexpr std::size_t kBudget = 40 * 1024;
    for (const auto& c : cases) {
        const BatchTopology t = topology(c[0], c[1], c[2]);
        const BatchShape s = shape_of(t, c[3], c[4]);
        const bool transforms = c[5] != 0;
        for (const ResidualSettings& q : {ResidualSettings{10, 8}, ResidualSettings{12, 40}}) {
            const ResidualLayout layout(t, q);
            for (const int max_rays : {1, 128, 512, 4096}) {
                for (const std::size_t per_ray : {std::size_t{0}, std::size_t{17}}) {
                    const std::vector<TuplePack> packs =
                        make_tuple_packs(layout, s, max_rays, per_ray, kBudget, transforms);
                    EXPECT_NO_THROW(
                        check_tuple_packs(packs, layout, s, per_ray, kBudget, transforms))
                        << "S=" << c[0] << " F=" << c[1] << " L=" << c[2] << " P=" << max_rays
                        << " Bytes/Strahl=" << per_ray;
                }
            }
        }
    }

    const BatchTopology t = topology(39, 3, 3);
    const ResidualLayout layout(t);
    const BatchShape s = shape_of(t, 6, 10);
    const std::vector<TuplePack> good = make_tuple_packs(layout, s, 256, 0, kNoLimit);
    ASSERT_EQ(good.size(), 3u);
    EXPECT_NO_THROW(check_tuple_packs(good, layout, s, 0, kNoLimit));
    const auto rejected = [&](const std::vector<TuplePack>& packs, std::size_t budget,
                              const char* what) {
        SCOPED_TRACE(what);
        EXPECT_THROW(check_tuple_packs(packs, layout, s, 0, budget), std::logic_error);
    };
    rejected({}, kNoLimit, "keine Pakete");
    std::vector<TuplePack> p = good;
    p.pop_back();
    rejected(p, kNoLimit, "letztes Paket fehlt");
    p = good;
    p.push_back(good.back());
    rejected(p, kNoLimit, "Paket doppelt");
    p = good;
    p[1].first_tuple += 1;
    rejected(p, kNoLimit, "Lücke");
    p = good;
    p[0].tuples = kMaxPackTuples + 1;
    rejected(p, kNoLimit, "zu viele Tupel");
    p = good;
    p[0].tuples = 0;
    rejected(p, kNoLimit, "leeres Paket");
    p = good;
    p[0].rays += 1;
    rejected(p, kNoLimit, "Strahlzahl");
    p = good;
    p[2].first_row += 1;
    rejected(p, kNoLimit, "erste Zeile");
    p = good;
    p[2].first_wavelength = 3;
    rejected(p, kNoLimit, "Fenster außerhalb");
    p = good;
    p[1].wavelengths = 1;
    rejected(p, kNoLimit, "Tupel außerhalb des Fensters");
    rejected(good, pack_dynamic_bytes(good, s, 0) - 1, "Budget");
    EXPECT_THROW(check_tuple_packs(good, layout, shape_of(topology(39, 2, 3), 6, 10), 0, kNoLimit),
                 std::logic_error);
}
