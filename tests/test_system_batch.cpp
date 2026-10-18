// SystemBatch (M2-T1): Indexierung, Varianten, Aufbau aus Systemdateien, CPU-Trace über eine
// Batch-Variante gegen die Einzeltabelle.

#include "mikroptim/batch/batch_trace_cpu.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/pupil/quadrature.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using mko::BatchShape;
using mko::BatchTopology;
using mko::SourceKind;
using mko::SurfaceKind;
using mko::SystemBatchHost;

namespace {

const std::string kSystems = MKO_SOURCE_DIR "/data/systems/";

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

// Topologie mit zwei Geisterplatten (Flächen 2 und 4), alle Größen verschieden.
BatchTopology ghost_topology() {
    BatchTopology t;
    t.surfaces = 6;
    t.wavelengths = 3;
    t.fields = 4;
    t.ghost_plates = 2;
    t.ghost_terms = 5;
    t.type = {SurfaceKind::Stop,   SurfaceKind::Sphere,     SurfaceKind::GhostPlate,
              SurfaceKind::Sphere, SurfaceKind::GhostPlate, SurfaceKind::Plane};
    return t;
}

// Eindeutiger Wert pro Feld und Indextupel (exakt darstellbar).
double code(int array, int v, int a = 0, int b = 0) {
    return array * 1e6 + v * 1e3 + a * 10.0 + b + 0.25;
}

// Bitgleichheit zweier CPU-Traces (Status, Fläche, Endzustand jedes Strahls).
void expect_identical(const mko::CpuTraceResult& a, const mko::CpuTraceResult& b) {
    ASSERT_EQ(a.fields, b.fields);
    ASSERT_EQ(a.wavelengths, b.wavelengths);
    ASSERT_EQ(a.samples, b.samples);
    ASSERT_EQ(a.rays.size(), b.rays.size());
    for (std::size_t i = 0; i < a.rays.size(); ++i) {
        ASSERT_EQ(a.status[i], b.status[i]) << "Strahl " << i;
        ASSERT_EQ(a.failed_surface[i], b.failed_surface[i]) << "Strahl " << i;
        if (a.status[i] != mko::TraceStatus::Ok) {
            continue;
        }
        const auto& p = a.rays[i];
        const auto& q = b.rays[i];
        ASSERT_TRUE(same_bits(p.p.x, q.p.x) && same_bits(p.p.y, q.p.y) && same_bits(p.p.z, q.p.z) &&
                    same_bits(p.d.x, q.d.x) && same_bits(p.d.y, q.d.y) && same_bits(p.d.z, q.d.z) &&
                    same_bits(p.opl, q.opl))
            << "Strahl " << i;
    }
}

} // namespace

// ---- Indexformeln ------------------------------------------------------------------------

TEST(BatchShape, IndexFormulas) {
    const BatchShape sh{3, 4, 2, 3, 2, 5}; // V, S, L, F, G, K
    EXPECT_EQ(sh.surface_at(0, 0), 0u);
    EXPECT_EQ(sh.surface_at(2, 3), 11u);        // 2·4 + 3
    EXPECT_EQ(sh.index_at(2, 3, 1), 23u);       // (2·4 + 3)·2 + 1
    EXPECT_EQ(sh.field_at(1, 2), 5u);           // 1·3 + 2
    EXPECT_EQ(sh.ghost_coeff_at(2, 1, 4), 29u); // (2·2 + 1)·5 + 4
    EXPECT_EQ(sh.ghost_index_at(2, 1, 1), 11u); // (2·2 + 1)·2 + 1
    EXPECT_EQ(sh.surface_stride(), 4u);
    EXPECT_EQ(sh.index_stride(), 8u);
    EXPECT_EQ(sh.field_stride(), 3u);
    EXPECT_EQ(sh.ghost_coeff_stride(), 10u);
    EXPECT_EQ(sh.ghost_index_stride(), 4u);
}

// Jede Formel bildet die Indextupel in lexikographischer Reihenfolge (Variante außen) lückenlos
// und kollisionsfrei auf 0, 1, 2, … ab, also bijektiv auf [0, V · Stride).
TEST(BatchShape, IndexFormulasAreBijectiveAndRowMajor) {
    const BatchShape sh{7, 6, 3, 4, 2, 5};
    std::size_t next = 0;
    for (int v = 0; v < sh.variants; ++v) {
        for (int s = 0; s < sh.surfaces; ++s) {
            ASSERT_EQ(sh.surface_at(v, s), next++);
        }
    }
    EXPECT_EQ(next, 7u * sh.surface_stride());
    next = 0;
    for (int v = 0; v < sh.variants; ++v) {
        for (int s = 0; s < sh.surfaces; ++s) {
            for (int l = 0; l < sh.wavelengths; ++l) {
                ASSERT_EQ(sh.index_at(v, s, l), next++);
            }
        }
    }
    EXPECT_EQ(next, 7u * sh.index_stride());
    next = 0;
    for (int v = 0; v < sh.variants; ++v) {
        for (int f = 0; f < sh.fields; ++f) {
            ASSERT_EQ(sh.field_at(v, f), next++);
        }
    }
    EXPECT_EQ(next, 7u * sh.field_stride());
    next = 0;
    for (int v = 0; v < sh.variants; ++v) {
        for (int g = 0; g < sh.ghost_plates; ++g) {
            for (int k = 0; k < sh.ghost_terms; ++k) {
                ASSERT_EQ(sh.ghost_coeff_at(v, g, k), next++);
            }
        }
    }
    EXPECT_EQ(next, 7u * sh.ghost_coeff_stride());
    next = 0;
    for (int v = 0; v < sh.variants; ++v) {
        for (int g = 0; g < sh.ghost_plates; ++g) {
            for (int l = 0; l < sh.wavelengths; ++l) {
                ASSERT_EQ(sh.ghost_index_at(v, g, l), next++);
            }
        }
    }
    EXPECT_EQ(next, 7u * sh.ghost_index_stride());
}

// ---- Host-Container ----------------------------------------------------------------------

TEST(SystemBatchHost, SizesAndGhostSurfaces) {
    const SystemBatchHost b(ghost_topology(), 7);
    EXPECT_EQ(b.variants(), 7);
    EXPECT_EQ(b.curvature_data().size(), 7u * 6u);
    EXPECT_EQ(b.thickness_data().size(), 7u * 6u);
    EXPECT_EQ(b.index_data().size(), 7u * 6u * 3u);
    EXPECT_EQ(b.pupil_radius_data().size(), 7u);
    EXPECT_EQ(b.field_angle_data().size(), 7u * 4u);
    EXPECT_EQ(b.field_target_data().size(), 7u * 4u);
    EXPECT_EQ(b.ghost_coeff_data().size(), 7u * 2u * 5u);
    EXPECT_EQ(b.ghost_index_data().size(), 7u * 2u * 3u);
    EXPECT_EQ(b.wavelength_nm_data().size(), 3u);
    EXPECT_EQ(b.wavelength_weight_data().size(), 3u);
    EXPECT_EQ(b.field_weight_data().size(), 4u);
    EXPECT_EQ(b.ghost_norm_radius_data().size(), 2u);
    ASSERT_EQ(b.ghost_surface_data().size(), 2u);
    EXPECT_EQ(b.ghost_surface(0), 2);
    EXPECT_EQ(b.ghost_surface(1), 4);
    for (double x : b.index_data()) {
        EXPECT_EQ(x, 0.0);
    }
}

// Werte über die Zugriffsfunktionen schreiben und über Zugriffsfunktionen, Rohfelder an der
// Formelposition, die Batch-Sicht und die Flächentabelle der Variante zurücklesen.
TEST(SystemBatchHost, WriteReadRoundtripAllIndices) {
    SystemBatchHost b(ghost_topology(), 7);
    const BatchShape sh = b.shape();
    for (int l = 0; l < sh.wavelengths; ++l) {
        b.wavelength_nm(l) = code(20, 0, l);
        b.wavelength_weight(l) = code(21, 0, l);
    }
    for (int f = 0; f < sh.fields; ++f) {
        b.field_weight(f) = code(22, 0, f);
    }
    for (int g = 0; g < sh.ghost_plates; ++g) {
        b.ghost_norm_radius(g) = code(23, 0, g);
    }
    for (int v = 0; v < sh.variants; ++v) {
        b.pupil_radius(v) = code(4, v);
        for (int s = 0; s < sh.surfaces; ++s) {
            b.curvature(v, s) = code(1, v, s);
            b.thickness(v, s) = code(2, v, s);
            for (int l = 0; l < sh.wavelengths; ++l) {
                b.index(v, s, l) = code(3, v, s, l);
            }
        }
        for (int f = 0; f < sh.fields; ++f) {
            b.field_angle(v, f) = code(5, v, f);
            b.field_target(v, f) = code(6, v, f);
        }
        for (int g = 0; g < sh.ghost_plates; ++g) {
            for (int k = 0; k < sh.ghost_terms; ++k) {
                b.ghost_coeff(v, g, k) = code(7, v, g, k);
            }
            for (int l = 0; l < sh.wavelengths; ++l) {
                b.ghost_index(v, g, l) = code(8, v, g, l);
            }
        }
    }

    const mko::SystemBatchView<double> view = b.view();
    EXPECT_EQ(view.shape.variants, 7);
    for (int l = 0; l < sh.wavelengths; ++l) {
        EXPECT_EQ(view.wavelength_nm[l], code(20, 0, l));
        EXPECT_EQ(view.wavelength_weight[l], code(21, 0, l));
    }
    for (int f = 0; f < sh.fields; ++f) {
        EXPECT_EQ(view.field_weight[f], code(22, 0, f));
    }
    for (int g = 0; g < sh.ghost_plates; ++g) {
        EXPECT_EQ(view.ghost_norm_radius[g], code(23, 0, g));
        EXPECT_EQ(view.ghost_surface[g], b.ghost_surface(g));
    }
    for (int v = 0; v < sh.variants; ++v) {
        EXPECT_EQ(b.pupil_radius(v), code(4, v));
        EXPECT_EQ(view.pupil_radius[v], code(4, v));
        const mko::SurfaceTableView<double> t = b.surface_table(v);
        EXPECT_EQ(t.surfaces, sh.surfaces);
        EXPECT_EQ(t.wavelengths, sh.wavelengths);
        for (int s = 0; s < sh.surfaces; ++s) {
            EXPECT_EQ(b.curvature(v, s), code(1, v, s));
            EXPECT_EQ(b.curvature_data()[sh.surface_at(v, s)], code(1, v, s));
            EXPECT_EQ(view.curvature[sh.surface_at(v, s)], code(1, v, s));
            EXPECT_EQ(t.curvature[s], code(1, v, s));
            EXPECT_EQ(b.thickness(v, s), code(2, v, s));
            EXPECT_EQ(view.thickness[sh.surface_at(v, s)], code(2, v, s));
            EXPECT_EQ(t.thickness[s], code(2, v, s));
            EXPECT_EQ(t.type[s], b.type(s));
            for (int l = 0; l < sh.wavelengths; ++l) {
                EXPECT_EQ(b.index(v, s, l), code(3, v, s, l));
                EXPECT_EQ(b.index_data()[sh.index_at(v, s, l)], code(3, v, s, l));
                EXPECT_EQ(view.index[sh.index_at(v, s, l)], code(3, v, s, l));
                EXPECT_EQ(t.index[s * t.wavelengths + l], code(3, v, s, l));
            }
        }
        EXPECT_EQ(b.image_distance(v), code(2, v, sh.surfaces - 1));
        EXPECT_EQ(view.image_distance(v), code(2, v, sh.surfaces - 1));
        for (int f = 0; f < sh.fields; ++f) {
            EXPECT_EQ(b.field_angle(v, f), code(5, v, f));
            EXPECT_EQ(view.field_angle[sh.field_at(v, f)], code(5, v, f));
            EXPECT_EQ(b.field_target(v, f), code(6, v, f));
            EXPECT_EQ(view.field_target[sh.field_at(v, f)], code(6, v, f));
        }
        for (int g = 0; g < sh.ghost_plates; ++g) {
            for (int k = 0; k < sh.ghost_terms; ++k) {
                EXPECT_EQ(b.ghost_coeff(v, g, k), code(7, v, g, k));
                EXPECT_EQ(view.ghost_coeff[sh.ghost_coeff_at(v, g, k)], code(7, v, g, k));
            }
            for (int l = 0; l < sh.wavelengths; ++l) {
                EXPECT_EQ(b.ghost_index(v, g, l), code(8, v, g, l));
                EXPECT_EQ(view.ghost_index[sh.ghost_index_at(v, g, l)], code(8, v, g, l));
            }
        }
    }
}

TEST(SystemBatchHost, AccessorsCheckEveryIndex) {
    SystemBatchHost b(ghost_topology(), 2);
    // Der lineare Index (0, S) = (1, 0) existiert, die Fläche S aber nicht.
    EXPECT_THROW(b.curvature(0, 6), std::out_of_range);
    EXPECT_THROW(b.curvature(2, 0), std::out_of_range);
    EXPECT_THROW(b.thickness(-1, 0), std::out_of_range);
    EXPECT_THROW(b.index(0, 0, 3), std::out_of_range);
    EXPECT_THROW(b.index(0, 6, 0), std::out_of_range);
    EXPECT_THROW(b.pupil_radius(2), std::out_of_range);
    EXPECT_THROW(b.field_angle(0, 4), std::out_of_range);
    EXPECT_THROW(b.field_target(1, -1), std::out_of_range);
    EXPECT_THROW(b.ghost_coeff(0, 0, 5), std::out_of_range);
    EXPECT_THROW(b.ghost_coeff(0, 2, 0), std::out_of_range);
    EXPECT_THROW(b.ghost_index(0, 1, 3), std::out_of_range);
    EXPECT_THROW(b.wavelength_nm(3), std::out_of_range);
    EXPECT_THROW(b.field_weight(4), std::out_of_range);
    EXPECT_THROW(b.ghost_norm_radius(2), std::out_of_range);
    EXPECT_THROW(b.ghost_surface(2), std::out_of_range);
    EXPECT_THROW(b.type(6), std::out_of_range);
    EXPECT_THROW(b.surface_table(2), std::out_of_range);
}

TEST(SystemBatchHost, VariantOperations) {
    SystemBatchHost b(ghost_topology(), 2);
    for (int v = 0; v < 2; ++v) {
        for (int s = 0; s < 6; ++s) {
            b.curvature(v, s) = code(1, v, s);
            for (int l = 0; l < 3; ++l) {
                b.index(v, s, l) = code(3, v, s, l);
            }
        }
        b.pupil_radius(v) = code(4, v);
        b.field_target(v, 3) = code(6, v, 3);
        b.ghost_coeff(v, 1, 4) = code(7, v, 1, 4);
        b.ghost_index(v, 1, 2) = code(8, v, 1, 2);
    }
    // Vergrößern: neue Varianten 0, alte unverändert
    b.resize(4);
    EXPECT_EQ(b.variants(), 4);
    EXPECT_EQ(b.curvature(1, 5), code(1, 1, 5));
    EXPECT_EQ(b.curvature(3, 5), 0.0);
    EXPECT_EQ(b.ghost_coeff(3, 1, 4), 0.0);

    // Kopie anhängen
    const int v4 = b.add_variant_copy(1);
    EXPECT_EQ(v4, 4);
    EXPECT_EQ(b.variants(), 5);
    for (int s = 0; s < 6; ++s) {
        EXPECT_EQ(b.curvature(4, s), code(1, 1, s));
        for (int l = 0; l < 3; ++l) {
            EXPECT_EQ(b.index(4, s, l), code(3, 1, s, l));
        }
    }
    EXPECT_EQ(b.pupil_radius(4), code(4, 1));
    EXPECT_EQ(b.field_target(4, 3), code(6, 1, 3));
    EXPECT_EQ(b.ghost_coeff(4, 1, 4), code(7, 1, 1, 4));
    EXPECT_EQ(b.ghost_index(4, 1, 2), code(8, 1, 1, 2));

    // Kopieren in die Mitte lässt die Nachbarn unverändert
    b.copy_variant(0, 2);
    EXPECT_EQ(b.curvature(2, 0), code(1, 0, 0));
    EXPECT_EQ(b.curvature(1, 5), code(1, 1, 5));
    EXPECT_EQ(b.curvature(3, 0), 0.0);

    // Kopie aus einem anderen Batch gleicher Topologie
    SystemBatchHost other(ghost_topology(), 1);
    other.pupil_radius(0) = 42.0;
    other.wavelength_nm(0) = 999.0; // gemeinsames Feld wird nicht übernommen
    b.copy_variant_from(other, 0, 3);
    EXPECT_EQ(b.pupil_radius(3), 42.0);
    EXPECT_EQ(b.wavelength_nm(0), 0.0);

    // Verkleinern
    b.resize(1);
    EXPECT_EQ(b.variants(), 1);
    EXPECT_EQ(b.curvature_data().size(), 6u);
    EXPECT_THROW(b.add_variant_copy(1), std::out_of_range);
    EXPECT_THROW(b.copy_variant(0, 1), std::out_of_range);
    EXPECT_THROW(b.resize(-1), std::invalid_argument);
}

TEST(SystemBatchHost, EqualityIsBitwise) {
    SystemBatchHost a(ghost_topology(), 2);
    SystemBatchHost b = a;
    EXPECT_TRUE(a == b);
    a.curvature(1, 1) = std::numeric_limits<double>::quiet_NaN();
    b.curvature(1, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(a == b); // gleiches Bitmuster
    b.thickness(0, 0) = -0.0;
    EXPECT_FALSE(a == b); // +0 und −0 unterscheiden sich
    b.thickness(0, 0) = 0.0;
    EXPECT_TRUE(a == b);
    b.source_na = 0.5;
    EXPECT_FALSE(a == b);
    b.source_na = 0.0;
    b.resize(3);
    EXPECT_FALSE(a == b);
}

TEST(SystemBatchHost, InconsistentTopologyThrows) {
    const auto expect_invalid = [](BatchTopology t) {
        EXPECT_THROW(SystemBatchHost(std::move(t), 1), std::invalid_argument);
    };
    BatchTopology t = ghost_topology();
    EXPECT_NO_THROW(mko::validate_topology(t));

    t = ghost_topology();
    t.surfaces = 5; // Typliste hat 6 Einträge
    expect_invalid(t);
    t = ghost_topology();
    t.ghost_plates = 1; // zwei Flächen vom Typ GhostPlate
    expect_invalid(t);
    t = ghost_topology();
    t.ghost_terms = 0; // G > 0 verlangt K >= 1
    expect_invalid(t);
    t = ghost_topology();
    t.wavelengths = 0;
    expect_invalid(t);
    t = ghost_topology();
    t.fields = 0;
    expect_invalid(t);
    t = ghost_topology();
    t.source = SourceKind::Point; // Punktquelle verlangt F = 1
    expect_invalid(t);
    t = ghost_topology();
    t.type[3] = static_cast<SurfaceKind>(7);
    expect_invalid(t);
    t = ghost_topology();
    t.source = static_cast<SourceKind>(5);
    expect_invalid(t);

    BatchTopology plain;
    plain.surfaces = 1;
    plain.wavelengths = 1;
    plain.fields = 1;
    plain.type = {SurfaceKind::Stop};
    expect_invalid(plain); // S < 2
    plain.surfaces = mko::kMaxSurfaces + 1;
    plain.type.assign(static_cast<std::size_t>(plain.surfaces), SurfaceKind::Plane);
    expect_invalid(plain); // S > kMaxSurfaces
    plain.surfaces = 3;
    plain.type.assign(3, SurfaceKind::Plane);
    plain.ghost_terms = 2; // ohne Geisterplatten K = 0
    expect_invalid(plain);
    plain.ghost_terms = 0;
    EXPECT_NO_THROW(SystemBatchHost(plain, 0));
    EXPECT_THROW(SystemBatchHost(plain, -1), std::invalid_argument);

    // Variante aus einem Batch anderer Topologie
    SystemBatchHost a(ghost_topology(), 1);
    const SystemBatchHost b(plain, 1);
    EXPECT_THROW(a.copy_variant_from(b, 0, 0), std::invalid_argument);
}

// ---- Aufbau aus Systemdateien ------------------------------------------------------------

TEST(MakeSystemBatch, SingleLensFromFile) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_bk7.json");
    const SystemBatchHost b = mko::make_system_batch(ls, 4);
    EXPECT_EQ(b.variants(), 4);
    EXPECT_EQ(b.surfaces(), 3);
    EXPECT_EQ(b.wavelengths(), 3);
    EXPECT_EQ(b.fields(), 3);
    EXPECT_EQ(b.ghost_plates(), 0);
    EXPECT_EQ(b.ghost_terms(), 0);
    EXPECT_EQ(b.source(), SourceKind::Collimated);
    EXPECT_EQ(b.reference_wavelength_nm, 587.56);
    EXPECT_NO_THROW(b.validate_values());

    const double f = mko::compute_paraxial(ls.system, ls.catalog).focal_length;
    for (int v = 0; v < b.variants(); ++v) {
        for (int s = 0; s < 3; ++s) {
            EXPECT_EQ(b.type(s), ls.table.type[static_cast<std::size_t>(s)]);
            EXPECT_EQ(b.curvature(v, s), ls.table.curvature[static_cast<std::size_t>(s)]);
            EXPECT_EQ(b.thickness(v, s), ls.table.thickness[static_cast<std::size_t>(s)]);
            for (int l = 0; l < 3; ++l) {
                EXPECT_EQ(b.index(v, s, l), ls.index.at(s, l));
            }
        }
        EXPECT_EQ(b.image_distance(v), 99.05);
        EXPECT_EQ(b.pupil_radius(v), 10.0);
        for (int fi = 0; fi < 3; ++fi) {
            const double theta = ls.system.source.field_angles_deg[static_cast<std::size_t>(fi)] *
                                 (std::numbers::pi / 180.0);
            EXPECT_EQ(b.field_angle(v, fi), theta);
            EXPECT_EQ(b.field_target(v, fi), fi == 0 ? 0.0 : f * std::tan(theta));
        }
    }
    for (int l = 0; l < 3; ++l) {
        EXPECT_EQ(b.wavelength_nm(l),
                  ls.system.spectrum.wavelengths_nm[static_cast<std::size_t>(l)]);
        EXPECT_EQ(b.wavelength_weight(l), 1.0);
    }
    for (int fi = 0; fi < 3; ++fi) {
        EXPECT_EQ(b.field_weight(fi), 1.0);
    }
    EXPECT_GT(b.field_target(0, 2), 3.5); // f ≈ 101 mm, tan 2° ≈ 0,035
    EXPECT_LT(b.field_target(0, 2), 3.6);

    // Gleiches Ergebnis über Beschreibung + Brechzahltabelle (λ_ref liegt im Spektrum)
    EXPECT_TRUE(mko::make_system_batch(ls.system, ls.index, 4) == b);
}

TEST(MakeSystemBatch, PlateFromFileIsAfocal) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "planplatte_bk7.json");
    const SystemBatchHost b = mko::make_system_batch(ls);
    EXPECT_EQ(b.variants(), 1);
    EXPECT_EQ(b.fields(), 2);
    EXPECT_EQ(b.wavelengths(), 1);
    EXPECT_EQ(b.type(0), SurfaceKind::Stop);
    EXPECT_EQ(b.type(1), SurfaceKind::Plane);
    EXPECT_EQ(b.thickness(0, 1), 10.0);
    EXPECT_EQ(b.image_distance(0), 20.0);
    EXPECT_EQ(b.pupil_radius(0), 5.0);
    EXPECT_EQ(b.field_angle(0, 1), 10.0 * (std::numbers::pi / 180.0));
    // Afokal: keine Brennweite, Feldziel unbestimmt -> 0
    EXPECT_EQ(b.field_target(0, 0), 0.0);
    EXPECT_EQ(b.field_target(0, 1), 0.0);
    EXPECT_NO_THROW(b.validate_values());
}

TEST(MakeSystemBatch, FieldTargetFromTargetFocalLength) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "einzellinse_bk7.json");
    sys.target = mko::Target{};
    sys.target->focal_length = 100.0;
    const mko::LoadedSystem ls = mko::prepare_system(sys, MKO_SOURCE_DIR);
    const SystemBatchHost b = mko::make_system_batch(ls);
    EXPECT_EQ(b.field_target(0, 1), 100.0 * std::tan(b.field_angle(0, 1)));
    // Ausdrücklich übergebene Brennweite hat Vorrang
    const SystemBatchHost c = mko::make_system_batch(ls.system, ls.index, 1, 50.0);
    EXPECT_EQ(c.field_target(0, 2), 50.0 * std::tan(c.field_angle(0, 2)));
    EXPECT_THROW(
        mko::make_system_batch(ls.system, ls.index, 1, std::numeric_limits<double>::infinity()),
        std::invalid_argument);
}

TEST(MakeSystemBatch, Errors) {
    const mko::LoadedSystem lens = mko::load_system(kSystems + "einzellinse_bk7.json");
    const mko::LoadedSystem plate = mko::load_system(kSystems + "planplatte_bk7.json");
    EXPECT_THROW(mko::make_system_batch(lens, 0), std::invalid_argument);
    // Brechzahltabelle mit 1 statt 3 Wellenlängen
    EXPECT_THROW(mko::make_system_batch(lens.system, plate.index, 1), std::invalid_argument);
    mko::IndexTable wrong = lens.index;
    wrong.surfaces = 2;
    EXPECT_THROW(mko::make_system_batch(lens.system, wrong, 1), std::invalid_argument);
}

TEST(SystemBatchHost, ValidateValues) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_bk7.json");
    SystemBatchHost b = mko::make_system_batch(ls, 3);
    EXPECT_NO_THROW(b.validate_values());
    SystemBatchHost c = b;
    c.index(2, 1, 0) = 0.0;
    EXPECT_THROW(c.validate_values(), std::invalid_argument);
    c = b;
    c.curvature(1, 2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(c.validate_values(), std::invalid_argument);
    c = b;
    c.pupil_radius(2) = 0.0;
    EXPECT_THROW(c.validate_values(), std::invalid_argument);
    c = b;
    c.reference_wavelength_nm = 0.0;
    EXPECT_THROW(c.validate_values(), std::invalid_argument);
    c = b;
    c.field_weight(1) = -1.0;
    EXPECT_THROW(c.validate_values(), std::invalid_argument);
    // Leerer Batch (alles 0) ist nicht gültig
    EXPECT_THROW(SystemBatchHost(ghost_topology(), 1).validate_values(), std::invalid_argument);
}

// ---- CPU-Trace über eine Batch-Variante ---------------------------------------------------

// Abnahme M2-T1 (Layout): Der Trace über surface_table(v) und batch_source_ray ist bitgleich
// zum CPU-Tracer über die Einzeltabelle des Systems, auch für gestörte Varianten, die aus
// anderen Batches hineinkopiert wurden.
TEST(BatchTrace, VariantMatchesSingleTableBitwise) {
    const mko::SystemDescription base = mko::read_system_file(kSystems + "einzellinse_bk7.json");
    const mko::LoadedSystem ls = mko::prepare_system(base, MKO_SOURCE_DIR);
    const auto q = mko::make_pupil_quadrature(8, 8, mko::PupilMode::Full);
    constexpr int kVariants = 6;
    SystemBatchHost batch = mko::make_system_batch(ls, kVariants);

    std::vector<mko::LoadedSystem> systems;
    systems.push_back(ls);
    for (int k = 1; k < kVariants; ++k) {
        mko::SystemDescription sys = base;
        sys.surfaces[1].curvature *= 1.0 + 0.05 * k;
        sys.surfaces[1].thickness += 0.3 * k;
        sys.surfaces[2].thickness -= 0.7 * k;
        sys.surfaces[0].semi_diameter = 9.0 + k;
        sys.source.field_angles_deg = {0.0, 0.5 * k, 3.0 + k};
        if (k == kVariants - 1) {
            // große Blende und stark gekrümmte Fläche: verfehlte und totalreflektierte Strahlen
            sys.surfaces[0].semi_diameter = 14.0;
            sys.surfaces[2].curvature = -0.08;
        }
        systems.push_back(mko::prepare_system(sys, MKO_SOURCE_DIR));
        const SystemBatchHost single = mko::make_system_batch(systems.back());
        batch.copy_variant_from(single, 0, k);
    }

    std::size_t failed = 0;
    for (int v = 0; v < kVariants; ++v) {
        const auto& sv = systems[static_cast<std::size_t>(v)];
        const mko::CpuTraceResult expected = mko::trace_system_cpu(sv.system, sv.table, q);
        const mko::CpuTraceResult actual = mko::trace_variant_cpu(batch, v, q);
        expect_identical(expected, actual);
        failed += actual.failed_count();
    }
    EXPECT_GT(failed, 0u);
    EXPECT_THROW(mko::trace_variant_cpu(batch, kVariants, q), std::out_of_range);
}

TEST(BatchTrace, PlateMatchesSingleTableBitwise) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "planplatte_bk7.json");
    const auto q = mko::make_pupil_quadrature(6, 6, mko::PupilMode::Half);
    const SystemBatchHost batch = mko::make_system_batch(ls, 2);
    expect_identical(mko::trace_system_cpu(ls.system, ls.table, q),
                     mko::trace_variant_cpu(batch, 1, q));
}

TEST(BatchTrace, PointSourceMatchesSingleTableBitwise) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "einzellinse_bk7.json");
    sys.source = mko::Source{};
    sys.source.type = mko::SourceType::Point;
    sys.source.point = {0.0, 0.5, -50.0};
    sys.source.numerical_aperture = 0.1;
    const mko::LoadedSystem ls = mko::prepare_system(sys, MKO_SOURCE_DIR);
    const SystemBatchHost batch = mko::make_system_batch(ls, 2);
    EXPECT_EQ(batch.source(), SourceKind::Point);
    EXPECT_EQ(batch.fields(), 1);
    EXPECT_EQ(batch.source_na, 0.1);
    EXPECT_EQ(batch.source_point[2], -50.0);
    EXPECT_EQ(batch.field_target(1, 0), 0.0);
    EXPECT_NO_THROW(batch.validate_values());
    const auto q = mko::make_pupil_quadrature(6, 6, mko::PupilMode::Full);
    expect_identical(mko::trace_system_cpu(ls.system, ls.table, q),
                     mko::trace_variant_cpu(batch, 1, q));

    SystemBatchHost bad = batch;
    bad.source_na = 1.5; // NA >= n
    EXPECT_THROW(mko::trace_variant_cpu(bad, 0, q), std::invalid_argument);
}
