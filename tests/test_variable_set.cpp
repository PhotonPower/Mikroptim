// Variablenabbildung Vektor ↔ Batch (M3-T1): Roundtrip, adressierte Felder, Schranken,
// Systembeschreibung, Abbildung Dateifläche → Tabellenfläche.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using mko::BatchTopology;
using mko::SurfaceKind;
using mko::SurfaceSelection;
using mko::SystemBatchHost;
using mko::SystemDescription;
using mko::VariableSet;
using mko::VariableSpec;
using mko::VariableType;

namespace {

const std::string kSystems = MKO_SOURCE_DIR "/data/systems/";
constexpr double kInf = std::numeric_limits<double>::infinity();

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

VariableSpec spec(VariableType type, SurfaceSelection sel, std::vector<int> surfaces = {}) {
    VariableSpec s;
    s.type = type;
    s.selection = sel;
    s.surfaces = std::move(surfaces);
    return s;
}

// Mehrlinser mit allen Variablentypen: alle Krümmungen (7 Sphären), Dicken 0, 2, 5 und der
// Arbeitsabstand, teils mit Schranken.
mko::LoadedSystem mehrlinser_with_variables() {
    mko::LoadedSystem loaded = mko::load_system(kSystems + "testsystem_mehrlinser.json");
    mko::Variables vars;
    VariableSpec c = spec(VariableType::Curvature, SurfaceSelection::All);
    c.min = -0.2;
    c.max = 0.2;
    vars.items.push_back(c);
    VariableSpec t = spec(VariableType::Thickness, SurfaceSelection::List, {5, 0, 2});
    t.min = 0.5;
    t.typical = 2.0;
    vars.items.push_back(t);
    VariableSpec w = spec(VariableType::WorkingDistance, SurfaceSelection::None);
    w.min = 1.0;
    w.max = 40.0;
    vars.items.push_back(w);
    loaded.system.variables = vars;
    mko::validate_system(loaded.system);
    return loaded;
}

// Unterschiedliche, nicht runde Werte für Variante v (innerhalb der Schranken).
std::vector<double> sample_vector(const VariableSet& set, int v) {
    std::vector<double> x(static_cast<std::size_t>(set.size()));
    for (int k = 0; k < set.size(); ++k) {
        const double u = std::sin(1.0 + 0.7 * k + 1.3 * v); // in [-1, 1]
        x[static_cast<std::size_t>(k)] =
            set[k].type == VariableType::Curvature ? 0.1 * u : 3.0 + 0.9 * u + 0.1 * v;
    }
    return x;
}

} // namespace

TEST(VariableSet, BuildsFromMehrlinser) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    const SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    const VariableSet set(loaded.system, batch.topology());
    // Reihenfolge wie resolve_variables: Krümmungen der Sphären 1, 2, 3, 4, 6, 7, dann die
    // Dicken in Listenreihenfolge, zuletzt der Arbeitsabstand.
    const std::vector<std::string> names = {"c[1]", "c[2]", "c[3]", "c[4]", "c[6]",
                                            "c[7]", "t[5]", "t[0]", "t[2]", "w"};
    ASSERT_EQ(set.size(), 10);
    for (int k = 0; k < set.size(); ++k) {
        EXPECT_EQ(set[k].name, names[static_cast<std::size_t>(k)]);
        EXPECT_EQ(set[k].batch_surface, k == 9 ? batch.surfaces() - 1 : set[k].file_surface);
    }
    EXPECT_EQ(set[9].type, VariableType::WorkingDistance);
    EXPECT_EQ(set[9].file_surface, 7);
    EXPECT_EQ(set.lower()[0], -0.2);
    EXPECT_EQ(set.upper()[5], 0.2);
    EXPECT_EQ(set.lower()[6], 0.5);
    EXPECT_EQ(set.upper()[6], kInf);
    EXPECT_EQ(set.typical()[7], 2.0);
    EXPECT_EQ(set.typical()[0], mko::kDefaultTypicalCurvature);
    EXPECT_EQ(set.typical()[9], mko::kDefaultTypicalWorkingDistance);
    EXPECT_THROW((void)set[10], std::out_of_range);
    EXPECT_THROW((void)set[-1], std::out_of_range);
}

// Abnahme: Vektor → Batch → Vektor bitgleich, für alle Typen und mehrere Varianten.
TEST(VariableSet, RoundtripVectorBatchVector) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    SystemBatchHost batch = mko::make_system_batch(loaded, 6);
    const VariableSet set(loaded.system, batch.topology());
    std::vector<std::vector<double>> xs;
    for (int v = 0; v < batch.variants(); ++v) {
        xs.push_back(sample_vector(set, v));
        xs.back()[0] = -0.0; // Vorzeichen der Null bleibt erhalten
        set.apply(batch, v, xs.back());
    }
    for (int v = 0; v < batch.variants(); ++v) {
        const std::vector<double> y = set.extract(batch, v);
        ASSERT_EQ(y.size(), xs[static_cast<std::size_t>(v)].size());
        for (std::size_t k = 0; k < y.size(); ++k) {
            EXPECT_TRUE(same_bits(y[k], xs[static_cast<std::size_t>(v)][k])) << v << " " << k;
        }
    }
    // Batch → Vektor → Batch ändert nichts.
    const SystemBatchHost before = batch;
    std::vector<double> x(static_cast<std::size_t>(set.size()));
    for (int v = 0; v < batch.variants(); ++v) {
        set.extract(batch, v, x);
        set.apply(batch, v, x);
    }
    EXPECT_TRUE(batch == before);
}

// Ausgangswerte im Batch sind die Werte der Datei.
TEST(VariableSet, ExtractMatchesDescription) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    const SystemBatchHost batch = mko::make_system_batch(loaded, 2);
    const VariableSet set(loaded.system, batch.topology());
    const std::vector<double> from_file = set.extract(loaded.system);
    EXPECT_EQ(from_file, set.extract(batch, 1));
    EXPECT_EQ(from_file[0], 0.03333);
    EXPECT_EQ(from_file[6], 2.0);   // t[5]
    EXPECT_EQ(from_file[9], 14.45); // w
    EXPECT_TRUE(set.feasible(from_file));
}

// apply ändert genau die adressierten Elemente der Variante v.
TEST(VariableSet, ApplyChangesOnlyAddressedFields) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    const SystemBatchHost original = mko::make_system_batch(loaded, 4);
    const VariableSet set(loaded.system, original.topology());
    SystemBatchHost batch = original;
    const std::vector<double> x = sample_vector(set, 2);
    set.apply(batch, 2, x);

    SystemBatchHost expected = original;
    for (int k = 0; k < set.size(); ++k) {
        const auto& var = set[k];
        const double value = x[static_cast<std::size_t>(k)];
        if (var.type == VariableType::Curvature) {
            expected.curvature(2, var.batch_surface) = value;
        } else {
            expected.thickness(2, var.batch_surface) = value;
        }
    }
    EXPECT_TRUE(batch == expected);
    // Gegenprobe: Anzahl geänderter Elemente der Rohfelder
    int changed = 0;
    for (std::size_t i = 0; i < batch.curvature_data().size(); ++i) {
        changed += batch.curvature_data()[i] != original.curvature_data()[i];
        changed += batch.thickness_data()[i] != original.thickness_data()[i];
    }
    EXPECT_EQ(changed, set.size());
    EXPECT_TRUE(std::equal(batch.index_data().begin(), batch.index_data().end(),
                           original.index_data().begin()));
    EXPECT_TRUE(std::equal(batch.field_target_data().begin(), batch.field_target_data().end(),
                           original.field_target_data().begin()));
}

TEST(VariableSet, ApplyRejectsInvalidInput) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    SystemBatchHost batch = mko::make_system_batch(loaded, 2);
    const VariableSet set(loaded.system, batch.topology());
    const SystemBatchHost before = batch;
    std::vector<double> x = set.extract(batch, 0);
    EXPECT_THROW(set.apply(batch, 2, x), std::out_of_range);
    EXPECT_THROW(set.apply(batch, -1, x), std::out_of_range);
    x.push_back(1.0);
    EXPECT_THROW(set.apply(batch, 0, x), std::invalid_argument);
    x.pop_back();
    x[3] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(set.apply(batch, 0, x), std::invalid_argument);
    EXPECT_TRUE(batch == before); // nichts teilweise geschrieben

    // anderer Batch (andere Topologie)
    const mko::LoadedSystem lens = mko::load_system(kSystems + "einzellinse_bk7.json");
    SystemBatchHost other = mko::make_system_batch(lens, 1);
    EXPECT_THROW(set.apply(other, 0, set.extract(batch, 0)), std::invalid_argument);
    EXPECT_THROW((void)set.extract(other, 0), std::invalid_argument);
    EXPECT_THROW((void)set.extract(lens.system), std::invalid_argument);
}

TEST(VariableSet, BoundsAndProjection) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    const SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    const VariableSet set(loaded.system, batch.topology());
    std::vector<double> x = set.extract(batch, 0);
    x[0] = 0.5;   // über c_max = 0,2
    x[1] = -0.3;  // unter −0,2
    x[6] = 0.1;   // unter t_min = 0,5
    x[7] = 1.0e3; // offen nach oben
    x[9] = 50.0;  // über w_max = 40
    EXPECT_FALSE(set.feasible(x));
    set.clamp_to_bounds(x);
    EXPECT_TRUE(set.feasible(x));
    EXPECT_EQ(x[0], 0.2);
    EXPECT_EQ(x[1], -0.2);
    EXPECT_EQ(x[6], 0.5);
    EXPECT_EQ(x[7], 1.0e3);
    EXPECT_EQ(x[9], 40.0);
    const std::vector<double> y = x;
    set.clamp_to_bounds(x); // idempotent
    EXPECT_EQ(x, y);
    std::vector<double> short_x(3);
    EXPECT_THROW(set.clamp_to_bounds(short_x), std::invalid_argument);
    EXPECT_THROW((void)set.feasible(short_x), std::invalid_argument);
}

// apply_to_description schreibt dieselben Werte wie apply in den Batch; die Datei bleibt
// gültig und ergibt beim Neuaufbau dieselben Krümmungen und Dicken.
TEST(VariableSet, ApplyToDescriptionMatchesBatch) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    const VariableSet set(loaded.system, batch.topology());
    const std::vector<double> x = sample_vector(set, 3);
    set.apply(batch, 0, x);

    SystemDescription sys = loaded.system;
    set.apply_to_description(sys, x);
    EXPECT_EQ(set.extract(sys), x);
    const SystemDescription reread = mko::read_system_json(mko::write_system_json(sys));
    EXPECT_TRUE(reread == sys);
    const SystemBatchHost rebuilt = mko::make_system_batch(sys, loaded.index, 1);
    EXPECT_TRUE(std::equal(rebuilt.curvature_data().begin(), rebuilt.curvature_data().end(),
                           batch.curvature_data().begin()));
    EXPECT_TRUE(std::equal(rebuilt.thickness_data().begin(), rebuilt.thickness_data().end(),
                           batch.thickness_data().begin()));
    // nur Krümmungen und Dicken der Variablen geändert
    SystemDescription expected = loaded.system;
    for (int k = 0; k < set.size(); ++k) {
        auto& s = expected.surfaces[static_cast<std::size_t>(set[k].file_surface)];
        (set[k].type == VariableType::Curvature ? s.curvature : s.thickness) =
            x[static_cast<std::size_t>(k)];
    }
    EXPECT_TRUE(sys == expected);
}

TEST(VariableSet, NoVariables) {
    const mko::LoadedSystem lens = mko::load_system(kSystems + "einzellinse_bk7.json");
    SystemBatchHost batch = mko::make_system_batch(lens, 2);
    const VariableSet set(lens.system, batch.topology());
    EXPECT_TRUE(set.empty());
    EXPECT_TRUE(set.extract(batch, 1).empty());
    const SystemBatchHost before = batch;
    set.apply(batch, 1, std::vector<double>{});
    EXPECT_TRUE(batch == before);
}

TEST(VariableSet, ExampleFile) {
    const mko::LoadedSystem loaded = mko::load_system(kSystems + "einzellinse_bk7_variablen.json");
    const SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    const VariableSet set(loaded.system, batch.topology());
    ASSERT_EQ(set.size(), 4);
    EXPECT_EQ(set.extract(batch, 0), (std::vector<double>{0.00967, -0.00967, 6.0, 99.05}));
    EXPECT_EQ(set[3].name, "w");
    EXPECT_EQ(set[3].typical, 10.0);
}

TEST(VariableSet, VariableNames) {
    EXPECT_EQ(mko::variable_name(VariableType::Curvature, 3), "c[3]");
    EXPECT_EQ(mko::variable_name(VariableType::Thickness, 12), "t[12]");
    EXPECT_EQ(mko::variable_name(VariableType::WorkingDistance, 7), "w");
}

// ---- Abbildung Dateifläche → Tabellenfläche, Spalte mit Geisterplatten ------------------

namespace {

// Einzellinse mit zwei Geisterplatten im Spalt nach der Blende (d = 1,0 und 3,5, T = 5):
// Tabelle Stop, G, G, Sphäre, Sphäre; Dicken 1,0 / 2,5 / 1,5 / 6 / 98,23.
mko::LoadedSystem ghost_system_with_variables(std::optional<double> gap_min = std::nullopt) {
    mko::LoadedSystem loaded = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    VariableSpec gap = spec(VariableType::Thickness, SurfaceSelection::List, {0, 1});
    gap.min = gap_min;
    loaded.system.variables =
        mko::Variables{{spec(VariableType::Curvature, SurfaceSelection::All), gap,
                        spec(VariableType::WorkingDistance, SurfaceSelection::None)}};
    mko::validate_system(loaded.system);
    return loaded;
}

} // namespace

TEST(VariableSet, SurfaceMapIsIdentityWithoutGhostPlates) {
    const mko::LoadedSystem loaded = mehrlinser_with_variables();
    const BatchTopology topo = mko::topology_of(loaded.system, loaded.index);
    const mko::SurfaceMap map = mko::checked_surface_map(loaded.system, topo);
    ASSERT_EQ(map.file_surfaces(), static_cast<int>(loaded.system.surfaces.size()));
    for (int i = 0; i < map.file_surfaces(); ++i) {
        EXPECT_EQ(map.table_of_file[static_cast<std::size_t>(i)], i);
    }
}

// Geisterplatten verschieben die Tabellenindizes; die Dicke des geteilten Spalts wirkt auf das
// letzte Teilstück nach der letzten Platte.
TEST(VariableSet, GhostPlatesShiftBatchSurfaces) {
    const mko::LoadedSystem loaded = ghost_system_with_variables();
    const SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    ASSERT_EQ(batch.surfaces(), 5);
    const VariableSet set(loaded.system, batch.topology());
    ASSERT_EQ(set.size(), 5);
    EXPECT_EQ(set[0].name, "c[1]");
    EXPECT_EQ(set[0].batch_surface, 3);
    EXPECT_EQ(set[1].batch_surface, 4);
    EXPECT_EQ(set[2].name, "t[0]");
    EXPECT_EQ(set[2].batch_surface, 2); // zweite Geisterplatte, Teilstück bis zur Linse
    EXPECT_TRUE(set[2].split);
    EXPECT_EQ(set[2].offset, 3.5);
    EXPECT_EQ(set[2].lower, 3.5); // ohne min: die Platte muss im Spalt bleiben
    EXPECT_EQ(set[3].name, "t[1]");
    EXPECT_EQ(set[3].batch_surface, 3);
    EXPECT_FALSE(set[3].split);
    EXPECT_EQ(set[4].batch_surface, 4);
    // Startwerte: Dateidicke T = 5 = 3,5 + 1,5
    EXPECT_EQ(set.extract(batch, 0), set.extract(loaded.system));
    EXPECT_EQ(set.extract(batch, 0)[2], 5.0);

    // Eine Schranke über dem Plattenabstand bleibt, eine darunter wird angehoben.
    EXPECT_EQ(VariableSet(ghost_system_with_variables(4.0).system, batch.topology())[2].lower, 4.0);
    EXPECT_EQ(VariableSet(ghost_system_with_variables(2.0).system, batch.topology())[2].lower, 3.5);
}

// Roundtrip mit Geisterplatten; apply ändert nur das letzte Teilstück, der Batch stimmt
// bitweise mit dem überein, der aus der geänderten Datei neu gebaut wird.
TEST(VariableSet, GhostGapRoundtrip) {
    const mko::LoadedSystem loaded = ghost_system_with_variables();
    const SystemBatchHost original = mko::make_system_batch(loaded, 3);
    const VariableSet set(loaded.system, original.topology());
    SystemBatchHost batch = original;
    for (int v = 0; v < batch.variants(); ++v) {
        std::vector<double> x = set.extract(batch, v);
        x[0] = 0.0123 + 0.001 * v;
        x[1] = -0.0087 - 0.002 * v;
        x[2] = 4.123456789 + 0.71 * v; // im Bereich [3,5; 7]: x − 3,5 ist exakt
        x[3] = 5.5 - 0.3 * v;
        x[4] = 97.0 + 0.123 * v;
        set.apply(batch, v, x);
        EXPECT_TRUE(same_bits(set.extract(batch, v)[2], x[2]));
        EXPECT_EQ(set.extract(batch, v), x);
        EXPECT_EQ(batch.thickness(v, 0), 1.0); // Plattenabstände fest
        EXPECT_EQ(batch.thickness(v, 1), 2.5);
        EXPECT_EQ(batch.thickness(v, 2), x[2] - 3.5);

        SystemDescription sys = loaded.system;
        set.apply_to_description(sys, x);
        EXPECT_NO_THROW(mko::validate_system(sys));
        const SystemBatchHost rebuilt = mko::make_system_batch(sys, loaded.index, 1);
        for (int s = 0; s < batch.surfaces(); ++s) {
            EXPECT_TRUE(same_bits(rebuilt.thickness(0, s), batch.thickness(v, s))) << s;
            EXPECT_TRUE(same_bits(rebuilt.curvature(0, s), batch.curvature(v, s))) << s;
        }
    }
    // Geisterkoeffizienten und Brechzahlen unverändert
    EXPECT_TRUE(std::equal(batch.ghost_coeff_data().begin(), batch.ghost_coeff_data().end(),
                           original.ghost_coeff_data().begin()));
    EXPECT_TRUE(std::equal(batch.index_data().begin(), batch.index_data().end(),
                           original.index_data().begin()));
}

// Roundtrip bei geteiltem Spalt: x = fl(d + fl(x − d)) gilt für x ≥ d bis auf den Fall, dass
// x − d genau in der Mitte zweier Gleitkommazahlen derselben Binade wie x liegt (dann ±1 ulp).
// Beispiel: d = 0,11 (letztes Bit 2⁻⁵⁶) und x in [0,235; 0,25) (ulp 2⁻⁵⁵). Der Batch folgt
// trotzdem bitweise make_surface_table.
TEST(VariableSet, GhostGapRoundtripTieCase) {
    mko::LoadedSystem loaded = ghost_system_with_variables();
    loaded.system.surfaces[0].thickness = 0.245;
    loaded.system.ghost_plates[0].distance = 0.05;
    loaded.system.ghost_plates[1].distance = 0.11;
    mko::validate_system(loaded.system);
    const SystemBatchHost base = mko::make_system_batch(loaded.system, loaded.index, 1);
    const VariableSet set(loaded.system, base.topology());
    EXPECT_EQ(set[2].offset, 0.11);
    SystemBatchHost batch = base;
    int inexact = 0;
    for (int n = 0; n < 200; ++n) {
        std::vector<double> x = set.extract(base, 0);
        x[2] = 0.24 + 2.5e-5 * n + 1e-12 * std::sqrt(static_cast<double>(n));
        set.apply(batch, 0, x);
        const double y = set.extract(batch, 0)[2];
        EXPECT_LE(std::abs(y - x[2]), std::nextafter(x[2], 1.0) - x[2]) << n;
        inexact += !same_bits(y, x[2]);
        SystemDescription sys = loaded.system;
        set.apply_to_description(sys, x);
        const SystemBatchHost rebuilt = mko::make_system_batch(sys, loaded.index, 1);
        EXPECT_TRUE(same_bits(rebuilt.thickness(0, 2), batch.thickness(0, 2))) << n;
    }
    EXPECT_GT(inexact, 0); // der Fall kommt tatsächlich vor
    EXPECT_LT(inexact, 200);
}

TEST(VariableSet, SurfaceMapRejectsMismatch) {
    const mko::LoadedSystem ghosts = ghost_system_with_variables();
    const mko::LoadedSystem lens = mko::load_system(kSystems + "einzellinse_bk7.json");
    const BatchTopology three = mko::topology_of(lens.system, lens.index);
    EXPECT_THROW((void)mko::checked_surface_map(ghosts.system, three), std::invalid_argument);
    EXPECT_THROW(VariableSet(ghosts.system, three), std::invalid_argument);

    BatchTopology topo = mko::topology_of(ghosts.system, ghosts.index);
    BatchTopology wrong_type = topo;
    wrong_type.type[3] = SurfaceKind::Plane; // Dateifläche 1 ist eine Sphäre
    EXPECT_THROW((void)mko::checked_surface_map(ghosts.system, wrong_type), std::invalid_argument);
    BatchTopology swapped = topo;
    std::swap(swapped.type[1], swapped.type[3]); // Geisterplatte an falscher Stelle
    EXPECT_THROW((void)mko::checked_surface_map(ghosts.system, swapped), std::invalid_argument);
}
