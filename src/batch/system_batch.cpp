// SystemBatch auf dem Host: Topologie, Felder pro Variante, Aufbau aus einem System.

#include "mikroptim/batch/system_batch.hpp"

#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/paraxial/paraxial.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(int n) {
    return static_cast<std::size_t>(n);
}

[[noreturn]] void fail_topology(const std::string& what) {
    throw std::invalid_argument("SystemBatch: " + what);
}

// Bitweiser Vergleich (NaN-Muster und Vorzeichen der Null zählen mit).
bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// Kopiert `stride` Elemente der Variante `from` aus `src` an Variante `to` in `dst`.
void copy_slice(const std::vector<double>& src, int from, std::vector<double>& dst, int to,
                std::size_t stride) {
    if (stride == 0) {
        return;
    }
    const auto first = src.begin() + static_cast<std::ptrdiff_t>(to_size(from) * stride);
    std::copy(first, first + static_cast<std::ptrdiff_t>(stride),
              dst.begin() + static_cast<std::ptrdiff_t>(to_size(to) * stride));
}

} // namespace

// ---- Topologie ---------------------------------------------------------------------------

void validate_topology(const BatchTopology& t) {
    if (t.surfaces < 2 || t.surfaces > kMaxSurfaces) {
        fail_topology(std::to_string(t.surfaces) + " Flächen, erlaubt sind 2 bis " +
                      std::to_string(kMaxSurfaces));
    }
    if (t.type.size() != to_size(t.surfaces)) {
        fail_topology("Typliste hat " + std::to_string(t.type.size()) + " Einträge, erwartet " +
                      std::to_string(t.surfaces));
    }
    if (t.wavelengths < 1) {
        fail_topology("mindestens eine Wellenlänge nötig");
    }
    if (t.fields < 1) {
        fail_topology("mindestens ein Feld nötig");
    }
    if (t.ghost_plates < 0 || t.ghost_terms < 0) {
        fail_topology("negative Zahl von Geisterplatten oder Koeffizienten");
    }
    if (t.ghost_plates > 0 && t.ghost_terms < 1) {
        fail_topology("Geisterplatten brauchen mindestens einen Koeffizienten (K >= 1)");
    }
    if (t.ghost_plates == 0 && t.ghost_terms != 0) {
        fail_topology("ohne Geisterplatten muss K = 0 sein");
    }
    if (t.source != SourceKind::Collimated && t.source != SourceKind::Point) {
        fail_topology("unbekannte Quellenart");
    }
    if (t.source == SourceKind::Point && t.fields != 1) {
        fail_topology("Punktquelle verlangt genau ein Feld");
    }
    int ghosts = 0;
    for (std::size_t s = 0; s < t.type.size(); ++s) {
        const SurfaceKind k = t.type[s];
        if (static_cast<unsigned>(k) > static_cast<unsigned>(SurfaceKind::GhostPlate)) {
            fail_topology("Fläche " + std::to_string(s) + ": unbekannter Flächentyp");
        }
        if (k == SurfaceKind::GhostPlate) {
            ++ghosts;
        }
    }
    if (ghosts != t.ghost_plates) {
        fail_topology(std::to_string(ghosts) +
                      " Flächen vom Typ GhostPlate, aber G = " + std::to_string(t.ghost_plates));
    }
    if (!t.pupil_mode.empty() && t.pupil_mode.size() != to_size(t.fields)) {
        fail_topology("Pupillenarten für " + std::to_string(t.pupil_mode.size()) +
                      " Felder, erwartet " + std::to_string(t.fields) + " oder keine");
    }
    for (std::size_t f = 0; f < t.pupil_mode.size(); ++f) {
        const PupilMode m = t.pupil_mode[f];
        if (m != PupilMode::Half && m != PupilMode::Full && m != PupilMode::Axis) {
            fail_topology("Feld " + std::to_string(f) + ": unbekannte Pupillenart");
        }
    }
}

// ---- SystemBatchHost ---------------------------------------------------------------------

SystemBatchHost::SystemBatchHost(BatchTopology topology, int variants)
    : topology_(std::move(topology)) {
    validate_topology(topology_);
    for (int s = 0; s < topology_.surfaces; ++s) {
        if (topology_.type[to_size(s)] == SurfaceKind::GhostPlate) {
            ghost_surface_.push_back(s);
        }
    }
    ghost_norm_radius_.assign(to_size(topology_.ghost_plates), 0.0);
    wavelength_nm_.assign(to_size(topology_.wavelengths), 0.0);
    wavelength_weight_.assign(to_size(topology_.wavelengths), 0.0);
    field_weight_.assign(to_size(topology_.fields), 0.0);
    resize(variants);
}

BatchShape SystemBatchHost::shape() const {
    return BatchShape{variants_,        topology_.surfaces,     topology_.wavelengths,
                      topology_.fields, topology_.ghost_plates, topology_.ghost_terms};
}

void SystemBatchHost::resize(int variants) {
    if (variants < 0) {
        throw std::invalid_argument("SystemBatch: negative Variantenzahl");
    }
    variants_ = variants;
    const BatchShape sh = shape();
    const std::size_t v = to_size(variants);
    curvature_.resize(v * sh.surface_stride(), 0.0);
    thickness_.resize(v * sh.surface_stride(), 0.0);
    index_.resize(v * sh.index_stride(), 0.0);
    pupil_radius_.resize(v, 0.0);
    field_angle_.resize(v * sh.field_stride(), 0.0);
    field_target_.resize(v * sh.field_stride(), 0.0);
    ghost_coeff_.resize(v * sh.ghost_coeff_stride(), 0.0);
    ghost_index_.resize(v * sh.ghost_index_stride(), 0.0);
}

int SystemBatchHost::add_variant_copy(int from) {
    check(from, variants_, "Variante");
    const int to = variants_;
    resize(variants_ + 1);
    copy_variant(from, to);
    return to;
}

void SystemBatchHost::copy_variant(int from, int to) {
    copy_variant_from(*this, from, to);
}

void SystemBatchHost::copy_variant_from(const SystemBatchHost& other, int from, int to) {
    if (!(other.topology_ == topology_)) {
        throw std::invalid_argument("SystemBatch: Variante aus einem Batch anderer Topologie");
    }
    check(from, other.variants_, "Quellvariante");
    check(to, variants_, "Zielvariante");
    const BatchShape sh = shape();
    copy_slice(other.curvature_, from, curvature_, to, sh.surface_stride());
    copy_slice(other.thickness_, from, thickness_, to, sh.surface_stride());
    copy_slice(other.index_, from, index_, to, sh.index_stride());
    copy_slice(other.pupil_radius_, from, pupil_radius_, to, 1);
    copy_slice(other.field_angle_, from, field_angle_, to, sh.field_stride());
    copy_slice(other.field_target_, from, field_target_, to, sh.field_stride());
    copy_slice(other.ghost_coeff_, from, ghost_coeff_, to, sh.ghost_coeff_stride());
    copy_slice(other.ghost_index_, from, ghost_index_, to, sh.ghost_index_stride());
}

void SystemBatchHost::check(int i, int count, const char* what) {
    if (i < 0 || i >= count) {
        throw std::out_of_range(std::string("SystemBatch: ") + what + " " + std::to_string(i) +
                                " außerhalb von [0, " + std::to_string(count) + ")");
    }
}

// Gemeinsame Felder
SurfaceKind SystemBatchHost::type(int s) const {
    check(s, topology_.surfaces, "Fläche");
    return topology_.type[to_size(s)];
}
int SystemBatchHost::ghost_surface(int g) const {
    check(g, topology_.ghost_plates, "Geisterplatte");
    return ghost_surface_[to_size(g)];
}
double& SystemBatchHost::ghost_norm_radius(int g) {
    check(g, topology_.ghost_plates, "Geisterplatte");
    return ghost_norm_radius_[to_size(g)];
}
double SystemBatchHost::ghost_norm_radius(int g) const {
    check(g, topology_.ghost_plates, "Geisterplatte");
    return ghost_norm_radius_[to_size(g)];
}
double& SystemBatchHost::wavelength_nm(int l) {
    check(l, topology_.wavelengths, "Wellenlänge");
    return wavelength_nm_[to_size(l)];
}
double SystemBatchHost::wavelength_nm(int l) const {
    check(l, topology_.wavelengths, "Wellenlänge");
    return wavelength_nm_[to_size(l)];
}
double& SystemBatchHost::wavelength_weight(int l) {
    check(l, topology_.wavelengths, "Wellenlänge");
    return wavelength_weight_[to_size(l)];
}
double SystemBatchHost::wavelength_weight(int l) const {
    check(l, topology_.wavelengths, "Wellenlänge");
    return wavelength_weight_[to_size(l)];
}
double& SystemBatchHost::field_weight(int f) {
    check(f, topology_.fields, "Feld");
    return field_weight_[to_size(f)];
}
double SystemBatchHost::field_weight(int f) const {
    check(f, topology_.fields, "Feld");
    return field_weight_[to_size(f)];
}

// Felder pro Variante
double& SystemBatchHost::curvature(int v, int s) {
    check(v, variants_, "Variante");
    check(s, topology_.surfaces, "Fläche");
    return curvature_[shape().surface_at(v, s)];
}
double SystemBatchHost::curvature(int v, int s) const {
    check(v, variants_, "Variante");
    check(s, topology_.surfaces, "Fläche");
    return curvature_[shape().surface_at(v, s)];
}
double& SystemBatchHost::thickness(int v, int s) {
    check(v, variants_, "Variante");
    check(s, topology_.surfaces, "Fläche");
    return thickness_[shape().surface_at(v, s)];
}
double SystemBatchHost::thickness(int v, int s) const {
    check(v, variants_, "Variante");
    check(s, topology_.surfaces, "Fläche");
    return thickness_[shape().surface_at(v, s)];
}
double& SystemBatchHost::index(int v, int s, int l) {
    check(v, variants_, "Variante");
    check(s, topology_.surfaces, "Fläche");
    check(l, topology_.wavelengths, "Wellenlänge");
    return index_[shape().index_at(v, s, l)];
}
double SystemBatchHost::index(int v, int s, int l) const {
    check(v, variants_, "Variante");
    check(s, topology_.surfaces, "Fläche");
    check(l, topology_.wavelengths, "Wellenlänge");
    return index_[shape().index_at(v, s, l)];
}
double& SystemBatchHost::pupil_radius(int v) {
    check(v, variants_, "Variante");
    return pupil_radius_[to_size(v)];
}
double SystemBatchHost::pupil_radius(int v) const {
    check(v, variants_, "Variante");
    return pupil_radius_[to_size(v)];
}
double& SystemBatchHost::field_angle(int v, int f) {
    check(v, variants_, "Variante");
    check(f, topology_.fields, "Feld");
    return field_angle_[shape().field_at(v, f)];
}
double SystemBatchHost::field_angle(int v, int f) const {
    check(v, variants_, "Variante");
    check(f, topology_.fields, "Feld");
    return field_angle_[shape().field_at(v, f)];
}
double& SystemBatchHost::field_target(int v, int f) {
    check(v, variants_, "Variante");
    check(f, topology_.fields, "Feld");
    return field_target_[shape().field_at(v, f)];
}
double SystemBatchHost::field_target(int v, int f) const {
    check(v, variants_, "Variante");
    check(f, topology_.fields, "Feld");
    return field_target_[shape().field_at(v, f)];
}
double& SystemBatchHost::ghost_coeff(int v, int g, int k) {
    check(v, variants_, "Variante");
    check(g, topology_.ghost_plates, "Geisterplatte");
    check(k, topology_.ghost_terms, "Koeffizient");
    return ghost_coeff_[shape().ghost_coeff_at(v, g, k)];
}
double SystemBatchHost::ghost_coeff(int v, int g, int k) const {
    check(v, variants_, "Variante");
    check(g, topology_.ghost_plates, "Geisterplatte");
    check(k, topology_.ghost_terms, "Koeffizient");
    return ghost_coeff_[shape().ghost_coeff_at(v, g, k)];
}
double& SystemBatchHost::ghost_index(int v, int g, int l) {
    check(v, variants_, "Variante");
    check(g, topology_.ghost_plates, "Geisterplatte");
    check(l, topology_.wavelengths, "Wellenlänge");
    return ghost_index_[shape().ghost_index_at(v, g, l)];
}
double SystemBatchHost::ghost_index(int v, int g, int l) const {
    check(v, variants_, "Variante");
    check(g, topology_.ghost_plates, "Geisterplatte");
    check(l, topology_.wavelengths, "Wellenlänge");
    return ghost_index_[shape().ghost_index_at(v, g, l)];
}

SystemBatchView<double> SystemBatchHost::view() const {
    SystemBatchView<double> b{};
    b.shape = shape();
    b.source = topology_.source;
    b.source_point[0] = source_point[0];
    b.source_point[1] = source_point[1];
    b.source_point[2] = source_point[2];
    b.source_na = source_na;
    b.reference_wavelength_nm = reference_wavelength_nm;
    b.type = topology_.type.data();
    b.ghost_surface = ghost_surface_.data();
    b.ghost_norm_radius = ghost_norm_radius_.data();
    b.wavelength_nm = wavelength_nm_.data();
    b.wavelength_weight = wavelength_weight_.data();
    b.field_weight = field_weight_.data();
    b.curvature = curvature_.data();
    b.thickness = thickness_.data();
    b.index = index_.data();
    b.pupil_radius = pupil_radius_.data();
    b.field_angle = field_angle_.data();
    b.field_target = field_target_.data();
    b.ghost_coeff = ghost_coeff_.data();
    b.ghost_index = ghost_index_.data();
    return b;
}

SurfaceTableView<double> SystemBatchHost::surface_table(int v) const {
    check(v, variants_, "Variante");
    return view().surface_table(v);
}

void SystemBatchHost::validate_values() const {
    const auto fail = [](const std::string& what) {
        throw std::invalid_argument("SystemBatch: " + what);
    };
    const auto finite = [&](const std::vector<double>& values, const char* name) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i])) {
                fail(std::string(name) + "[" + std::to_string(i) + "] ist nicht endlich");
            }
        }
    };
    finite(ghost_norm_radius_, "ghost_norm_radius");
    finite(wavelength_nm_, "wavelength_nm");
    finite(wavelength_weight_, "wavelength_weight");
    finite(field_weight_, "field_weight");
    finite(curvature_, "curvature");
    finite(thickness_, "thickness");
    finite(index_, "index");
    finite(pupil_radius_, "pupil_radius");
    finite(field_angle_, "field_angle");
    finite(field_target_, "field_target");
    finite(ghost_coeff_, "ghost_coeff");
    finite(ghost_index_, "ghost_index");
    if (!(reference_wavelength_nm > 0.0) || !std::isfinite(reference_wavelength_nm)) {
        fail("reference_wavelength_nm muss endlich und > 0 sein");
    }
    for (int l = 0; l < topology_.wavelengths; ++l) {
        if (!(wavelength_nm_[to_size(l)] > 0.0)) {
            fail("wavelength_nm[" + std::to_string(l) + "] muss > 0 sein");
        }
        if (wavelength_weight_[to_size(l)] < 0.0) {
            fail("wavelength_weight[" + std::to_string(l) + "] ist negativ");
        }
    }
    for (int f = 0; f < topology_.fields; ++f) {
        if (field_weight_[to_size(f)] < 0.0) {
            fail("field_weight[" + std::to_string(f) + "] ist negativ");
        }
    }
    for (int g = 0; g < topology_.ghost_plates; ++g) {
        if (!(ghost_norm_radius_[to_size(g)] > 0.0)) {
            fail("ghost_norm_radius[" + std::to_string(g) + "] muss > 0 sein");
        }
    }
    for (std::size_t i = 0; i < index_.size(); ++i) {
        if (!(index_[i] > 0.0)) {
            fail("index[" + std::to_string(i) + "] muss > 0 sein");
        }
    }
    for (std::size_t i = 0; i < ghost_index_.size(); ++i) {
        if (!(ghost_index_[i] > 0.0)) {
            fail("ghost_index[" + std::to_string(i) + "] muss > 0 sein");
        }
    }
    const BatchShape sh = shape();
    for (int v = 0; v < variants_; ++v) {
        if (topology_.source == SourceKind::Collimated) {
            if (!(pupil_radius_[to_size(v)] > 0.0)) {
                fail("Variante " + std::to_string(v) + ": Pupillenradius muss > 0 sein");
            }
            continue;
        }
        if (!std::isfinite(source_na)) {
            fail("NA der Punktquelle ist nicht endlich");
        }
        for (int l = 0; l < topology_.wavelengths; ++l) {
            const double n = index_[sh.index_at(v, 0, l)];
            if (!(source_na > 0.0 && source_na < n)) {
                fail("Variante " + std::to_string(v) + ", Wellenlänge " + std::to_string(l) +
                     ": NA der Punktquelle muss in (0, n) liegen");
            }
        }
    }
    for (double x : source_point) {
        if (!std::isfinite(x)) {
            fail("Quellpunkt ist nicht endlich");
        }
    }
    check_pupil_modes(*this);
}

void check_pupil_modes(const SystemBatchHost& batch, int first, int count) {
    const int total = batch.variants();
    if (first < 0 || first > total || count > total - first) {
        throw std::out_of_range("check_pupil_modes: Variantenbereich außerhalb des Batches");
    }
    if (count < 0) {
        count = total - first;
    }
    const std::vector<PupilMode>& modes = batch.topology().pupil_mode;
    if (modes.empty()) {
        return;
    }
    const auto fail = [](const std::string& what) {
        throw std::invalid_argument("SystemBatch: " + what);
    };
    if (batch.source() == SourceKind::Point) {
        // Ein Feld; der Quellpunkt ist gemeinsam für alle Varianten.
        const PupilMode m = modes[0];
        const auto& q = batch.source_point;
        if ((m == PupilMode::Axis && !(q[0] == 0.0 && q[1] == 0.0)) ||
            (m == PupilMode::Half && !(q[0] == 0.0))) {
            fail("Quellpunkt passt nicht zur Pupillenart des Feldes 0 (Axis verlangt x = y = 0, "
                 "Half x = 0)");
        }
        return;
    }
    for (int v = first; v < first + count; ++v) {
        for (int f = 0; f < batch.fields(); ++f) {
            if (modes[to_size(f)] == PupilMode::Axis && batch.field_angle(v, f) != 0.0) {
                fail("Variante " + std::to_string(v) + ", Feld " + std::to_string(f) +
                     ": Achsfeld (Pupillenart Axis) verlangt Feldwinkel 0");
            }
        }
    }
}

bool operator==(const SystemBatchHost& a, const SystemBatchHost& b) {
    return a.topology_ == b.topology_ && a.variants_ == b.variants_ &&
           a.ghost_surface_ == b.ghost_surface_ &&
           same_bits(a.reference_wavelength_nm, b.reference_wavelength_nm) &&
           same_bits(a.source_point[0], b.source_point[0]) &&
           same_bits(a.source_point[1], b.source_point[1]) &&
           same_bits(a.source_point[2], b.source_point[2]) && same_bits(a.source_na, b.source_na) &&
           same_bits(a.ghost_norm_radius_, b.ghost_norm_radius_) &&
           same_bits(a.wavelength_nm_, b.wavelength_nm_) &&
           same_bits(a.wavelength_weight_, b.wavelength_weight_) &&
           same_bits(a.field_weight_, b.field_weight_) && same_bits(a.curvature_, b.curvature_) &&
           same_bits(a.thickness_, b.thickness_) && same_bits(a.index_, b.index_) &&
           same_bits(a.pupil_radius_, b.pupil_radius_) &&
           same_bits(a.field_angle_, b.field_angle_) &&
           same_bits(a.field_target_, b.field_target_) &&
           same_bits(a.ghost_coeff_, b.ghost_coeff_) && same_bits(a.ghost_index_, b.ghost_index_);
}

// ---- Aufbau aus einem System -------------------------------------------------------------

BatchTopology topology_of(const SystemDescription& system, const IndexTable& index) {
    const HostSurfaceTable table = make_surface_table(system, index);
    BatchTopology t;
    t.surfaces = table.surfaces;
    t.wavelengths = table.wavelengths;
    t.fields = source_field_count(system);
    t.source = system.source.type == SourceType::Point ? SourceKind::Point : SourceKind::Collimated;
    t.type = table.type;
    // Pupillenart pro Feld, gleiche Regel wie wavefront_pupil_mode (merit/wavefront.cpp)
    for (int f = 0; f < t.fields; ++f) {
        if (t.source == SourceKind::Collimated) {
            t.pupil_mode.push_back(system.source.field_angles_deg[to_size(f)] == 0.0
                                       ? PupilMode::Axis
                                       : PupilMode::Half);
        } else {
            const auto& q = system.source.point;
            t.pupil_mode.push_back(q[0] == 0.0 ? (q[1] == 0.0 ? PupilMode::Axis : PupilMode::Half)
                                               : PupilMode::Full);
        }
    }
    return t;
}

namespace {

// Gibt es einen Feldwinkel ≠ 0 (nur dann braucht das Feldziel eine Brennweite)?
bool has_off_axis_field(const SystemDescription& system) {
    return system.source.type == SourceType::Collimated &&
           std::any_of(system.source.field_angles_deg.begin(), system.source.field_angles_deg.end(),
                       [](double a) { return a != 0.0; });
}

// Brennweite aus Paraxialdaten; leer bei afokalen Systemen.
std::optional<double> focal_length_of(const ParaxialData& p) {
    if (p.afocal || !std::isfinite(p.focal_length)) {
        return std::nullopt;
    }
    return p.focal_length;
}

// Prüft Variantenzahl und Brechzahltabelle (vor der Paraxialrechnung für die Feldziele) und
// liefert die Flächentabelle.
HostSurfaceTable checked_table(const SystemDescription& system, const IndexTable& index,
                               int variants) {
    if (variants < 1) {
        throw std::invalid_argument("SystemBatch: mindestens eine Variante nötig");
    }
    HostSurfaceTable table = make_surface_table(system, index);
    if (system.spectrum.wavelengths_nm.size() != to_size(table.wavelengths) ||
        system.spectrum.weights.size() != to_size(table.wavelengths)) {
        throw std::invalid_argument(
            "SystemBatch: Brechzahltabelle passt nicht zum Spektrum des Systems");
    }
    return table;
}

// Baut den Batch; `focal_length` ist bereits aufgelöst (leer: Feldziel unbestimmt → 0).
SystemBatchHost build_batch(const SystemDescription& system, const IndexTable& index, int variants,
                            std::optional<double> focal_length) {
    const HostSurfaceTable table = checked_table(system, index, variants);
    BatchTopology topology = topology_of(system, index);
    const bool collimated = topology.source == SourceKind::Collimated;
    if (collimated && system.source.field_weights.size() != system.source.field_angles_deg.size()) {
        throw std::invalid_argument("SystemBatch: Feldgewichte passen nicht zu den Feldwinkeln");
    }

    SystemBatchHost b(std::move(topology), 1);
    const int S = b.surfaces();
    const int L = b.wavelengths();
    const int F = b.fields();
    for (int l = 0; l < L; ++l) {
        b.wavelength_nm(l) = system.spectrum.wavelengths_nm[to_size(l)];
        b.wavelength_weight(l) = system.spectrum.weights[to_size(l)];
    }
    b.reference_wavelength_nm = system.spectrum.reference_nm;
    if (!collimated) {
        b.source_point = system.source.point;
        b.source_na = system.source.numerical_aperture;
    }
    for (int s = 0; s < S; ++s) {
        b.curvature(0, s) = table.curvature[to_size(s)];
        b.thickness(0, s) = table.thickness[to_size(s)];
        for (int l = 0; l < L; ++l) {
            b.index(0, s, l) = table.index[to_size(s) * to_size(L) + to_size(l)];
        }
    }
    b.pupil_radius(0) = table.semi_diameter[0];
    for (int f = 0; f < F; ++f) {
        if (!collimated) {
            b.field_weight(f) = 1.0; // Punktquelle: ein Feld auf der Achse, h = 0
            continue;
        }
        b.field_weight(f) = system.source.field_weights[to_size(f)];
        // Gleicher Ausdruck wie source_ray (cpu_trace.cpp), damit die Winkel bitgleich sind.
        const double theta =
            system.source.field_angles_deg[to_size(f)] * (std::numbers::pi / 180.0);
        b.field_angle(0, f) = theta;
        if (theta != 0.0 && focal_length) {
            b.field_target(0, f) = *focal_length * std::tan(theta);
        }
    }
    b.resize(variants);
    for (int v = 1; v < variants; ++v) {
        b.copy_variant(0, v);
    }
    return b;
}

} // namespace

SystemBatchHost make_system_batch(const SystemDescription& system, const IndexTable& index,
                                  int variants, std::optional<double> focal_length) {
    checked_table(system, index, variants);
    if (focal_length && !std::isfinite(*focal_length)) {
        throw std::invalid_argument("SystemBatch: Brennweite für die Feldziele nicht endlich");
    }
    if (!focal_length && system.target && system.target->focal_length) {
        focal_length = system.target->focal_length;
    }
    if (!focal_length && has_off_axis_field(system)) {
        const auto& wl = system.spectrum.wavelengths_nm;
        const auto ref = std::find(wl.begin(), wl.end(), system.spectrum.reference_nm);
        if (ref != wl.end()) {
            focal_length = focal_length_of(
                compute_paraxial(system, index, static_cast<int>(ref - wl.begin())));
        }
    }
    return build_batch(system, index, variants, focal_length);
}

SystemBatchHost make_system_batch(const LoadedSystem& loaded, int variants) {
    const SystemDescription& system = loaded.system;
    checked_table(system, loaded.index, variants);
    std::optional<double> focal_length;
    if (system.target && system.target->focal_length) {
        focal_length = system.target->focal_length;
    } else if (has_off_axis_field(system)) {
        focal_length = focal_length_of(compute_paraxial(system, loaded.catalog));
    }
    return build_batch(system, loaded.index, variants, focal_length);
}

} // namespace mko
