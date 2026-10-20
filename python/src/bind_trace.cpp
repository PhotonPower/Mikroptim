// Python-Bindings: Wellenfront (trace), SystemBatch, Residuen und Merit auf CPU und GPU
// (TraceEngine) (M9-T1).

#include "bindings.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mko::py {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const char* pupil_mode_name(PupilMode mode) {
    switch (mode) {
    case PupilMode::Half:
        return "half";
    case PupilMode::Full:
        return "full";
    case PupilMode::Axis:
        return "axis";
    }
    return "unknown";
}

// Stützstellen und OPD eines Tupels (Feld f, Wellenlänge l).
struct TupleSamples {
    std::vector<double> rho;    // ρ der Stützstelle (numerik.md, Abschnitt 8)
    std::vector<double> phi;    // φ [rad]
    std::vector<double> weight; // W_i V_j
    std::vector<double> opd;    // OPD nach Kolbenabzug [mm], NaN bei Versagen
    std::vector<std::uint8_t> status;
};

// Ergebnis von trace(): Wellenfront pro Feld und Wellenlänge.
struct PyWavefront {
    bool gpu = false;
    int device = -1;
    std::string device_name;
    bool df64 = false;
    int rings = 0;
    int spokes = 0;
    int fields = 0;
    int wavelengths = 0;
    double reference_nm = 0.0;
    double target_focal_length = kNaN;
    std::vector<double> wavelengths_nm; // [L]
    std::vector<double> rms_waves;      // [F·L]
    std::vector<double> rms_mm;         // [F·L]
    std::vector<double> piston_mm;      // [F·L], NaN auf der GPU
    std::vector<std::int64_t> rays;     // [F·L]
    std::vector<std::int64_t> failed;   // [F·L]
    std::vector<double> image_points;   // [F·3]
    std::vector<std::string> pupil_modes;
    std::vector<TupleSamples> samples; // [F·L]

    const TupleSamples& tuple(int f, int l) const {
        if (f < 0 || f >= fields || l < 0 || l >= wavelengths) {
            throw std::out_of_range("Feld " + std::to_string(f) + " oder Wellenlänge " +
                                    std::to_string(l) + " existiert nicht");
        }
        return samples[sz(f) * sz(wavelengths) + sz(l)];
    }
};

void init_shape(PyWavefront& w, const SystemDescription& sys, const WavefrontOptions& o) {
    w.rings = o.rings;
    w.spokes = o.spokes;
    w.reference_nm = sys.spectrum.reference_nm;
    w.wavelengths_nm = sys.spectrum.wavelengths_nm;
}

PyWavefront wavefront_cpu(const LoadedSystem& loaded, const WavefrontOptions& options) {
    const WavefrontResult r = evaluate_wavefront(loaded, options);
    PyWavefront w;
    init_shape(w, loaded.system, options);
    w.fields = r.fields;
    w.wavelengths = r.wavelengths;
    w.target_focal_length = r.focal_length;
    w.pupil_modes.resize(sz(r.fields));
    w.image_points.resize(sz(r.fields) * 3);
    for (const WavefrontMap& m : r.maps) {
        w.rms_waves.push_back(m.rms_waves);
        w.rms_mm.push_back(m.rms);
        w.piston_mm.push_back(m.piston);
        w.rays.push_back(m.samples());
        w.failed.push_back(m.failed);
        w.pupil_modes[sz(m.field)] = pupil_mode_name(m.quadrature.mode);
        w.image_points[sz(m.field) * 3 + 0] = m.image_point.x;
        w.image_points[sz(m.field) * 3 + 1] = m.image_point.y;
        w.image_points[sz(m.field) * 3 + 2] = m.image_point.z;
        TupleSamples t;
        for (std::size_t k = 0; k < m.pupil.size(); ++k) {
            t.rho.push_back(m.pupil[k].rho);
            t.phi.push_back(m.pupil[k].phi);
            t.weight.push_back(m.pupil[k].weight);
            t.opd.push_back(m.opd[k]);
            t.status.push_back(static_cast<std::uint8_t>(m.status[k]));
        }
        w.samples.push_back(std::move(t));
    }
    return w;
}

#ifdef MIKROPTIM_HAS_CUDA

// Wie `mikroptim trace --gpu` (src/cli/trace_command.cpp, evaluate_wavefront_gpu): eine
// Variante mit den idealen Bildpunkten, Kernel RESIDUALS mit hartem Versagen, OPD und Status,
// RMS über tuple_wavefronts. Der Kolben wird nicht ausgegeben (NaN).
PyWavefront wavefront_gpu(const LoadedSystem& loaded, const WavefrontOptions& options,
                          cuda::TraceEngine& engine, bool df64) {
    const SystemDescription& sys = loaded.system;
    std::optional<double> focal_length;
    if (sys.source.type == SourceType::Collimated) {
        focal_length = target_focal_length(sys, loaded.catalog);
    }
    const SystemBatchHost batch = make_system_batch(sys, loaded.index, 1, focal_length);
    cuda::TraceSettings settings;
    settings.quadrature.rings = options.rings;
    settings.quadrature.spokes = options.spokes;
    settings.quadrature.failure = FailureMode::Hard;
    settings.opd = true;
    settings.status = true;
    settings.precision = df64 ? cuda::Precision::Df64 : cuda::Precision::Fp64;
    const cuda::TraceOutput out = engine.run(batch, cuda::TraceMode::Residuals, settings);
    const std::vector<TupleWavefront> tuples = tuple_wavefronts(out.residuals, out.layout, 0);

    PyWavefront w;
    init_shape(w, sys, options);
    w.gpu = true;
    w.device = engine.device();
    w.device_name = engine.device_name();
    w.df64 = df64;
    w.fields = batch.fields();
    w.wavelengths = batch.wavelengths();
    w.target_focal_length = focal_length ? *focal_length : kNaN;
    const double lambda_ref_mm = sys.spectrum.reference_nm * 1e-6;
    const double image_distance = batch.image_distance(0);
    for (int f = 0; f < w.fields; ++f) {
        w.pupil_modes.emplace_back(pupil_mode_name(out.layout.pupil_mode(f)));
        w.image_points.push_back(0.0);
        w.image_points.push_back(batch.field_target(0, f));
        w.image_points.push_back(image_distance);
    }
    for (const TupleWavefront& tw : tuples) {
        w.rms_mm.push_back(tw.rms);
        w.rms_waves.push_back(tw.valid() ? tw.rms / lambda_ref_mm : kNaN);
        w.piston_mm.push_back(kNaN);
        w.rays.push_back(tw.samples);
        w.failed.push_back(tw.failed);
        const PupilQuadrature<double>& q = out.layout.quadrature(tw.field);
        const std::int64_t offset = out.layout.offset(tw.field, tw.wavelength);
        TupleSamples t;
        for (int k = 0; k < tw.samples; ++k) {
            const PupilSample<double> s = sample(q, k);
            const std::size_t i = out.residuals.ray_index(0, offset + k);
            const auto status = out.residuals.status[i];
            t.rho.push_back(s.rho);
            t.phi.push_back(s.phi);
            t.weight.push_back(s.weight);
            // Der Kernel schreibt bei hartem Versagen OPD 0; wie die CPU NaN
            t.opd.push_back(
                status == static_cast<std::uint8_t>(TraceStatus::Ok) ? out.residuals.opd[i] : kNaN);
            t.status.push_back(status);
        }
        w.samples.push_back(std::move(t));
    }
    return w;
}

#endif

PyWavefront trace_system(const PySystem& s, int rings, int spokes, bool gpu, int device,
                         const std::string& precision) {
    const bool df64 = parse_df64(precision);
    if (rings < 1 || rings > kMaxPupilRings) {
        throw std::invalid_argument("rings muss in [1, " + std::to_string(kMaxPupilRings) +
                                    "] liegen");
    }
    if (spokes < 1) {
        throw std::invalid_argument("spokes muss >= 1 sein");
    }
    if (df64 && !gpu) {
        throw std::invalid_argument("precision=\"df64\" verlangt gpu=True (die CPU rechnet "
                                    "immer in FP64)");
    }
    WavefrontOptions options;
    options.rings = rings;
    options.spokes = spokes;
    if (gpu) {
        require_gpu(device);
#ifdef MIKROPTIM_HAS_CUDA
        nb::gil_scoped_release release;
        cuda::TraceEngine engine(device);
        return wavefront_gpu(s.loaded, options, engine, df64);
#endif
    }
    nb::gil_scoped_release release;
    return wavefront_cpu(s.loaded, options);
}

// ---- SystemBatch ----------------------------------------------------------------------------

// SystemBatch mit der Systembeschreibung, aus der er gebaut wurde (Variablenabbildung).
struct PyBatch {
    SystemBatchHost batch;
    SystemDescription system;
};

ResidualSettings residual_settings(int rings, int spokes, const std::string& failure,
                                   std::optional<double> fail_weight) {
    ResidualSettings r;
    r.rings = rings;
    r.spokes = spokes;
    if (failure == "soft") {
        r.failure = FailureMode::Soft;
    } else if (failure == "hard") {
        r.failure = FailureMode::Hard;
    } else {
        throw std::invalid_argument("failure erwartet \"soft\" oder \"hard\", nicht \"" + failure +
                                    "\"");
    }
    if (fail_weight) {
        r.fail_weight = *fail_weight;
    }
    return r;
}

// Residuen (CPU oder Download vom Gerät) als dict von NumPy-Arrays (Kopien).
nb::dict residuals_dict(ResidualResult r, const ResidualLayout& layout) {
    nb::dict d;
    const std::size_t v = sz(r.variants);
    d["variants"] = r.variants;
    d["rays_per_variant"] = r.rays_per_variant;
    d["penalty_rows"] = r.penalty_rows;
    d["stride"] = r.stride;
    std::vector<std::int64_t> offsets;
    for (int f = 0; f < layout.fields(); ++f) {
        for (int l = 0; l < layout.wavelengths(); ++l) {
            offsets.push_back(layout.offset(f, l));
        }
    }
    d["offsets"] = to_numpy(std::move(offsets), {sz(layout.fields()), sz(layout.wavelengths())});
    if (!r.opd.empty() && !r.status.empty()) {
        // RMS-Wellenfront pro Variante, Feld und Wellenlänge (tuple_wavefronts) [mm]
        std::vector<double> rms;
        std::vector<std::int64_t> failed;
        for (int i = 0; i < r.variants; ++i) {
            for (const TupleWavefront& t : tuple_wavefronts(r, layout, i)) {
                rms.push_back(t.rms);
                failed.push_back(t.failed);
            }
        }
        const std::vector<std::size_t> shape{v, sz(layout.fields()), sz(layout.wavelengths())};
        d["rms_mm"] = to_numpy(std::move(rms), shape);
        d["failed"] = to_numpy(std::move(failed), shape);
    }
    d["residuals"] = to_numpy(std::move(r.residuals), {v, sz(r.stride)});
    if (r.opd.empty()) {
        d["opd"] = nb::none();
    } else {
        d["opd"] = to_numpy(std::move(r.opd), {v, sz(r.rays_per_variant)});
    }
    if (r.status.empty()) {
        d["status"] = nb::none();
    } else {
        d["status"] = to_numpy(std::move(r.status), {v, sz(r.rays_per_variant)});
    }
    return d;
}

nb::dict merit_dict(MeritResult r) {
    nb::dict d;
    d["variants"] = r.variants;
    d["rays_per_variant"] = r.rays_per_variant;
    d["merit"] = to_numpy(std::move(r.merit));
    d["failed"] = to_numpy(std::move(r.failed));
    d["hard_failed"] = to_numpy(std::move(r.hard_failed));
    return d;
}

#ifdef MIKROPTIM_HAS_CUDA
using EnginePtr = std::unique_ptr<cuda::TraceEngine>;
#else
struct NoEngine {
    int device() const { return -1; }
    std::string device_name() const { return {}; }
};
using EnginePtr = std::unique_ptr<NoEngine>;
#endif

// TraceEngine (GPU): nur in CUDA-Builds benutzbar.
struct PyEngine {
    EnginePtr engine;
};

} // namespace

void bind_trace(nb::module_& m) {
    nb::class_<PyWavefront>(
        m, "Wavefront",
        "RMS-Wellenfront pro Feld f und Wellenlänge l (Ergebnis von trace()). Die Arrays sind "
        "schreibgeschützte Sichten auf das Ergebnisobjekt (kein Kopieren).")
        .def_prop_ro("backend", [](const PyWavefront& w) { return w.gpu ? "gpu" : "cpu"; })
        .def_prop_ro("device",
                     [](const PyWavefront& w) -> std::optional<int> {
                         if (!w.gpu) {
                             return std::nullopt;
                         }
                         return w.device;
                     })
        .def_prop_ro("device_name", [](const PyWavefront& w) { return w.device_name; })
        .def_prop_ro("precision", [](const PyWavefront& w) { return w.df64 ? "df64" : "fp64"; })
        .def_ro("rings", &PyWavefront::rings)
        .def_ro("spokes", &PyWavefront::spokes)
        .def_ro("fields", &PyWavefront::fields)
        .def_ro("wavelengths", &PyWavefront::wavelengths)
        .def_ro("reference_nm", &PyWavefront::reference_nm)
        .def_ro("target_focal_length", &PyWavefront::target_focal_length,
                "Sollbrennweite der idealen Bildpunkte [mm]; NaN bei der Punktquelle.")
        .def_ro("pupil_modes", &PyWavefront::pupil_modes)
        .def_prop_ro(
            "wavelengths_nm",
            [](const PyWavefront& w) {
                return readonly_view(w.wavelengths_nm.data(), {w.wavelengths_nm.size()}, {});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "rms_waves",
            [](const PyWavefront& w) {
                return readonly_view(w.rms_waves.data(), {sz(w.fields), sz(w.wavelengths)}, {});
            },
            nb::rv_policy::reference_internal, "RMS in Wellen bei λ_ref, Form (F, L).")
        .def_prop_ro(
            "rms_mm",
            [](const PyWavefront& w) {
                return readonly_view(w.rms_mm.data(), {sz(w.fields), sz(w.wavelengths)}, {});
            },
            nb::rv_policy::reference_internal, "RMS [mm], Form (F, L).")
        .def_prop_ro(
            "piston_mm",
            [](const PyWavefront& w) {
                return readonly_view(w.piston_mm.data(), {sz(w.fields), sz(w.wavelengths)}, {});
            },
            nb::rv_policy::reference_internal,
            "Kolben Σ W·OPL / Σ W [mm], Form (F, L); NaN auf der GPU.")
        .def_prop_ro(
            "rays",
            [](const PyWavefront& w) {
                return readonly_view(w.rays.data(), {sz(w.fields), sz(w.wavelengths)}, {});
            },
            nb::rv_policy::reference_internal, "Pupillenstützstellen pro Tupel, Form (F, L).")
        .def_prop_ro(
            "failed",
            [](const PyWavefront& w) {
                return readonly_view(w.failed.data(), {sz(w.fields), sz(w.wavelengths)}, {});
            },
            nb::rv_policy::reference_internal, "Versagte Strahlen pro Tupel, Form (F, L).")
        .def_prop_ro(
            "image_points",
            [](const PyWavefront& w) {
                return readonly_view(w.image_points.data(), {sz(w.fields), 3}, {});
            },
            nb::rv_policy::reference_internal,
            "Mittelpunkte M_f der Referenzsphäre (lokale Koordinaten der letzten Fläche) [mm], "
            "Form (F, 3).")
        .def(
            "samples",
            [](const PyWavefront& w, int field, int wavelength) {
                const TupleSamples& t = w.tuple(field, wavelength);
                nb::dict d;
                d["rho"] = to_numpy(t.rho);
                d["phi"] = to_numpy(t.phi);
                d["weight"] = to_numpy(t.weight);
                d["opd_mm"] = to_numpy(t.opd);
                std::vector<double> waves;
                const double lambda_ref_mm = w.reference_nm * 1e-6;
                for (const double v : t.opd) {
                    waves.push_back(v / lambda_ref_mm);
                }
                d["opd_waves"] = to_numpy(std::move(waves));
                d["status"] = to_numpy(t.status);
                return d;
            },
            nb::arg("field"), nb::arg("wavelength"),
            "Stützstellen der Pupillenquadratur eines Tupels als dict von NumPy-Kopien: rho, phi "
            "(numerik.md, Abschnitt 8), weight, opd_mm und opd_waves (NaN bei Versagen), "
            "status (TraceStatus, 0 = Ok).");

    m.def("trace", &trace_system, nb::arg("system"), nb::arg("rings") = 10, nb::arg("spokes") = 8,
          nb::arg("gpu") = false, nb::arg("device") = 0, nb::arg("precision") = "fp64",
          "RMS-Wellenfront pro Feld und Wellenlänge mit den idealen Bildpunkten wie "
          "`mikroptim trace` (CPU: evaluate_wavefront; gpu=True: Kernel RESIDUALS über die "
          "TraceEngine, precision \"fp64\" oder \"df64\"). Afokal ohne target.focal_length: "
          "ValueError.");

    nb::class_<PyBatch>(
        m, "SystemBatch",
        "V Varianten eines Systems gleicher Topologie (SoA, FP64). Die Array-Eigenschaften sind "
        "beschreibbare Sichten auf den Speicher des Batches (kein Kopieren): Änderungen wirken "
        "direkt auf die nächste Rechnung. Die Sichten bleiben gültig, solange sie existieren "
        "(sie halten den Batch am Leben; seine Größe ist fest). Flächen sind Tabellenflächen "
        "(mit eingefügten Geisterplatten).")
        .def(
            "__init__",
            [](PyBatch* self, const PySystem& s, int variants) {
                new (self) PyBatch{make_system_batch(s.loaded, variants), s.loaded.system};
            },
            nb::arg("system"), nb::arg("variants") = 1,
            "Batch mit `variants` gleichen Varianten (make_system_batch; Feldziele aus "
            "target.focal_length bzw. der paraxialen Brennweite).")
        .def(
            "copy", [](const PyBatch& b) { return PyBatch{b.batch, b.system}; }, "Tiefe Kopie.")
        .def(
            "select",
            [](const PyBatch& b, int first, int count) {
                PyBatch out{SystemBatchHost(b.batch.topology(), 0), b.system};
                out.batch.assign_variants(b.batch, first, count);
                return out;
            },
            nb::arg("first"), nb::arg("count") = 1,
            "Neuer Batch aus den Varianten [first, first + count) (Kopie).")
        .def_prop_ro("variants", [](const PyBatch& b) { return b.batch.variants(); })
        .def_prop_ro("surfaces", [](const PyBatch& b) { return b.batch.surfaces(); })
        .def_prop_ro("wavelengths", [](const PyBatch& b) { return b.batch.wavelengths(); })
        .def_prop_ro("fields", [](const PyBatch& b) { return b.batch.fields(); })
        .def_prop_ro("ghost_plates", [](const PyBatch& b) { return b.batch.ghost_plates(); })
        .def_prop_ro("ghost_terms", [](const PyBatch& b) { return b.batch.ghost_terms(); })
        .def_prop_rw(
            "reference_nm", [](const PyBatch& b) { return b.batch.reference_wavelength_nm; },
            [](PyBatch& b, double v) { b.batch.reference_wavelength_nm = v; })
        .def_prop_ro(
            "curvatures",
            [](PyBatch& b) {
                return writable_view(b.batch.curvature_data().data(),
                                     {sz(b.batch.variants()), sz(b.batch.surfaces())}, {});
            },
            nb::rv_policy::reference_internal, "Krümmung [1/mm], Form (V, S), Sicht.")
        .def_prop_ro(
            "thicknesses",
            [](PyBatch& b) {
                return writable_view(b.batch.thickness_data().data(),
                                     {sz(b.batch.variants()), sz(b.batch.surfaces())}, {});
            },
            nb::rv_policy::reference_internal,
            "Dicke nach Fläche s [mm], Form (V, S); [:, S-1] ist der Arbeitsabstand. Sicht.")
        .def_prop_ro(
            "indices",
            [](PyBatch& b) {
                return writable_view(
                    b.batch.index_data().data(),
                    {sz(b.batch.variants()), sz(b.batch.surfaces()), sz(b.batch.wavelengths())},
                    {});
            },
            nb::rv_policy::reference_internal, "Brechzahl nach Fläche s, Form (V, S, L), Sicht.")
        .def_prop_ro(
            "paraxial_indices",
            [](PyBatch& b) {
                return writable_view(b.batch.paraxial_index_data().data(),
                                     {sz(b.batch.variants()), sz(b.batch.surfaces())}, {});
            },
            nb::rv_policy::reference_internal,
            "Brechzahl bei λ_ref (Paraxialrechnung), Form (V, S), Sicht.")
        .def_prop_ro(
            "pupil_radius",
            [](PyBatch& b) {
                return writable_view(b.batch.pupil_radius_data().data(), {sz(b.batch.variants())},
                                     {});
            },
            nb::rv_policy::reference_internal, "Pupillenradius [mm], Form (V,), Sicht.")
        .def_prop_ro(
            "field_angles",
            [](PyBatch& b) {
                return writable_view(b.batch.field_angle_data().data(),
                                     {sz(b.batch.variants()), sz(b.batch.fields())}, {});
            },
            nb::rv_policy::reference_internal, "Feldwinkel [rad], Form (V, F), Sicht.")
        .def_prop_ro(
            "field_targets",
            [](PyBatch& b) {
                return writable_view(b.batch.field_target_data().data(),
                                     {sz(b.batch.variants()), sz(b.batch.fields())}, {});
            },
            nb::rv_policy::reference_internal, "Ideale Bildhöhe h_f [mm], Form (V, F), Sicht.")
        .def_prop_ro(
            "ghost_coeffs",
            [](PyBatch& b) {
                return writable_view(
                    b.batch.ghost_coeff_data().data(),
                    {sz(b.batch.variants()), sz(b.batch.ghost_plates()), sz(b.batch.ghost_terms())},
                    {});
            },
            nb::rv_policy::reference_internal,
            "Koeffizienten der Geisterplatten [mm], Form (V, G, K), Sicht.")
        .def_prop_ro(
            "wavelengths_nm",
            [](PyBatch& b) {
                return writable_view(b.batch.wavelength_nm_data().data(),
                                     {sz(b.batch.wavelengths())}, {});
            },
            nb::rv_policy::reference_internal, "Wellenlängen [nm], Form (L,), Sicht.")
        .def_prop_ro(
            "wavelength_weights",
            [](PyBatch& b) {
                return writable_view(b.batch.wavelength_weight_data().data(),
                                     {sz(b.batch.wavelengths())}, {});
            },
            nb::rv_policy::reference_internal, "Spektralgewichte, Form (L,), Sicht.")
        .def_prop_ro(
            "field_weights",
            [](PyBatch& b) {
                return writable_view(b.batch.field_weight_data().data(), {sz(b.batch.fields())},
                                     {});
            },
            nb::rv_policy::reference_internal, "Feldgewichte, Form (F,), Sicht.")
        .def(
            "validate", [](const PyBatch& b) { b.batch.validate_values(); },
            "Prüft die Werte (validate_values); ValueError bei der ersten Verletzung.")
        .def(
            "variable_names",
            [](const PyBatch& b) {
                const VariableSet set(b.system, b.batch.topology());
                std::vector<std::string> names;
                for (const Variable& v : set.variables()) {
                    names.push_back(v.name);
                }
                return names;
            },
            "Namen der Variablen (Abschnitt variables), Reihenfolge der Spalten von "
            "get_variables/set_variables.")
        .def(
            "get_variables",
            [](const PyBatch& b) {
                const VariableSet set(b.system, b.batch.topology());
                std::vector<double> x;
                for (int v = 0; v < b.batch.variants(); ++v) {
                    const std::vector<double> xv = set.extract(b.batch, v);
                    x.insert(x.end(), xv.begin(), xv.end());
                }
                return to_numpy(std::move(x), {sz(b.batch.variants()), sz(set.size())});
            },
            "Variablenvektoren aller Varianten (VariableSet::extract), Form (V, P), Kopie.")
        .def(
            "set_variables",
            [](PyBatch& b,
               nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu> x) {
                const VariableSet set(b.system, b.batch.topology());
                if (x.shape(0) != sz(b.batch.variants()) || x.shape(1) != sz(set.size())) {
                    throw std::invalid_argument("set_variables: Form (" +
                                                std::to_string(b.batch.variants()) + ", " +
                                                std::to_string(set.size()) + ") erwartet");
                }
                for (int v = 0; v < b.batch.variants(); ++v) {
                    set.apply(
                        b.batch, v,
                        std::span<const double>(x.data() + sz(v) * sz(set.size()), sz(set.size())));
                }
            },
            nb::arg("x"),
            "Setzt die Variablen aller Varianten (VariableSet::apply), x mit Form (V, P).")
        .def("__repr__", [](const PyBatch& b) {
            return "<mikroptim.SystemBatch " + std::to_string(b.batch.variants()) + " Varianten, " +
                   std::to_string(b.batch.surfaces()) + " Flächen>";
        });

    m.def(
        "residuals_cpu",
        [](const PyBatch& b, int rings, int spokes, const std::string& failure,
           std::optional<double> fail_weight) {
            const ResidualLayout layout(b.batch.topology(),
                                        residual_settings(rings, spokes, failure, fail_weight));
            ResidualResult r;
            {
                nb::gil_scoped_release release;
                r = residuals_cpu(b.batch, layout);
            }
            return residuals_dict(std::move(r), layout);
        },
        nb::arg("batch"), nb::arg("rings") = 10, nb::arg("spokes") = 8, nb::arg("failure") = "soft",
        nb::arg("fail_weight") = nb::none(),
        "CPU-Referenz des Kernels RESIDUALS für alle Varianten. Rückgabe: dict mit residuals "
        "(V, stride) [Wellen bei λ_ref], opd und status (V, rays_per_variant), rms_mm und "
        "failed (V, F, L), offsets (F, L) und den Größen der Ablage (Kopien).");
    m.def(
        "merit_cpu",
        [](const PyBatch& b, int rings, int spokes, const std::string& failure,
           std::optional<double> fail_weight) {
            const ResidualLayout layout(b.batch.topology(),
                                        residual_settings(rings, spokes, failure, fail_weight));
            MeritResult r;
            {
                nb::gil_scoped_release release;
                r = merit_cpu(b.batch, layout, false);
            }
            return merit_dict(std::move(r));
        },
        nb::arg("batch"), nb::arg("rings") = 10, nb::arg("spokes") = 8, nb::arg("failure") = "soft",
        nb::arg("fail_weight") = nb::none(),
        "CPU-Referenz des Kernels MERIT: dict mit merit, failed und hard_failed pro Variante "
        "(Kopien). Merit in Wellen², ohne Zusatzresiduen.");

    nb::class_<PyEngine>(
        m, "TraceEngine",
        "Trace-Kernel auf einem CUDA-Gerät für viele Varianten (Tranchen nach freiem "
        "Gerätespeicher). Nur in CUDA-Builds; sonst GpuUnavailableError beim Erzeugen.")
        .def(
            "__init__",
            [](PyEngine* self, int device) {
                require_gpu(device);
#ifdef MIKROPTIM_HAS_CUDA
                new (self) PyEngine{std::make_unique<cuda::TraceEngine>(device)};
#else
                new (self) PyEngine{};
#endif
            },
            nb::arg("device") = 0)
        .def_prop_ro("device", [](const PyEngine& e) { return e.engine->device(); })
        .def_prop_ro("device_name", [](const PyEngine& e) { return e.engine->device_name(); })
#ifdef MIKROPTIM_HAS_CUDA
        .def(
            "residuals",
            [](PyEngine& e, const PyBatch& b, int rings, int spokes, const std::string& failure,
               std::optional<double> fail_weight, bool opd, bool status,
               const std::string& precision) {
                cuda::TraceSettings settings;
                settings.quadrature = residual_settings(rings, spokes, failure, fail_weight);
                settings.opd = opd;
                settings.status = status;
                settings.precision =
                    parse_df64(precision) ? cuda::Precision::Df64 : cuda::Precision::Fp64;
                cuda::TraceOutput out;
                {
                    nb::gil_scoped_release release;
                    e.engine->run(b.batch, cuda::TraceMode::Residuals, settings, out);
                }
                return residuals_dict(std::move(out.residuals), out.layout);
            },
            nb::arg("batch"), nb::arg("rings") = 10, nb::arg("spokes") = 8,
            nb::arg("failure") = "soft", nb::arg("fail_weight") = nb::none(), nb::arg("opd") = true,
            nb::arg("status") = true, nb::arg("precision") = "fp64",
            "Kernel RESIDUALS; Rückgabe wie residuals_cpu (opd/status None, wenn nicht "
            "angefordert).")
        .def(
            "merit",
            [](PyEngine& e, const PyBatch& b, int rings, int spokes, const std::string& failure,
               std::optional<double> fail_weight, const std::string& precision) {
                cuda::TraceSettings settings;
                settings.quadrature = residual_settings(rings, spokes, failure, fail_weight);
                settings.precision =
                    parse_df64(precision) ? cuda::Precision::Df64 : cuda::Precision::Fp64;
                cuda::TraceOutput out;
                {
                    nb::gil_scoped_release release;
                    e.engine->run(b.batch, cuda::TraceMode::Merit, settings, out);
                }
                return merit_dict(std::move(out.merit));
            },
            nb::arg("batch"), nb::arg("rings") = 10, nb::arg("spokes") = 8,
            nb::arg("failure") = "soft", nb::arg("fail_weight") = nb::none(),
            nb::arg("precision") = "fp64", "Kernel MERIT; Rückgabe wie merit_cpu.")
        .def(
            "release", [](PyEngine& e) { e.engine->release(); }, "Gibt die Gerätepuffer frei.")
#endif
        ;
}

} // namespace mko::py
