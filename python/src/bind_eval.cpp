// Python-Bindings: Auswertung auf dem dichten Pupillenraster (Modus EVAL) und skalare PSF, MTF,
// Strehl (M9-T1).

#include "bindings.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/psf.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mko::py {

namespace {

// Ergebnis von evaluate(): Batch (eine Variante), Rohdaten und Kennzahlen.
struct PyEvaluation {
    SystemBatchHost batch;
    EvalResult result;
    std::vector<EvalTupleStats> stats; // [F·L]
    bool gpu = false;
    int device = -1;

    void check_tuple(int f, int l) const {
        if (f < 0 || f >= result.fields || l < 0 || l >= result.wavelengths) {
            throw std::out_of_range("Feld " + std::to_string(f) + " oder Wellenlänge " +
                                    std::to_string(l) + " existiert nicht");
        }
    }
};

// Skalare PSF eines Tupels.
struct PyPsf {
    PsfResult result;
    bool gpu = false;
};

OpdRemoval parse_removal(const std::string& remove) {
    if (remove == "piston") {
        return OpdRemoval::Piston;
    }
    if (remove == "tilt") {
        return OpdRemoval::Tilt;
    }
    if (remove == "defocus") {
        return OpdRemoval::Defocus;
    }
    throw std::invalid_argument("remove erwartet \"piston\", \"tilt\" oder \"defocus\", nicht \"" +
                                remove + "\"");
}

EvalReference parse_reference(const std::string& reference) {
    if (reference == "exit-pupil") {
        return EvalReference::ExitPupil;
    }
    if (reference == "vertex") {
        return EvalReference::Vertex;
    }
    throw std::invalid_argument("reference erwartet \"exit-pupil\" oder \"vertex\", nicht \"" +
                                reference + "\"");
}

PyEvaluation evaluate_system(const PySystem& s, int grid, bool half, const std::string& reference,
                             double field_ray_offset, bool gpu, int device) {
    EvalSettings settings;
    settings.grid = grid;
    settings.half_pupil = half;
    settings.reference = parse_reference(reference);
    settings.field_ray_offset = field_ray_offset;
    if (gpu) {
        require_gpu(device);
    }
    nb::gil_scoped_release release;
    PyEvaluation e;
    e.batch = make_system_batch(s.loaded, 1);
#ifdef MIKROPTIM_HAS_CUDA
    if (gpu) {
        cuda::TraceEngine engine(device);
        engine.evaluate(e.batch, settings, e.result);
        e.gpu = true;
        e.device = device;
    } else {
        e.result = eval_cpu(e.batch, settings);
    }
#else
    e.result = eval_cpu(e.batch, settings);
#endif
    e.stats = eval_statistics(e.result, 0);
    return e;
}

// Kennzahl `get` aller Tupel als Array (F, L).
auto stat_array(const PyEvaluation& e, const std::function<double(const EvalTupleStats&)>& get) {
    std::vector<double> v;
    v.reserve(e.stats.size());
    for (const EvalTupleStats& t : e.stats) {
        v.push_back(get(t));
    }
    return to_numpy(std::move(v), {sz(e.result.fields), sz(e.result.wavelengths)});
}

nb::dict stats_dict(const PyEvaluation& e) {
    using S = EvalTupleStats;
    nb::dict d;
    d["rays"] = stat_array(e, [](const S& t) { return static_cast<double>(t.rays); });
    d["failed"] = stat_array(e, [](const S& t) { return static_cast<double>(t.failed); });
    d["valid_weight"] = stat_array(e, [](const S& t) { return t.valid_weight; });
    d["rms_opd"] = stat_array(e, [](const S& t) { return t.rms_opd; });
    d["pv_opd"] = stat_array(e, [](const S& t) { return t.pv_opd; });
    d["rms_opd_tilt"] = stat_array(e, [](const S& t) { return t.rms_opd_tilt; });
    d["rms_opd_defocus"] = stat_array(e, [](const S& t) { return t.rms_opd_defocus; });
    d["tilt_x"] = stat_array(e, [](const S& t) { return t.tilt_x; });
    d["tilt_y"] = stat_array(e, [](const S& t) { return t.tilt_y; });
    d["defocus"] = stat_array(e, [](const S& t) { return t.defocus; });
    d["centroid_x"] = stat_array(e, [](const S& t) { return t.centroid_x; });
    d["centroid_y"] = stat_array(e, [](const S& t) { return t.centroid_y; });
    d["rms_spot"] = stat_array(e, [](const S& t) { return t.rms_spot; });
    d["image_height"] = stat_array(e, [](const S& t) { return t.image_height; });
    d["chief_x"] = stat_array(e, [](const S& t) { return t.chief_x; });
    d["chief_y"] = stat_array(e, [](const S& t) { return t.chief_y; });
    d["distortion_chief"] = stat_array(e, [](const S& t) { return t.distortion_chief; });
    d["distortion_centroid"] = stat_array(e, [](const S& t) { return t.distortion_centroid; });
    d["focus_tangential"] = stat_array(e, [](const S& t) { return t.focus_tangential; });
    d["focus_sagittal"] = stat_array(e, [](const S& t) { return t.focus_sagittal; });
    d["astigmatism"] = stat_array(e, [](const S& t) { return t.astigmatism(); });
    return d;
}

PyPsf compute_psf(const PyEvaluation& e, int field, int wavelength, double padding,
                  double numerical_aperture, bool gpu, int device) {
    e.check_tuple(field, wavelength);
    PsfSettings settings;
    settings.padding = padding;
    settings.numerical_aperture = numerical_aperture;
    check_psf_settings(settings);
    if (gpu) {
        require_gpu(device);
    }
    nb::gil_scoped_release release;
    const PsfPupil pupil = make_psf_pupil(e.batch, e.result, 0, field, wavelength, settings);
    const int fft = psf_fft_size(e.result.settings.grid, settings.padding);
    PyPsf p;
#ifdef MIKROPTIM_HAS_CUDA
    if (gpu) {
        cuda::PsfEngine engine(device);
        p.result = std::move(engine.compute(std::span<const PsfPupil>(&pupil, 1), fft)[0]);
        p.gpu = true;
        return p;
    }
#endif
    p.result = psf_cpu(pupil, fft);
    return p;
}

} // namespace

void bind_eval(nb::module_& m) {
    nb::class_<PyPsf>(m, "Psf",
                      "Skalare PSF, MTF-Schnitte und Strehl eines Tupels (psf_cpu bzw. "
                      "PsfEngine). Arrays sind schreibgeschützte Sichten auf das Objekt.")
        .def_prop_ro("field", [](const PyPsf& p) { return p.result.field; })
        .def_prop_ro("wavelength", [](const PyPsf& p) { return p.result.wavelength; })
        .def_prop_ro("wavelength_nm", [](const PyPsf& p) { return p.result.wavelength_nm; })
        .def_prop_ro("backend", [](const PyPsf& p) { return p.gpu ? "gpu" : "cpu"; })
        .def_prop_ro("grid", [](const PyPsf& p) { return p.result.grid; })
        .def_prop_ro("fft_size", [](const PyPsf& p) { return p.result.fft_size; })
        .def_prop_ro("na_x", [](const PyPsf& p) { return p.result.na_x; })
        .def_prop_ro("na_y", [](const PyPsf& p) { return p.result.na_y; })
        .def_prop_ro(
            "dx", [](const PyPsf& p) { return p.result.dx; }, "Bildpunktabstand x [mm].")
        .def_prop_ro(
            "dy", [](const PyPsf& p) { return p.result.dy; }, "Bildpunktabstand y [mm].")
        .def_prop_ro("airy_radius", [](const PyPsf& p) { return p.result.airy_radius; })
        .def_prop_ro(
            "strehl", [](const PyPsf& p) { return p.result.strehl; },
            "Strehl direkt nach Abzug der Kippung.")
        .def_prop_ro("strehl_fft", [](const PyPsf& p) { return p.result.strehl_fft; })
        .def_prop_ro("strehl_origin", [](const PyPsf& p) { return p.result.strehl_origin; })
        .def_prop_ro("energy", [](const PyPsf& p) { return p.result.energy; })
        .def_prop_ro("transmitted", [](const PyPsf& p) { return p.result.transmitted; })
        .def_prop_ro("peak", [](const PyPsf& p) { return p.result.peak; })
        .def_prop_ro(
            "peak_position",
            [](const PyPsf& p) { return std::make_pair(p.result.peak_x, p.result.peak_y); })
        .def_prop_ro(
            "cutoff_x", [](const PyPsf& p) { return p.result.cutoff_x(); },
            "Grenzfrequenz 2 NA_x/λ [Linienpaare/mm].")
        .def_prop_ro("cutoff_y", [](const PyPsf& p) { return p.result.cutoff_y(); })
        .def_prop_ro(
            "psf",
            [](const PyPsf& p) {
                const auto size = sz(p.result.fft_size);
                return readonly_view(p.result.psf.data(), {size, size}, {});
            },
            nb::rv_policy::reference_internal,
            "PSF (M, M), psf[j, i] im Bildpunkt ((i − M/2)·dx, (j − M/2)·dy) relativ zum idealen "
            "Bildpunkt, normiert auf das Maximum der ungestörten Pupille.")
        .def_prop_ro(
            "mtf_sagittal",
            [](const PyPsf& p) {
                return readonly_view(p.result.mtf_sagittal.data(), {p.result.mtf_sagittal.size()},
                                     {});
            },
            nb::rv_policy::reference_internal,
            "MTF sagittal bei f = k·f_c/N, k = 0 … N (normiert auf MTF(0) = 1).")
        .def_prop_ro(
            "mtf_tangential",
            [](const PyPsf& p) {
                return readonly_view(p.result.mtf_tangential.data(),
                                     {p.result.mtf_tangential.size()}, {});
            },
            nb::rv_policy::reference_internal, "MTF tangential bei f = k·f_c/N, k = 0 … N.")
        .def(
            "frequencies",
            [](const PyPsf& p) {
                std::vector<double> nu;
                for (int k = 0; k <= p.result.grid; ++k) {
                    nu.push_back(static_cast<double>(k) / static_cast<double>(p.result.grid));
                }
                return to_numpy(std::move(nu));
            },
            "Normierte Frequenzen ν = f/f_c der MTF-Stützstellen (Kopie); f = ν · cutoff_x "
            "bzw. cutoff_y.");

    nb::class_<PyEvaluation>(
        m, "Evaluation",
        "Auswertung auf dem dichten kartesischen Pupillenraster (Modus EVAL, eine Variante). "
        "Rückgabewerte der Methoden und von stats sind NumPy-Kopien.")
        .def_prop_ro("backend", [](const PyEvaluation& e) { return e.gpu ? "gpu" : "cpu"; })
        .def_prop_ro("grid", [](const PyEvaluation& e) { return e.result.settings.grid; })
        .def_prop_ro("half_pupil",
                     [](const PyEvaluation& e) { return e.result.settings.half_pupil; })
        .def_prop_ro("grid_points", [](const PyEvaluation& e) { return e.result.samples(); })
        .def_prop_ro("fields", [](const PyEvaluation& e) { return e.result.fields; })
        .def_prop_ro("wavelengths", [](const PyEvaluation& e) { return e.result.wavelengths; })
        .def_prop_ro("reference_nm", [](const PyEvaluation& e) { return e.result.reference_nm; })
        .def_prop_ro("exit_pupil_position",
                     [](const PyEvaluation& e) { return e.result.exit_pupil_position[0]; })
        .def_prop_ro("failed_rays", [](const PyEvaluation& e) { return e.result.failed_count(); })
        .def_prop_ro("stats", &stats_dict,
                     "Kennzahlen pro Tupel (eval_statistics) als dict von Arrays der Form (F, L): "
                     "rms_opd, pv_opd, rms_opd_tilt, rms_opd_defocus, tilt_x, tilt_y, defocus "
                     "[mm]; centroid_x, centroid_y, rms_spot, chief_x, chief_y, image_height "
                     "[mm]; distortion_chief, distortion_centroid [%]; focus_tangential, "
                     "focus_sagittal, astigmatism [mm]; rays, failed, valid_weight.")
        .def(
            "pupil_points",
            [](const PyEvaluation& e) {
                std::vector<double> v;
                for (const EvalPoint& p : e.result.grid.points) {
                    v.push_back(p.x);
                    v.push_back(p.y);
                    v.push_back(p.weight);
                }
                return to_numpy(std::move(v), {e.result.grid.points.size(), 3});
            },
            "Rasterpunkte in normierten Pupillenkoordinaten mit Gewicht, Form (n, 3): x, y, w.")
        .def(
            "opd_map",
            [](const PyEvaluation& e, int field, int wavelength, const std::string& remove) {
                e.check_tuple(field, wavelength);
                EvalMap map = eval_opd_map(e.result, 0, field, wavelength, parse_removal(remove));
                const auto n = sz(map.size);
                return to_numpy(std::move(map.values), {n, n});
            },
            nb::arg("field"), nb::arg("wavelength"), nb::arg("remove") = "piston",
            "OPD-Karte [mm] auf dem vollen N×N-Raster, Zeile j = y, Spalte i = x; NaN außerhalb "
            "der Pupille und bei versagten Strahlen. remove: \"piston\" (wie MERIT), \"tilt\", "
            "\"defocus\".")
        .def(
            "spots",
            [](const PyEvaluation& e, int field, int wavelength) {
                e.check_tuple(field, wavelength);
                const std::size_t n = sz(e.result.samples());
                const std::size_t first = e.result.ray_index(0, field, wavelength, 0);
                const auto begin = static_cast<std::ptrdiff_t>(first);
                const auto end = static_cast<std::ptrdiff_t>(first + n);
                nb::dict d;
                d["x"] = to_numpy(std::vector<double>(e.result.spot_x.begin() + begin,
                                                      e.result.spot_x.begin() + end));
                d["y"] = to_numpy(std::vector<double>(e.result.spot_y.begin() + begin,
                                                      e.result.spot_y.begin() + end));
                d["status"] = to_numpy(std::vector<std::uint8_t>(e.result.status.begin() + begin,
                                                                 e.result.status.begin() + end));
                return d;
            },
            nb::arg("field"), nb::arg("wavelength"),
            "Spotdiagramm: Durchstoßpunkte in der Bildebene relativ zum idealen Bildpunkt [mm] "
            "pro Rasterpunkt (Reihenfolge wie pupil_points; NaN bei Versagen) und Status.")
        .def("psf", &compute_psf, nb::arg("field"), nb::arg("wavelength"), nb::arg("padding") = 4.0,
             nb::arg("numerical_aperture") = 0.0, nb::arg("gpu") = false, nb::arg("device") = 0,
             "Skalare PSF, MTF und Strehl des Tupels (wie `mikroptim eval --psf`); "
             "numerical_aperture 0: aus den Feldstrahlen.");

    m.def("evaluate", &evaluate_system, nb::arg("system"), nb::arg("grid") = 64,
          nb::arg("half") = false, nb::arg("reference") = "exit-pupil",
          nb::arg("field_ray_offset") = 1e-3, nb::arg("gpu") = false, nb::arg("device") = 0,
          "Auswertung auf dem dichten Pupillenraster wie `mikroptim eval` (CPU: eval_cpu; "
          "gpu=True: TraceEngine.evaluate). reference: \"exit-pupil\" oder \"vertex\".");
}

} // namespace mko::py
