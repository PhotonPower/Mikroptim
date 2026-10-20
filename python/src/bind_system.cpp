// Python-Bindings: Systeme laden, lesen, als JSON schreiben und Paraxialdaten (M9-T1).

#include "bindings.hpp"
#include "cli/cli_util.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/paraxial/paraxial.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mko::py {

namespace fs = std::filesystem;

PySystem load_py_system(const fs::path& file) {
    PySystem s;
    {
        nb::gil_scoped_release release;
        s.loaded = load_system(file);
    }
    std::error_code ec;
    s.base_dir = fs::absolute(file, ec).parent_path();
    if (ec) {
        s.base_dir = file.parent_path();
    }
    s.file = file;
    return s;
}

PySystem prepare_py_system(SystemDescription system, const fs::path& base_dir) {
    PySystem s;
    std::error_code ec;
    s.base_dir = fs::absolute(base_dir.empty() ? fs::path(".") : base_dir, ec);
    if (ec) {
        s.base_dir = base_dir;
    }
    nb::gil_scoped_release release;
    s.loaded = prepare_system(std::move(system), s.base_dir);
    return s;
}

void require_gpu(int device) {
#ifdef MIKROPTIM_HAS_CUDA
    const int count = cuda::device_count();
    if (count == 0) {
        throw GpuUnavailable("kein CUDA-Gerät gefunden (GPU-Backend nicht verfügbar)");
    }
    if (device < 0 || device >= count) {
        throw std::invalid_argument("device " + std::to_string(device) +
                                    ": Gerät existiert nicht (" + std::to_string(count) +
                                    " CUDA-Gerät(e))");
    }
#else
    (void)device;
    throw GpuUnavailable("ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), GPU-Backend nicht "
                         "verfügbar");
#endif
}

bool parse_df64(const std::string& precision) {
    if (precision == "fp64") {
        return false;
    }
    if (precision == "df64") {
        return true;
    }
    throw std::invalid_argument("precision erwartet \"fp64\" oder \"df64\", nicht \"" + precision +
                                "\"");
}

namespace {

// Paraxialdaten mit der Wellenlänge, bei der sie gerechnet wurden.
struct PyParaxial {
    ParaxialData data;
    double wavelength_nm = 0.0;
};

// Nicht endliche Werte als None (wie die JSON-Ausgabe der CLI, docs/cli.md).
nb::object finite_or_none(double v) {
    if (!std::isfinite(v)) {
        return nb::none();
    }
    return nb::float_(v);
}

// Paraxialdaten im Aufbau des Feldes `paraxial` von `mikroptim trace --json`.
nb::dict paraxial_dict(const PyParaxial& p) {
    const ParaxialData& d = p.data;
    nb::dict j;
    j["wavelength_nm"] = finite_or_none(p.wavelength_nm);
    j["afocal"] = d.afocal;
    j["power"] = finite_or_none(d.power);
    j["focal_length"] = finite_or_none(d.focal_length);
    j["back_focal_distance"] = finite_or_none(d.back_focal_distance);
    j["front_focal_distance"] = finite_or_none(d.front_focal_distance);
    j["front_principal_plane"] = finite_or_none(d.front_principal_plane);
    j["rear_principal_plane"] = finite_or_none(d.rear_principal_plane);
    j["image_defocus"] = finite_or_none(d.image_defocus);
    j["petzval_sum"] = finite_or_none(d.petzval_sum);
    j["petzval_curvature"] = finite_or_none(d.petzval_curvature);
    j["petzval_radius"] = finite_or_none(d.petzval_radius);
    nb::dict ep;
    ep["at_infinity"] = d.exit_pupil_at_infinity;
    ep["position"] = finite_or_none(d.exit_pupil_position);
    ep["magnification"] = finite_or_none(d.exit_pupil_magnification);
    ep["semi_diameter"] = finite_or_none(d.exit_pupil_semi_diameter);
    j["exit_pupil"] = ep;
    nb::list abcd;
    for (const double v : d.abcd) {
        abcd.append(finite_or_none(v));
    }
    j["abcd"] = abcd;
    return j;
}

std::string material_name(const MaterialSpec& m) {
    switch (m.kind) {
    case MaterialKind::Air:
        return "AIR";
    case MaterialKind::Catalog:
        return m.name;
    case MaterialKind::Constant:
        return "constant";
    case MaterialKind::Buchdahl:
        return "buchdahl";
    }
    return "?";
}

const SystemDescription& desc(const PySystem& s) {
    return s.loaded.system;
}

// Werte pro Fläche als NumPy-Kopie.
template <class F>
auto per_surface(const PySystem& s, F&& value) {
    std::vector<double> v;
    v.reserve(desc(s).surfaces.size());
    for (const Surface& surface : desc(s).surfaces) {
        v.push_back(value(surface));
    }
    return to_numpy(std::move(v));
}

// UTF-8-Text eines Pfads mit den Trennzeichen des Systems (wie os.fspath).
std::string path_text(const fs::path& p) {
    const std::u8string s = p.u8string();
    return std::string(s.begin(), s.end());
}

nb::list variables_list(const PySystem& s) {
    nb::list out;
    const SystemDescription& sys = desc(s);
    if (!sys.variables || sys.variables->items.empty()) {
        return out;
    }
    const VariableSet set(sys, topology_of(sys, s.loaded.index));
    const std::vector<double> x = set.extract(sys);
    for (int k = 0; k < set.size(); ++k) {
        const Variable& v = set[k];
        nb::dict e;
        e["name"] = v.name;
        e["type"] = std::string(to_string(v.type));
        const bool ghost = v.type == VariableType::GhostCoefficient;
        e["surface"] = ghost ? nb::none() : nb::object(nb::int_(v.file_surface));
        e["plate"] = ghost ? nb::object(nb::int_(v.ghost_plate)) : nb::none();
        e["order"] = ghost ? nb::object(nb::int_(v.ghost_order)) : nb::none();
        e["lower"] = v.lower;
        e["upper"] = v.upper;
        e["typical"] = v.typical;
        e["value"] = x[sz(k)];
        out.append(e);
    }
    return out;
}

} // namespace

void bind_system(nb::module_& m) {
    nb::class_<PySystem>(m, "System",
                         "Geladenes optisches System (mikroptim.system/1) mit Glaskatalogen, "
                         "Brechzahl- und Flächentabelle. Unveränderlich; Änderungen über "
                         "to_json()/System.from_json() oder als Ergebnis von optimize(). "
                         "Arrays sind Kopien.")
        .def_static(
            "from_json",
            [](const std::string& text, const fs::path& base_dir) {
                std::vector<std::string> warnings;
                SystemDescription sys = read_system_json(text, &warnings);
                PySystem s = prepare_py_system(std::move(sys), base_dir);
                s.loaded.warnings.insert(s.loaded.warnings.begin(), warnings.begin(),
                                         warnings.end());
                return s;
            },
            nb::arg("text"), nb::arg("base_dir") = fs::path("."),
            "System aus JSON-Text (mikroptim.system/1); relative Glaskataloge werden ab "
            "base_dir gesucht (und in dessen Elternverzeichnissen).")
        .def(
            "to_json", [](const PySystem& s) { return write_system_json(desc(s)); },
            "Systembeschreibung als JSON-Text (verlustfreie Zahlen, Glaskataloge wie geladen).")
        .def(
            "save",
            [](const PySystem& s, const fs::path& file) {
                std::error_code ec;
                fs::path out_dir = fs::absolute(file, ec).parent_path();
                if (ec) {
                    out_dir = file.parent_path();
                }
                SystemDescription sys = desc(s);
                sys.glass_catalogs = cli::catalogs_for(sys, s.loaded.catalog_files, out_dir);
                write_system_file(sys, file);
            },
            nb::arg("path"),
            "Schreibt die Systemdatei; Einträge von glass_catalogs werden so angepasst, dass "
            "sie vom Verzeichnis der neuen Datei aus gefunden werden (wie mikroptim optimize -o).")
        .def_prop_ro("name", [](const PySystem& s) { return desc(s).name; })
        .def_prop_ro("description", [](const PySystem& s) { return desc(s).description; })
        .def_prop_ro(
            "file",
            [](const PySystem& s) -> std::optional<std::string> {
                if (s.file.empty()) {
                    return std::nullopt;
                }
                return path_text(s.file);
            },
            "Systemdatei wie übergeben, None ohne Datei (from_json, Optimierungsergebnis).")
        .def_prop_ro("base_dir", [](const PySystem& s) { return path_text(s.base_dir); })
        .def_prop_ro(
            "warnings", [](const PySystem& s) { return s.loaded.warnings; },
            "Warnungen des Laders (ignorierte Felder).")
        .def_prop_ro("glass_catalogs", [](const PySystem& s) { return desc(s).glass_catalogs; })
        .def_prop_ro("catalog_files",
                     [](const PySystem& s) {
                         std::vector<std::string> out;
                         for (const fs::path& p : s.loaded.catalog_files) {
                             out.push_back(path_text(p));
                         }
                         return out;
                     })
        .def_prop_ro("num_surfaces",
                     [](const PySystem& s) { return static_cast<int>(desc(s).surfaces.size()); })
        .def_prop_ro("surface_types",
                     [](const PySystem& s) {
                         std::vector<std::string> out;
                         for (const Surface& f : desc(s).surfaces) {
                             out.emplace_back(to_string(f.type));
                         }
                         return out;
                     })
        .def_prop_ro("curvatures",
                     [](const PySystem& s) {
                         return per_surface(s, [](const Surface& f) { return f.curvature; });
                     })
        .def_prop_ro("thicknesses",
                     [](const PySystem& s) {
                         return per_surface(s, [](const Surface& f) { return f.thickness; });
                     })
        .def_prop_ro("semi_diameters",
                     [](const PySystem& s) {
                         return per_surface(s, [](const Surface& f) { return f.semi_diameter; });
                     })
        .def_prop_ro(
            "materials",
            [](const PySystem& s) {
                std::vector<std::string> out;
                for (const Surface& f : desc(s).surfaces) {
                    out.push_back(material_name(f.material));
                }
                return out;
            },
            "Medium nach jeder Fläche: Katalogname, \"AIR\", \"constant\" oder "
            "\"buchdahl\".")
        .def_prop_ro(
            "indices",
            [](const PySystem& s) {
                const IndexTable& t = s.loaded.index;
                return to_numpy(t.n, {sz(t.surfaces), sz(t.wavelengths)});
            },
            "Brechzahl n[s, l] des Mediums nach Dateifläche s bei Wellenlänge l (Kopie).")
        .def_prop_ro("wavelengths_nm",
                     [](const PySystem& s) { return to_numpy(desc(s).spectrum.wavelengths_nm); })
        .def_prop_ro("wavelength_weights",
                     [](const PySystem& s) { return to_numpy(desc(s).spectrum.weights); })
        .def_prop_ro("reference_nm",
                     [](const PySystem& s) { return desc(s).spectrum.reference_nm; })
        .def_prop_ro("source_type",
                     [](const PySystem& s) { return std::string(to_string(desc(s).source.type)); })
        .def_prop_ro(
            "field_angles_deg",
            [](const PySystem& s) { return to_numpy(desc(s).source.field_angles_deg); },
            "Feldwinkel (kollimierte Quelle; leer bei der Punktquelle).")
        .def_prop_ro("field_weights",
                     [](const PySystem& s) { return to_numpy(desc(s).source.field_weights); })
        .def_prop_ro("source_point",
                     [](const PySystem& s) -> nb::object {
                         const Source& q = desc(s).source;
                         if (q.type != SourceType::Point) {
                             return nb::none();
                         }
                         return nb::cast(
                             to_numpy(std::vector<double>(q.point.begin(), q.point.end())));
                     })
        .def_prop_ro("numerical_aperture",
                     [](const PySystem& s) -> std::optional<double> {
                         const Source& q = desc(s).source;
                         if (q.type != SourceType::Point) {
                             return std::nullopt;
                         }
                         return q.numerical_aperture;
                     })
        .def_prop_ro(
            "target",
            [](const PySystem& s) {
                nb::dict d;
                if (desc(s).target) {
                    const Target& t = *desc(s).target;
                    d["focal_length"] = t.focal_length;
                    d["parfocal_length"] = t.parfocal_length;
                    d["working_distance"] = t.working_distance;
                }
                return d;
            },
            "Sollwerte (focal_length, parfocal_length, working_distance; None, wenn "
            "nicht gesetzt); leer ohne Abschnitt target.")
        .def_prop_ro(
            "num_ghost_plates",
            [](const PySystem& s) { return static_cast<int>(desc(s).ghost_plates.size()); })
        .def("variables", &variables_list,
             "Optimierungsvariablen (Abschnitt variables) als Liste von dict mit name, type, "
             "surface, plate, order, lower, upper, typical und dem aktuellen Wert value.")
        .def("__repr__", [](const PySystem& s) {
            return "<mikroptim.System '" + desc(s).name + "', " +
                   std::to_string(desc(s).surfaces.size()) + " Flächen>";
        });

    m.def(
        "load_system", [](const fs::path& path) { return load_py_system(path); }, nb::arg("path"),
        "Lädt eine Systemdatei mikroptim.system/1 samt Glaskatalogen (relativ zur Datei). "
        "Fehler: FormatError (mit json_path), GlassError, RuntimeError (Datei nicht lesbar).");

    nb::class_<PyParaxial>(m, "Paraxial",
                           "Paraxialdaten bei einer Wellenlänge (Längen in mm, Lagen relativ zu "
                           "Fläche 0 bzw. zur letzten Fläche, docs/numerik.md, Abschnitt 9).")
        .def_ro("wavelength_nm", &PyParaxial::wavelength_nm)
        .def_prop_ro("afocal", [](const PyParaxial& p) { return p.data.afocal; })
        .def_prop_ro("power", [](const PyParaxial& p) { return p.data.power; })
        .def_prop_ro("focal_length", [](const PyParaxial& p) { return p.data.focal_length; })
        .def_prop_ro("back_focal_distance",
                     [](const PyParaxial& p) { return p.data.back_focal_distance; })
        .def_prop_ro("front_focal_distance",
                     [](const PyParaxial& p) { return p.data.front_focal_distance; })
        .def_prop_ro("front_principal_plane",
                     [](const PyParaxial& p) { return p.data.front_principal_plane; })
        .def_prop_ro("rear_principal_plane",
                     [](const PyParaxial& p) { return p.data.rear_principal_plane; })
        .def_prop_ro("image_defocus", [](const PyParaxial& p) { return p.data.image_defocus; })
        .def_prop_ro("petzval_sum", [](const PyParaxial& p) { return p.data.petzval_sum; })
        .def_prop_ro("petzval_curvature",
                     [](const PyParaxial& p) { return p.data.petzval_curvature; })
        .def_prop_ro("petzval_radius", [](const PyParaxial& p) { return p.data.petzval_radius; })
        .def_prop_ro("exit_pupil_at_infinity",
                     [](const PyParaxial& p) { return p.data.exit_pupil_at_infinity; })
        .def_prop_ro("exit_pupil_position",
                     [](const PyParaxial& p) { return p.data.exit_pupil_position; })
        .def_prop_ro("exit_pupil_magnification",
                     [](const PyParaxial& p) { return p.data.exit_pupil_magnification; })
        .def_prop_ro("exit_pupil_semi_diameter",
                     [](const PyParaxial& p) { return p.data.exit_pupil_semi_diameter; })
        .def_prop_ro(
            "abcd",
            [](const PyParaxial& p) {
                return to_numpy(std::vector<double>(p.data.abcd.begin(), p.data.abcd.end()),
                                {2, 2});
            },
            "Systemmatrix [[A, B], [C, D]] in reduzierten Winkeln (Kopie).")
        .def("to_dict", &paraxial_dict,
             "Aufbau wie das Feld paraxial von `mikroptim trace --json` (nicht endliche Werte "
             "als None).")
        .def("__repr__", [](const PyParaxial& p) {
            return "<mikroptim.Paraxial f'=" + std::to_string(p.data.focal_length) + " mm bei " +
                   std::to_string(p.wavelength_nm) + " nm>";
        });

    m.def(
        "paraxial",
        [](const PySystem& s, std::optional<double> wavelength_nm) {
            PyParaxial p;
            p.wavelength_nm = wavelength_nm ? *wavelength_nm : desc(s).spectrum.reference_nm;
            p.data = compute_paraxial(desc(s), s.loaded.catalog, p.wavelength_nm);
            return p;
        },
        nb::arg("system"), nb::arg("wavelength_nm") = nb::none(),
        "Paraxialdaten bei wavelength_nm (Standard: Referenzwellenlänge spectrum.reference_nm), "
        "wie in `mikroptim trace`.");
}

} // namespace mko::py
