// Erweiterungsmodul mikroptim._mikroptim (M9-T1): Ausnahmen, Version, GPU-Abfrage und die
// Registrierung der Teile. Python-API: python/README.md.

#include "bindings.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nanobind/stl/string.h>

#include <exception>

namespace nb = nanobind;

namespace {

// Typobjekt von mikroptim.FormatError (für den Übersetzer, der die Attribute setzt).
PyObject* g_format_error = nullptr;

// FormatError mit den Attributen json_path, message und file (leere Teile: None).
void translate_format_error(const std::exception_ptr& p, void* /*payload*/) {
    try {
        std::rethrow_exception(p);
    } catch (const mko::FormatError& e) {
        nb::handle type(g_format_error);
        nb::object exc = type(nb::str(e.what()));
        const auto opt = [](const std::string& s) -> nb::object {
            return s.empty() ? nb::none() : nb::object(nb::str(s.c_str()));
        };
        nb::setattr(exc, "json_path", opt(e.json_path()));
        nb::setattr(exc, "message", nb::str(e.message().c_str()));
        nb::setattr(exc, "file", opt(e.file()));
        PyErr_SetObject(g_format_error, exc.ptr());
    }
}

nb::list devices() {
    nb::list out;
#ifdef MIKROPTIM_HAS_CUDA
    for (const int d : mko::cuda::usable_devices()) {
        const mko::cuda::TraceEngine engine(d);
        const mko::cuda::DeviceMemoryInfo mem = mko::cuda::device_memory_info(d);
        nb::dict e;
        e["index"] = d;
        e["name"] = engine.device_name();
        e["total_bytes"] = mem.total_bytes;
        e["free_bytes"] = mem.free_bytes;
        out.append(e);
    }
#endif
    return out;
}

} // namespace

NB_MODULE(_mikroptim, m) {
    m.doc() = "Mikroptim: Raytracing, Auswertung und Optimierung von Mikroskopobjektiven "
              "(Bindings des C++-Kerns, M9-T1)";
    m.attr("__version__") = mko::kVersion;
#ifdef MIKROPTIM_HAS_CUDA
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif

    // Ausnahmen. FormatError trägt JSON-Pfad, Meldung und Datei als Attribute.
    // Die Typobjekte gehören dem Modul (Attribute); die Übersetzer halten nur Zeiger darauf.
    // Der eigene Übersetzer für FormatError wird nach dem von nb::exception registriert und
    // deshalb zuerst gefragt.
    g_format_error = nb::exception<mko::FormatError>(m, "FormatError", PyExc_ValueError).ptr();
    nb::register_exception_translator(translate_format_error);
    nb::exception<mko::GlassError>(m, "GlassError", PyExc_ValueError);
    nb::exception<mko::cuda::CudaError>(m, "CudaError", PyExc_RuntimeError);
    nb::exception<mko::py::GpuUnavailable>(m, "GpuUnavailableError", PyExc_RuntimeError);

    m.def(
        "device_count",
        []() {
#ifdef MIKROPTIM_HAS_CUDA
            return mko::cuda::device_count();
#else
            return 0;
#endif
        },
        "Zahl der sichtbaren CUDA-Geräte (0 ohne CUDA-Build oder ohne Gerät).");
    m.def("devices", &devices,
          "Nutzbare CUDA-Geräte (ab sm_60) als Liste von dict mit index, name, total_bytes, "
          "free_bytes; leer ohne CUDA-Build oder ohne Gerät.");

    mko::py::bind_system(m);
    mko::py::bind_trace(m);
    mko::py::bind_eval(m);
    mko::py::bind_optimize(m);
    mko::py::bind_cli(m);
}
