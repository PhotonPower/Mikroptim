// Unterbefehl `mikroptim eval` (M8-T1): JSON-Ausgabe gegen eval_cpu und eval_statistics
// (bitgleich), OPD-Karte als CSV gegen eval_opd_map, mit --psf (M8-T2) Kennzahlen gegen psf_cpu
// und PSF-/MTF-Dateien, mit --vector (M8-T5) gegen vector_psf_cpu und Komponentendateien,
// Textausgabe, Fehlerfälle, --gpu (im
// CUDA-Build GPU- gegen CPU-Ausgabe, übersprungen ohne Gerät; sonst Exit-Code 3).

#include "cli/cli_util.hpp"
#include "cli/eval_command.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "mikroptim/eval/vector_psf.hpp"
#include "temp_paths.hpp"
#include "tolerances.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <numbers>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using nlohmann::json;

namespace {

const std::string kEinzellinse = std::string(MKO_SOURCE_DIR) + "/data/systems/einzellinse_bk7.json";

struct RunResult {
    int code = -1;
    std::string out;
    std::string err;
};

RunResult run(std::initializer_list<std::string_view> args) {
    const std::vector<std::string_view> v(args);
    std::ostringstream out;
    std::ostringstream err;
    RunResult r;
    r.code = mko::cli::run_eval_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

// Zerlegt eine CSV-Zeile an Kommas (leere Felder bleiben erhalten).
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::istringstream in(line);
    while (std::getline(in, cell, ',')) {
        out.push_back(cell);
    }
    if (!line.empty() && line.back() == ',') {
        out.emplace_back();
    }
    return out;
}

} // namespace

TEST(CliEval, JsonMatchesLibrary) {
    const RunResult r = run({kEinzellinse, "--json", "--grid", "24"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["format"], "mikroptim.eval/1");
    EXPECT_EQ(j["backend"], "cpu");
    EXPECT_EQ(j["grid"], 24);
    EXPECT_EQ(j["reference_sphere"], "exit-pupil");

    const mko::LoadedSystem ls = mko::load_system(kEinzellinse);
    const mko::SystemBatchHost b = mko::make_system_batch(ls, 1);
    mko::EvalSettings s;
    s.grid = 24;
    const mko::EvalResult e = mko::eval_cpu(b, s);
    const auto stats = mko::eval_statistics(e, 0);
    EXPECT_EQ(j["grid_points"], e.samples());
    EXPECT_EQ(j["exit_pupil_position"].get<double>(), e.exit_pupil_position[0]);
    ASSERT_EQ(j["tuples"].size(), stats.size());
    const double lambda_ref_mm = 587.56 * 1e-6;
    for (std::size_t i = 0; i < stats.size(); ++i) {
        const json& t = j["tuples"][i];
        EXPECT_EQ(t["field"], stats[i].field);
        EXPECT_EQ(t["wavelength"], stats[i].wavelength);
        EXPECT_EQ(t["rms_opd_mm"].get<double>(), stats[i].rms_opd);
        EXPECT_EQ(t["rms_opd_waves"].get<double>(), stats[i].rms_opd / lambda_ref_mm);
        EXPECT_EQ(t["rms_spot_mm"].get<double>(), stats[i].rms_spot);
        EXPECT_EQ(t["focus_sagittal_mm"].get<double>(), stats[i].focus_sagittal);
        EXPECT_EQ(t["failed_rays"], 0);
        const double lambda_mm = b.wavelength_nm(stats[i].wavelength) * 1e-6;
        const double phase = 2.0 * std::numbers::pi * stats[i].rms_opd / lambda_mm;
        EXPECT_NEAR(t["strehl_marechal"].get<double>(), std::exp(-phase * phase),
                    mko::test::kUnitFp64);
        if (stats[i].field == 0) {
            EXPECT_TRUE(t["distortion_chief_percent"].is_null()); // Achsfeld
        } else {
            EXPECT_EQ(t["distortion_chief_percent"].get<double>(), stats[i].distortion_chief);
        }
    }
    EXPECT_EQ(j["failed_rays"], 0);

    // Lage Scheitel und halbe Pupille werden durchgereicht
    const RunResult v =
        run({kEinzellinse, "--json", "--grid", "24", "--reference", "vertex", "--half"});
    ASSERT_EQ(v.code, mko::cli::kExitOk) << v.err;
    const json jv = json::parse(v.out);
    EXPECT_EQ(jv["reference_sphere"], "vertex");
    EXPECT_EQ(jv["half_pupil"], true);
    EXPECT_TRUE(jv["exit_pupil_position"].is_null());
}

TEST(CliEval, OpdMapCsvMatchesLibrary) {
    const std::filesystem::path file = mko::test::temp_dir() / "mikroptim_test_eval_map.csv";
    const std::string path = file.string();
    const RunResult r =
        run({kEinzellinse, "--grid", "16", "--map", "2,1", "--map-file", path, "--remove", "tilt"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const mko::LoadedSystem ls = mko::load_system(kEinzellinse);
    mko::EvalSettings s;
    s.grid = 16;
    const mko::EvalResult e = mko::eval_cpu(mko::make_system_batch(ls, 1), s);
    const mko::EvalMap m = mko::eval_opd_map(e, 0, 2, 1, mko::OpdRemoval::Tilt);
    std::ifstream in(file);
    std::string line;
    int row = 0;
    while (std::getline(in, line)) {
        const std::vector<std::string> cells = split(line);
        ASSERT_EQ(cells.size(), 16u);
        const int j = 15 - row; // erste Zeile = größtes y
        for (int i = 0; i < 16; ++i) {
            const double v = m.at(i, j);
            if (std::isnan(v)) {
                EXPECT_TRUE(cells[static_cast<std::size_t>(i)].empty());
            } else {
                // verlustfrei geschrieben (17 Stellen), Umrechnung wie im Programm
                EXPECT_EQ(std::stod(cells[static_cast<std::size_t>(i)]), v / (587.56 * 1e-6));
            }
        }
        ++row;
    }
    EXPECT_EQ(row, 16);
    in.close();
    std::error_code ec;
    std::filesystem::remove(file, ec);
}

TEST(CliEval, TextOutputAndErrors) {
    const RunResult r = run({kEinzellinse, "--grid", "16"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("System: Einzellinse N-BK7"), std::string::npos);
    EXPECT_NE(r.out.find("Referenzsphäre: durch die Austrittspupille"), std::string::npos);
    EXPECT_NE(r.out.find("Versagte Strahlen: 0 von"), std::string::npos);

    EXPECT_EQ(run({kEinzellinse, "--grid", "0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--reference", "irgendwo"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--map", "0,0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--device", "0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--unbekannt"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({"--help"}).code, mko::cli::kExitOk);
    const RunResult missing =
        run({std::string(MKO_SOURCE_DIR) + "/data/systems/gibt_es_nicht.json"});
    EXPECT_EQ(missing.code, mko::cli::kExitError);
    EXPECT_NE(missing.err.find("mikroptim eval: Fehler"), std::string::npos);
    const RunResult bad_map = run({kEinzellinse, "--grid", "8", "--map", "7,0", "--map-file",
                                   (mko::test::temp_dir() / "x.csv").string()});
    EXPECT_EQ(bad_map.code, mko::cli::kExitError);
}

TEST(CliEval, PsfJsonMatchesLibrary) {
    const RunResult r = run({kEinzellinse, "--json", "--grid", "24", "--psf", "--padding", "3"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    const mko::LoadedSystem ls = mko::load_system(kEinzellinse);
    const mko::SystemBatchHost b = mko::make_system_batch(ls, 1);
    mko::EvalSettings s;
    s.grid = 24;
    const mko::EvalResult e = mko::eval_cpu(b, s);
    mko::PsfSettings ps;
    ps.padding = 3.0;
    const std::vector<mko::PsfResult> p = mko::psf_cpu(b, e, 0, ps);
    EXPECT_EQ(j["psf"]["fft_size"], 128);
    EXPECT_EQ(j["psf"]["padding"].get<double>(), 3.0);
    EXPECT_TRUE(j["psf"]["numerical_aperture"].is_null());
    ASSERT_EQ(j["tuples"].size(), p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        const json& t = j["tuples"][i]["psf"];
        EXPECT_EQ(t["strehl"].get<double>(), p[i].strehl);
        EXPECT_EQ(t["strehl_fft"].get<double>(), p[i].strehl_fft);
        EXPECT_EQ(t["strehl_origin"].get<double>(), p[i].strehl_origin);
        EXPECT_EQ(t["peak"].get<double>(), p[i].peak);
        EXPECT_EQ(t["airy_radius_mm"].get<double>(), p[i].airy_radius);
        EXPECT_EQ(t["na"][1].get<double>(), p[i].na_y);
        EXPECT_EQ(t["pixel_mm"][0].get<double>(), p[i].dx);
        EXPECT_EQ(t["mtf_half_cutoff"][1].get<double>(), p[i].mtf_tangential_at(0.5));
        EXPECT_EQ(t["energy"].get<double>(), p[i].energy);
    }
    // ohne --psf: null
    const json n = json::parse(run({kEinzellinse, "--json", "--grid", "8"}).out);
    EXPECT_TRUE(n["psf"].is_null());
    EXPECT_TRUE(n["tuples"][0]["psf"].is_null());
    // NA vorgegeben
    const json v = json::parse(run({kEinzellinse, "--json", "--grid", "16", "--na", "0.1"}).out);
    EXPECT_EQ(v["psf"]["numerical_aperture"].get<double>(), 0.1);
    EXPECT_EQ(v["tuples"][0]["psf"]["na"][0].get<double>(), 0.1);
}

TEST(CliEval, PsfAndMtfFilesMatchLibrary) {
    const auto dir = mko::test::temp_dir();
    const std::string psf_path = (dir / "mikroptim_test_eval_psf.csv").string();
    const std::string mtf_path = (dir / "mikroptim_test_eval_mtf.csv").string();
    const RunResult r = run({kEinzellinse, "--grid", "16", "--psf-tuple", "1,2", "--psf-file",
                             psf_path, "--mtf-file", mtf_path});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("PSF skalar (FFT 64 x 64"), std::string::npos);
    // volle Öffnung (Spot ~0,15 mm) bei N = 16: Bildfenster λN/(2 NA) ≈ 0,04 mm zu klein
    EXPECT_NE(r.err.find("PSF zurückgefaltet"), std::string::npos);
    const mko::LoadedSystem ls = mko::load_system(kEinzellinse);
    const mko::SystemBatchHost b = mko::make_system_batch(ls, 1);
    mko::EvalSettings s;
    s.grid = 16;
    const mko::EvalResult e = mko::eval_cpu(b, s);
    const mko::PsfResult p = mko::psf_cpu(mko::make_psf_pupil(b, e, 0, 1, 2), 64);
    {
        std::ifstream in(psf_path);
        std::string line;
        int row = 0;
        while (std::getline(in, line)) {
            const std::vector<std::string> cells = split(line);
            ASSERT_EQ(cells.size(), 64u);
            const int jj = 63 - row; // erste Zeile = größtes y
            for (int i = 0; i < 64; ++i) {
                EXPECT_EQ(std::stod(cells[static_cast<std::size_t>(i)]), p.psf_at(i, jj));
            }
            ++row;
        }
        EXPECT_EQ(row, 64);
    }
    {
        std::ifstream in(mtf_path);
        std::string line;
        std::getline(in, line);
        EXPECT_EQ(line, "k,nu,f_sagittal_lpmm,mtf_sagittal,f_tangential_lpmm,mtf_tangential");
        int k = 0;
        while (std::getline(in, line)) {
            const std::vector<std::string> cells = split(line);
            ASSERT_EQ(cells.size(), 6u);
            EXPECT_EQ(std::stoi(cells[0]), k);
            EXPECT_EQ(std::stod(cells[3]), p.mtf_sagittal[static_cast<std::size_t>(k)]);
            EXPECT_EQ(std::stod(cells[5]), p.mtf_tangential[static_cast<std::size_t>(k)]);
            ++k;
        }
        EXPECT_EQ(k, 17);
    }
    std::error_code ec;
    std::filesystem::remove(psf_path, ec);
    std::filesystem::remove(mtf_path, ec);

    EXPECT_EQ(run({kEinzellinse, "--padding", "1.5"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--na", "0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--psf-tuple", "1"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--grid", "8", "--psf-tuple", "9,0"}).code, mko::cli::kExitError);
}

TEST(CliEval, VectorPsfJsonMatchesLibrary) {
    const RunResult r = run({kEinzellinse, "--json", "--grid", "24", "--padding", "3", "--vector",
                             "--polarization", "unpolarized", "--defocus", "0.001"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    const mko::LoadedSystem ls = mko::load_system(kEinzellinse);
    const mko::SystemBatchHost b = mko::make_system_batch(ls, 1);
    mko::EvalSettings s;
    s.grid = 24;
    const mko::EvalResult e = mko::eval_cpu(b, s);
    mko::PsfSettings ps;
    ps.padding = 3.0;
    mko::VectorPsfSettings vs;
    vs.polarization = mko::Polarization::Unpolarized;
    vs.defocus = 0.001;
    const std::vector<mko::VectorPsfResult> p = mko::vector_psf_cpu(b, e, 0, ps, vs);
    EXPECT_EQ(j["psf"]["model"], "vector");
    EXPECT_EQ(j["psf"]["polarization"], "unpolarized");
    EXPECT_EQ(j["psf"]["defocus_mm"].get<double>(), 0.001);
    EXPECT_EQ(j["psf"]["fft_size"], 128);
    ASSERT_EQ(j["tuples"].size(), p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        const json& t = j["tuples"][i]["psf"];
        EXPECT_EQ(t["strehl"].get<double>(), p[i].strehl);
        EXPECT_EQ(t["strehl_fft"].get<double>(), p[i].strehl_fft);
        EXPECT_EQ(t["peak"].get<double>(), p[i].peak);
        EXPECT_EQ(t["energy"].get<double>(), p[i].energy);
        const json& v = t["vector"];
        EXPECT_EQ(v["fwhm_mm"][0].get<double>(), p[i].fwhm_x);
        EXPECT_EQ(v["fwhm_mm"][1].get<double>(), p[i].fwhm_y);
        EXPECT_EQ(v["energy_fraction"][2].get<double>(), p[i].energy_fraction[2]);
        EXPECT_EQ(v["component_peak"][2].get<double>(), p[i].component_peak[2]);
        EXPECT_EQ(v["reference_intensity"].get<double>(), p[i].reference_intensity);
        EXPECT_EQ(v["evanescent_points"].get<long long>(), p[i].evanescent);
    }
    // skalar: Modell "scalar", keine vektoriellen Felder
    const json sc = json::parse(run({kEinzellinse, "--json", "--grid", "8", "--psf"}).out);
    EXPECT_EQ(sc["psf"]["model"], "scalar");
    EXPECT_TRUE(sc["psf"]["polarization"].is_null());
    EXPECT_TRUE(sc["tuples"][0]["psf"]["vector"].is_null());
}

TEST(CliEval, VectorComponentFilesAndErrors) {
    const auto dir = mko::test::temp_dir();
    const std::string prefix = (dir / "mikroptim_test_eval_vpsf").string();
    const std::string psf_path = prefix + "_total.csv";
    const RunResult r = run({kEinzellinse, "--grid", "16", "--psf-tuple", "1,2", "--polarization",
                             "y", "--psf-file", psf_path, "--component-files", prefix});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("PSF vektoriell (Richards-Wolf, Polarisation y"), std::string::npos);
    const mko::LoadedSystem ls = mko::load_system(kEinzellinse);
    const mko::SystemBatchHost b = mko::make_system_batch(ls, 1);
    mko::EvalSettings s;
    s.grid = 16;
    const mko::EvalResult e = mko::eval_cpu(b, s);
    mko::VectorPsfSettings vs;
    vs.polarization = mko::Polarization::Y;
    const mko::VectorPsfResult p =
        mko::vector_psf_cpu(mko::make_vector_psf_pupil(b, e, 0, 1, 2), 64, vs);
    const auto check = [&](const std::string& path, const std::vector<double>& values) {
        std::ifstream in(path);
        std::string line;
        int row = 0;
        while (std::getline(in, line)) {
            const std::vector<std::string> cells = split(line);
            EXPECT_EQ(cells.size(), 64u);
            const int jj = 63 - row;
            for (int i = 0; i < 64 && i < static_cast<int>(cells.size()); ++i) {
                EXPECT_EQ(std::stod(cells[static_cast<std::size_t>(i)]),
                          values[static_cast<std::size_t>(jj) * 64 + static_cast<std::size_t>(i)]);
            }
            ++row;
        }
        EXPECT_EQ(row, 64) << path;
    };
    check(psf_path, p.psf);
    check(prefix + "_ex.csv", p.component[0]);
    check(prefix + "_ey.csv", p.component[1]);
    check(prefix + "_ez.csv", p.component[2]);
    std::error_code ec;
    for (const char* suffix : {"_total.csv", "_ex.csv", "_ey.csv", "_ez.csv"}) {
        std::filesystem::remove(prefix + suffix, ec);
    }
    EXPECT_EQ(run({kEinzellinse, "--polarization", "z"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--defocus", "nan"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--defocus"}).code, mko::cli::kExitUsage);
}

#ifdef MIKROPTIM_HAS_CUDA
TEST(CliEvalGpu, VectorPsfMatchesCpu) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const RunResult c = run({kEinzellinse, "--json", "--grid", "32", "--vector"});
    const RunResult g = run({kEinzellinse, "--json", "--grid", "32", "--vector", "--gpu"});
    ASSERT_EQ(c.code, mko::cli::kExitOk) << c.err;
    ASSERT_EQ(g.code, mko::cli::kExitOk) << g.err;
    const json jc = json::parse(c.out);
    const json jg = json::parse(g.out);
    ASSERT_EQ(jc["tuples"].size(), jg["tuples"].size());
    for (std::size_t i = 0; i < jc["tuples"].size(); ++i) {
        const json& a = jc["tuples"][i]["psf"];
        const json& b = jg["tuples"][i]["psf"];
        // EVAL auf der GPU (OPD bis ~1e-13 mm verschieden) und PSF über cuFFT
        EXPECT_NEAR(b["strehl_fft"].get<double>(), a["strehl_fft"].get<double>(), 1e-9);
        EXPECT_NEAR(b["strehl"].get<double>(), a["strehl"].get<double>(), 1e-9);
        EXPECT_NEAR(b["vector"]["energy_fraction"][2].get<double>(),
                    a["vector"]["energy_fraction"][2].get<double>(), 1e-9);
    }
}

TEST(CliEvalGpu, PsfMatchesCpu) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const RunResult c = run({kEinzellinse, "--json", "--grid", "32", "--psf"});
    const RunResult g = run({kEinzellinse, "--json", "--grid", "32", "--psf", "--gpu"});
    ASSERT_EQ(c.code, mko::cli::kExitOk) << c.err;
    ASSERT_EQ(g.code, mko::cli::kExitOk) << g.err;
    const json jc = json::parse(c.out);
    const json jg = json::parse(g.out);
    ASSERT_EQ(jc["tuples"].size(), jg["tuples"].size());
    for (std::size_t i = 0; i < jc["tuples"].size(); ++i) {
        const json& a = jc["tuples"][i]["psf"];
        const json& b = jg["tuples"][i]["psf"];
        // NA aus den Feldstrahlen (GPU-EVAL) und PSF über cuFFT
        EXPECT_NEAR(b["strehl_fft"].get<double>(), a["strehl_fft"].get<double>(), 1e-9);
        EXPECT_NEAR(b["strehl"].get<double>(), a["strehl"].get<double>(), 1e-9);
        EXPECT_NEAR(b["na"][1].get<double>(), a["na"][1].get<double>(), 1e-9);
    }
}

TEST(CliEvalGpu, MatchesCpu) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const RunResult c = run({kEinzellinse, "--json", "--grid", "32"});
    const RunResult g = run({kEinzellinse, "--json", "--grid", "32", "--gpu"});
    ASSERT_EQ(c.code, mko::cli::kExitOk) << c.err;
    ASSERT_EQ(g.code, mko::cli::kExitOk) << g.err;
    const json jc = json::parse(c.out);
    const json jg = json::parse(g.out);
    EXPECT_EQ(jg["backend"], "gpu");
    ASSERT_EQ(jc["tuples"].size(), jg["tuples"].size());
    for (std::size_t i = 0; i < jc["tuples"].size(); ++i) {
        const double a = jc["tuples"][i]["rms_opd_mm"].get<double>();
        const double b = jg["tuples"][i]["rms_opd_mm"].get<double>();
        EXPECT_NEAR(b, a, std::max(mko::test::kCpuGpuFp64, mko::test::kCpuGpuFp64Relative * a));
    }
    EXPECT_EQ(jg["failed_rays"], jc["failed_rays"]);
}
#else
TEST(CliEval, GpuUnavailableWithoutCuda) {
    const RunResult r = run({kEinzellinse, "--gpu"});
    EXPECT_EQ(r.code, mko::cli::kExitNoGpu);
    EXPECT_NE(r.err.find("ohne CUDA gebaut"), std::string::npos);
}
#endif
