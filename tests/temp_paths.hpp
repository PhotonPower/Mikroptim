#pragma once

// Temporäres Verzeichnis pro Testprozess. Mehrere Testläufe auf demselben Rechner (parallele
// ctest-Jobs, mehrere Arbeitsbäume) dürfen sich keine Dateien im gemeinsamen System-Temp teilen:
// Ein fester Name wie temp/mikroptim_cli_optimize_gpu würde vom anderen Lauf gelöscht oder
// überschrieben. Jeder Prozess bekommt deshalb temp/mikroptim_tests_<pid>; es wird beim
// Prozessende samt Inhalt entfernt.

#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace mko::test {

namespace detail {

class ProcessTempDir {
public:
    ProcessTempDir() {
#ifdef _WIN32
        const long pid = static_cast<long>(_getpid());
#else
        const long pid = static_cast<long>(getpid());
#endif
        path_ = std::filesystem::temp_directory_path() / ("mikroptim_tests_" + std::to_string(pid));
        std::error_code ec;
        std::filesystem::create_directories(path_, ec);
    }
    ~ProcessTempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    ProcessTempDir(const ProcessTempDir&) = delete;
    ProcessTempDir& operator=(const ProcessTempDir&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

} // namespace detail

// Verzeichnis für temporäre Testdateien dieses Prozesses (existiert nach dem Aufruf).
inline const std::filesystem::path& temp_dir() {
    static const detail::ProcessTempDir dir;
    return dir.path();
}

} // namespace mko::test
