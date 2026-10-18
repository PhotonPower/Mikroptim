#pragma once

// Fehler beim Einlesen, Validieren oder Schreiben von Dateien (docs/dateiformat.md).

#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

// Ausnahme mit JSON-Pfad der beanstandeten Stelle, z. B. `surfaces[1].material`.
// what() liefert "<datei>: <pfad>: <meldung>"; leere Teile entfallen.
class FormatError : public std::runtime_error {
public:
    FormatError(std::string json_path, std::string message, std::string file = {})
        : std::runtime_error(compose(file, json_path, message)), json_path_(std::move(json_path)),
          message_(std::move(message)), file_(std::move(file)) {}

    // JSON-Pfad, leer bei Fehlern ohne Ortsangabe (z. B. Syntaxfehler)
    const std::string& json_path() const noexcept { return json_path_; }
    // Meldung ohne Datei und Pfad
    const std::string& message() const noexcept { return message_; }
    // Datei, falls bekannt
    const std::string& file() const noexcept { return file_; }

private:
    static std::string compose(const std::string& file, const std::string& path,
                               const std::string& message) {
        std::string s;
        if (!file.empty()) {
            s += file;
            s += ": ";
        }
        if (!path.empty()) {
            s += path;
            s += ": ";
        }
        s += message;
        return s;
    }

    std::string json_path_;
    std::string message_;
    std::string file_;
};

} // namespace mko
