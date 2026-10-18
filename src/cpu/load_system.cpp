// Laden eines Systems zum Tracen: Systemdatei, Glaskataloge, Brechzahl- und Flächentabelle.

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"

#include <string_view>
#include <system_error>
#include <utility>

namespace mko {

namespace {

LoadedSystem prepare(SystemDescription system, const std::filesystem::path& base_dir,
                     std::vector<std::string> warnings) {
    LoadedSystem out;
    out.warnings = std::move(warnings);
    out.catalog_files = resolve_glass_catalogs(system, base_dir);
    for (const std::filesystem::path& file : out.catalog_files) {
        const GlassCatalog cat = GlassCatalog::from_file(file);
        for (const std::string& w : cat.warnings()) {
            out.warnings.push_back(w);
        }
        out.catalog.merge(cat);
    }
    validate_materials(system,
                       [&out](std::string_view name) { return out.catalog.contains(name); });

    out.index = make_system_index_table(system, out.catalog);
    out.table = make_surface_table(system, out.index);
    out.system = std::move(system);
    return out;
}

} // namespace

LoadedSystem load_system(const std::filesystem::path& file) {
    std::vector<std::string> warnings;
    SystemDescription system = read_system_file(file, &warnings);
    std::error_code ec;
    std::filesystem::path base = std::filesystem::absolute(file, ec);
    if (ec) {
        base = file;
    }
    return prepare(std::move(system), base.parent_path(), std::move(warnings));
}

LoadedSystem prepare_system(SystemDescription system, const std::filesystem::path& base_dir) {
    validate_system(system);
    return prepare(std::move(system), base_dir, {});
}

} // namespace mko
