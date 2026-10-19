// Variantengenerator: gestörte Varianten und Muster der finiten Differenzen
// (docs/optimierung.md, „Variantengenerator“).

#include "mikroptim/optim/variant_generator.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Variantengenerator: " + message);
}

// x₀ hat Länge P, ist endlich und zulässig.
void check_base_vector(const VariableSet& set, std::span<const double> x0) {
    if (static_cast<int>(x0.size()) != set.size()) {
        invalid("x0 hat Länge " + std::to_string(x0.size()) + ", erwartet " +
                std::to_string(set.size()));
    }
    for (int k = 0; k < set.size(); ++k) {
        const double x = x0[static_cast<std::size_t>(k)];
        if (!std::isfinite(x)) {
            invalid("x0 von " + set[k].name + " ist nicht endlich");
        }
        if (!(x >= set[k].lower && x <= set[k].upper)) {
            invalid("x0 von " + set[k].name + " liegt außerhalb der Schranken");
        }
    }
}

void check_scale(const PerturbationScale& s, const char* what) {
    if (!(std::isfinite(s.relative) && s.relative >= 0.0 && std::isfinite(s.absolute) &&
          s.absolute >= 0.0)) {
        invalid(std::string("Störungsgröße für ") + what + " muss endlich und >= 0 sein");
    }
}

// Vorlage als eigener Batch mit einer Variante, damit `out` auch `base` sein darf.
SystemBatchHost template_of(const SystemBatchHost& base, int base_variant) {
    SystemBatchHost t(base.topology(), 1);
    t.copy_variant_from(base, base_variant, 0);
    return t;
}

void check_range(const SystemBatchHost& out, int first, int count) {
    if (first < 0 || count < 0 || first > out.variants() - count) {
        throw std::out_of_range("Variantengenerator: Varianten [" + std::to_string(first) + ", " +
                                std::to_string(first) + " + " + std::to_string(count) +
                                ") passen nicht in einen Batch mit " +
                                std::to_string(out.variants()) + " Varianten");
    }
}

} // namespace

// ---- Zufallszahlen ----------------------------------------------------------------------

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

std::uint64_t variant_seed(std::uint64_t seed, std::uint64_t j) {
    return splitmix64(seed + j * 0x9E3779B97F4A7C15ULL);
}

double unit_uniform(std::uint64_t bits) {
    return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

const PerturbationScale& PerturbationOptions::scale(VariableType type) const {
    switch (type) {
    case VariableType::Curvature:
        return curvature;
    case VariableType::Thickness:
        return thickness;
    case VariableType::WorkingDistance:
        return working_distance;
    case VariableType::GhostCoefficient:
        return ghost_coefficient;
    case VariableType::Glass:
        return glass;
    }
    return curvature;
}

std::vector<double> perturbed_vector(const VariableSet& set, std::span<const double> x0,
                                     const PerturbationOptions& options, std::uint64_t j) {
    check_base_vector(set, x0);
    check_scale(options.curvature, "curvature");
    check_scale(options.thickness, "thickness");
    check_scale(options.working_distance, "working_distance");
    check_scale(options.ghost_coefficient, "ghost_coefficient");
    check_scale(options.glass, "glass");

    std::vector<double> y(x0.begin(), x0.end());
    if (options.first_is_base && j == 0) {
        return y;
    }
    std::mt19937_64 engine(variant_seed(options.seed, j));
    for (int k = 0; k < set.size(); ++k) {
        double xi = 0.0;
        if (options.distribution == Distribution::Uniform) {
            xi = 2.0 * unit_uniform(engine()) - 1.0; // exakt: 2u ist exakt, 2u − 1 auch
        } else {
            double sum = 0.0;
            for (int i = 0; i < 12; ++i) {
                sum += unit_uniform(engine());
            }
            xi = sum - 6.0;
        }
        const std::size_t kk = static_cast<std::size_t>(k);
        const PerturbationScale& sc = options.scale(set[k].type);
        // fma explizit, damit das Ergebnis nicht von der Kontraktion des Compilers abhängt
        const double s = std::fma(sc.relative, std::abs(x0[kk]), sc.absolute);
        y[kk] = std::fma(s, xi, x0[kk]);
    }
    set.clamp_to_bounds(y);
    return y;
}

void write_perturbed_variants(const VariableSet& set, const SystemBatchHost& base, int base_variant,
                              std::span<const double> x0, const PerturbationOptions& options,
                              SystemBatchHost& out, int first, int count) {
    check_range(out, first, count);
    check_base_vector(set, x0);
    const SystemBatchHost tmpl = template_of(base, base_variant);
    for (int j = 0; j < count; ++j) {
        const std::vector<double> y =
            perturbed_vector(set, x0, options, static_cast<std::uint64_t>(j));
        out.copy_variant_from(tmpl, 0, first + j);
        set.apply(out, first + j, y);
    }
}

SystemBatchHost make_perturbed_batch(const VariableSet& set, const SystemBatchHost& base,
                                     int base_variant, std::span<const double> x0, int variants,
                                     const PerturbationOptions& options) {
    if (variants < 0) {
        invalid("negative Variantenzahl");
    }
    SystemBatchHost out = base; // gemeinsame Felder aus `base`
    out.resize(variants);
    write_perturbed_variants(set, base, base_variant, x0, options, out, 0, variants);
    return out;
}

// ---- Finite Differenzen ------------------------------------------------------------------

double fd_step(double x, double typical) {
    return kFdStepFactor * std::max(std::abs(x), typical);
}

int fd_variant_count(FdScheme scheme, int variables) {
    if (variables < 0) {
        invalid("negative Variablenzahl");
    }
    return scheme == FdScheme::Central ? 2 * variables + 1 : variables + 1;
}

int fd_plus_variant(FdScheme scheme, int k) {
    return scheme == FdScheme::Central ? 2 * k + 1 : k + 1;
}

int fd_minus_variant(int k) {
    return 2 * k + 2;
}

FdPattern fd_pattern(const VariableSet& set, std::span<const double> x0, FdScheme scheme) {
    check_base_vector(set, x0);
    const std::size_t p = x0.size();
    FdPattern pat;
    pat.scheme = scheme;
    pat.nominal.resize(p);
    pat.point_plus.resize(p);
    pat.step_plus.resize(p);
    pat.stencil.resize(p);
    if (scheme == FdScheme::Central) {
        pat.point_minus.resize(p);
        pat.step_minus.resize(p);
    }
    for (std::size_t k = 0; k < p; ++k) {
        const Variable& var = set[static_cast<int>(k)];
        const double x = x0[k];
        const double h = fd_step(x, var.typical);
        pat.nominal[k] = h;
        const double up = x + h;
        const double down = x - h;
        if (scheme == FdScheme::Central) {
            double xa = 0.0;
            double xb = 0.0;
            FdStencil stencil = FdStencil::Central;
            if (up <= var.upper && down >= var.lower) {
                xa = up; // (1) zentral
                xb = down;
            } else {
                // (2) einseitig zur Seite mit mehr Platz, falls dort x₀ + 2σh zulässig ist
                const bool towards_upper = var.upper - x >= x - var.lower;
                const double far = towards_upper ? x + 2.0 * h : x - 2.0 * h;
                if (towards_upper ? far <= var.upper : far >= var.lower) {
                    stencil = FdStencil::OneSided;
                    xa = towards_upper ? up : down;
                    xb = far;
                } else {
                    // (3) Bereich kürzer als etwa 3h: gekürzt bis an die Schranken
                    stencil = FdStencil::Truncated;
                    xa = up <= var.upper ? up : var.upper;
                    xb = down >= var.lower ? down : var.lower;
                }
            }
            pat.stencil[k] = stencil;
            pat.point_plus[k] = xa;
            pat.point_minus[k] = xb;
            pat.step_plus[k] = xa - x;
            pat.step_minus[k] = xb - x;
        } else {
            double xp = 0.0;
            if (up <= var.upper) {
                xp = up;
            } else if (down >= var.lower) {
                xp = down;
            } else {
                xp = var.upper - x >= x - var.lower ? var.upper : var.lower;
            }
            pat.stencil[k] = FdStencil::Forward;
            pat.point_plus[k] = xp;
            pat.step_plus[k] = xp - x;
        }
    }
    return pat;
}

FdPattern write_fd_variants(const VariableSet& set, const SystemBatchHost& base, int base_variant,
                            std::span<const double> x0, FdScheme scheme, SystemBatchHost& out,
                            int first) {
    const int count = fd_variant_count(scheme, set.size());
    check_range(out, first, count);
    FdPattern pat = fd_pattern(set, x0, scheme);
    const SystemBatchHost tmpl = template_of(base, base_variant);
    std::vector<double> y(x0.begin(), x0.end());
    for (int j = 0; j < count; ++j) {
        out.copy_variant_from(tmpl, 0, first + j);
    }
    set.apply(out, first, y);
    for (int k = 0; k < set.size(); ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        y[kk] = pat.point_plus[kk];
        set.apply(out, first + fd_plus_variant(scheme, k), y);
        if (scheme == FdScheme::Central) {
            y[kk] = pat.point_minus[kk];
            set.apply(out, first + fd_minus_variant(k), y);
        }
        y[kk] = x0[kk];
    }
    return pat;
}

FdBatch make_fd_batch(const VariableSet& set, const SystemBatchHost& base, int base_variant,
                      std::span<const double> x0, FdScheme scheme) {
    FdBatch result;
    result.batch = base;
    result.batch.resize(fd_variant_count(scheme, set.size()));
    result.pattern = write_fd_variants(set, base, base_variant, x0, scheme, result.batch, 0);
    return result;
}

} // namespace mko
