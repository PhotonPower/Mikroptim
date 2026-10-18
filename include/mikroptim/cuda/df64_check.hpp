#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// Prüfkernel für den Zahlentyp df64 (M7-T1): dieselben Operationen auf Host und Gerät, damit
// ein Test die Bitgleichheit prüfen kann (fehlerfreie Transformationen ohne FMA-Kontraktion,
// docs/numerik.md, Abschnitt 12).

#include "mikroptim/config.hpp"
#include "mikroptim/math/df64.hpp"

#include <vector>

namespace mko::cuda {

// Zahl der df64-Ergebnisse pro Eingabe in df64_check_ops.
inline constexpr int kDf64CheckOps = 12;

// Wendet alle geprüften Operationen auf (a, b, x) an: out[0 … kDf64CheckOps − 1] =
// a + b, a − b, a · b, a / b, √|a|, sin a, cos a, a · b.hi (df64 mal float), df64(x),
// a · b + a (zusammengesetzt: darf nicht zu einem FMA verschmelzen), √(a² + b²), a − x;
// *as_double = double(a). Gemeinsam für Host und Gerät.
MKO_HD inline void df64_check_ops(df64 a, df64 b, double x, df64* out, double* as_double) {
    out[0] = a + b;
    out[1] = a - b;
    out[2] = a * b;
    out[3] = a / b;
    out[4] = msqrt(a < df64(0) ? -a : a);
    df64 s;
    df64 c;
    msincos(a, s, c);
    out[5] = s;
    out[6] = c;
    out[7] = a * b.hi;
    out[8] = df64(x);
    out[9] = a * b + a;
    out[10] = msqrt(a * a + b * b);
    out[11] = a - df64(x);
    *as_double = static_cast<double>(a);
}

// Führt df64_check_ops für alle i < a.size() auf dem aktuellen Gerät aus (ein Thread pro
// Eingabe). a, b und x müssen gleich lang sein. Ergebnisse: out[i · kDf64CheckOps + j] und
// as_double[i]. Rückgabe false bei CUDA-Fehlern oder ungleichen Längen.
bool df64_check_ops_gpu(const std::vector<df64>& a, const std::vector<df64>& b,
                        const std::vector<double>& x, std::vector<df64>& out,
                        std::vector<double>& as_double);

} // namespace mko::cuda
