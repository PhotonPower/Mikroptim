#pragma once

#include "mikroptim/config.hpp"

#include <math.h>

namespace mko {

// Wurzel für alle unterstützten Zahlentypen. Weitere Typen (z. B. df64 in M7)
// stellen eine eigene Überladung von msqrt bereit.
MKO_HD inline double msqrt(double x) { return ::sqrt(x); }
MKO_HD inline float msqrt(float x) { return ::sqrtf(x); }

} // namespace mko
