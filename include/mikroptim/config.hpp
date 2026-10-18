#pragma once

// Gemeinsame Makros für Host- und Device-Code (siehe docs/adr/0005-gemeinsamer-mathe-kern.md).

#if defined(__CUDACC__)
#define MKO_HD __host__ __device__
#else
#define MKO_HD
#endif
