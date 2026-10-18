#pragma once

// Zahlentyp df64 ("double-float", M7-T1): Paar (hi, lo) aus zwei float mit nicht überlappenden
// Mantissen, Wert hi + lo, |lo| <= ulp(hi)/2. Etwa 48 Bit Genauigkeit (Einheitsrundung
// u² = 2⁻⁴⁸ ≈ 3,6e-15) bei der Rechenrate von FP32; für GeForce-Karten (RTX 3090: FP64 mit 1/64
// der FP32-Rate), ADR-0003. Header-only, MKO_HD, ohne Ausnahmen, Allokation und std-Container.
// Verfahren, Fehlerschranken, Wertebereich und Grenzen: docs/numerik.md, Abschnitt 12.
//
// Grundoperationen nach M. Joldes, J.-M. Muller, V. Popescu: Tight and rigorous error bounds for
// basic building blocks of double-word arithmetic, ACM TOMS 44 (2017), Wurzel nach V. Lefèvre,
// N. Louvet, J.-M. Muller, J. Picot, L. Rideau: Accurate calculation of Euclidean norms using
// double-word arithmetic, ACM TOMS 49 (2023). Relative Fehlerschranken (u = 2⁻²⁴):
//
//   a + b, a − b   AccurateDWPlusDW (Alg. 6)   3u² + 13u³
//   a · b          DWTimesDW3 (Alg. 12, FMA)    4u²
//   a · f (float)  DWTimesFP3 (Alg. 9, FMA)     2u²
//   a / b          DWDivDW2 (Alg. 17)           15u² + 56u³
//   √a             SQRTDWtoDW                   25/8 u²
//
// FMA-Kontraktion: Die fehlerfreien Transformationen (two_sum, fast_two_sum, two_prod) sind nur
// korrekt, wenn jede Operation einzeln gerundet wird. Im Device-Code rechnet jede Operation über
// die Intrinsics __fadd_rn, __fsub_rn, __fmul_rn, __fmaf_rn, __fdiv_rn und __fsqrt_rn, die nvcc
// nie zu einem FMA zusammenzieht (auch nicht über Funktionsgrenzen hinweg, wenn ein Operand
// selbst ein Produkt ist). Auf dem Host steht jede Operation in einer eigenen Anweisung; die
// Host-Compiler des Projekts ziehen über Anweisungen hinweg nicht zusammen (GCC im ISO-Modus
// -std=c++20: -ffp-contract=off; MSVC /fp:precise erzeugt ohne /arch:AVX2 keine FMA). Der Test
// CudaDf64.DeviceMatchesHostBitwise prüft, dass Host und Gerät bitgleich rechnen.
//
// Wertebereich: der von float. Ergebnisse über FLT_MAX ≈ 3,4e38 werden NaN (nicht ±∞, die
// fehlerfreien Transformationen bilden dann ∞ − ∞); volle Genauigkeit für |x| >= 2⁻¹⁰⁰ ≈ 7,9e-31,
// darunter wird lo subnormal (absoluter Fehler dann ≈ 1e-45). Division durch 0 ergibt NaN.
// Vergleiche sind für normierte Werte exakt; NaN verhält sich wie bei double.

#include "mikroptim/config.hpp"

#include <float.h>
#include <math.h>

#include <type_traits>

namespace mko {

namespace df64_detail {

// Einzeln gerundete float-Operationen (Rundung zur nächsten, keine Kontraktion).
MKO_HD inline float add_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
MKO_HD inline float sub_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
MKO_HD inline float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
MKO_HD inline float fma_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return ::fmaf(a, b, c);
#endif
}
MKO_HD inline float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
MKO_HD inline float sqrt_rn(float a) {
#if defined(__CUDA_ARCH__)
    return __fsqrt_rn(a);
#else
    return ::sqrtf(a);
#endif
}

// Fehlerfreie Transformationen: s + e = a + b bzw. p + e = a · b exakt (ohne Über- und
// Unterlauf).
MKO_HD inline void two_sum(float a, float b, float& s, float& e) {
    s = add_rn(a, b);
    const float bb = sub_rn(s, a);
    e = add_rn(sub_rn(a, sub_rn(s, bb)), sub_rn(b, bb));
}
// Vorbedingung |a| >= |b| (oder a = 0).
MKO_HD inline void fast_two_sum(float a, float b, float& s, float& e) {
    s = add_rn(a, b);
    e = sub_rn(b, sub_rn(s, a));
}
MKO_HD inline void two_prod(float a, float b, float& p, float& e) {
    p = mul_rn(a, b);
    e = fma_rn(a, b, -p);
}

} // namespace df64_detail

struct df64 {
    float hi;
    float lo;

    // Trivialer Standardkonstruktor (Shared Memory, trivial kopierbar); der Wert ist dann
    // unbestimmt wie bei double.
    df64() = default;

    // Aus double: hi = float(x), lo = float(x − hi) (x − hi ist in double exakt). Relativer
    // Fehler <= 2⁻⁴⁹ für 2⁻¹⁰⁰ <= |x| < FLT_MAX (darunter wird lo subnormal); |x| > FLT_MAX
    // ergibt ±∞, NaN bleibt NaN (lo = 0).
    // Im Device-Code eine FP64-Subtraktion und zwei Umwandlungen.
    MKO_HD constexpr explicit df64(double x) : hi(static_cast<float>(x)), lo(0.0f) {
        const double rest = x - static_cast<double>(hi);
        lo = rest - rest == 0.0 ? static_cast<float>(rest) : 0.0f; // rest = ±∞ oder NaN: lo = 0
    }
    MKO_HD constexpr explicit df64(float x) : hi(x), lo(0.0f) {}
    // Aus einer Ganzzahl: exakt für |i| < 2⁴⁸.
    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    MKO_HD constexpr explicit df64(I i)
        : hi(static_cast<float>(static_cast<long long>(i))),
          lo(static_cast<float>(static_cast<long long>(i) - static_cast<long long>(hi))) {}

    // Aus den beiden Teilen (Vorbedingung: normiert, |lo| <= ulp(hi)/2).
    MKO_HD static constexpr df64 from_parts(float h, float l) {
        df64 r;
        r.hi = h;
        r.lo = l;
        return r;
    }

    // Nach double (eine Rundung; exakt, wenn hi + lo in 53 Bit passt, z. B. für jedes df64 aus
    // einem double). Im Device-Code eine FP64-Addition.
    MKO_HD constexpr explicit operator double() const {
        return static_cast<double>(hi) + static_cast<double>(lo);
    }
};

static_assert(sizeof(df64) == sizeof(double), "df64 belegt 8 Byte wie double");
static_assert(alignof(df64) <= alignof(double));
static_assert(std::is_trivially_default_constructible_v<df64>);
static_assert(std::is_trivially_copyable_v<df64>);
static_assert(std::is_standard_layout_v<df64>);

// ---- Arithmetik ----

MKO_HD inline df64 operator-(df64 a) {
    return df64::from_parts(-a.hi, -a.lo);
}

// AccurateDWPlusDW (Joldes, Muller, Popescu 2017, Alg. 6): relativer Fehler <= 3u² + 13u³,
// auch bei Auslöschung (OPL − Kolben).
MKO_HD inline df64 operator+(df64 a, df64 b) {
    using namespace df64_detail;
    float sh, sl, th, tl, vh, vl, zh, zl;
    two_sum(a.hi, b.hi, sh, sl);
    two_sum(a.lo, b.lo, th, tl);
    const float c = add_rn(sl, th);
    fast_two_sum(sh, c, vh, vl);
    const float w = add_rn(tl, vl);
    fast_two_sum(vh, w, zh, zl);
    return df64::from_parts(zh, zl);
}

MKO_HD inline df64 operator-(df64 a, df64 b) {
    return a + (-b);
}

// DWTimesDW3 (Alg. 12, mit FMA): relativer Fehler <= 4u².
MKO_HD inline df64 operator*(df64 a, df64 b) {
    using namespace df64_detail;
    float ch, cl1, zh, zl;
    two_prod(a.hi, b.hi, ch, cl1);
    const float tl0 = mul_rn(a.lo, b.lo);
    const float tl1 = fma_rn(a.hi, b.lo, tl0);
    const float cl2 = fma_rn(a.lo, b.hi, tl1);
    const float cl3 = add_rn(cl1, cl2);
    fast_two_sum(ch, cl3, zh, zl);
    return df64::from_parts(zh, zl);
}

// DWTimesFP3 (Alg. 9, mit FMA): df64 mal float, relativer Fehler <= 2u².
MKO_HD inline df64 operator*(df64 a, float b) {
    using namespace df64_detail;
    float ch, cl1, zh, zl;
    two_prod(a.hi, b, ch, cl1);
    const float cl3 = fma_rn(a.lo, b, cl1);
    fast_two_sum(ch, cl3, zh, zl);
    return df64::from_parts(zh, zl);
}
MKO_HD inline df64 operator*(float a, df64 b) {
    return b * a;
}

// DWDivDW2 (Alg. 17): relativer Fehler <= 15u² + 56u³. Division durch 0 ergibt NaN (nicht ±∞
// wie double); der Tracer dividiert nur durch geprüfte Größen (d_z > 0, Nenner > 0, n > 0).
MKO_HD inline df64 operator/(df64 a, df64 b) {
    using namespace df64_detail;
    const float th = div_rn(a.hi, b.hi);
    const df64 r = b * th;
    const float pi_h = sub_rn(a.hi, r.hi);
    const float d_l = sub_rn(a.lo, r.lo);
    const float d = add_rn(pi_h, d_l);
    const float tl = div_rn(d, b.hi);
    float zh, zl;
    fast_two_sum(th, tl, zh, zl);
    return df64::from_parts(zh, zl);
}

MKO_HD inline df64& operator+=(df64& a, df64 b) {
    a = a + b;
    return a;
}
MKO_HD inline df64& operator-=(df64& a, df64 b) {
    a = a - b;
    return a;
}
MKO_HD inline df64& operator*=(df64& a, df64 b) {
    a = a * b;
    return a;
}
MKO_HD inline df64& operator/=(df64& a, df64 b) {
    a = a / b;
    return a;
}

// ---- Vergleiche (für normierte Werte exakt; NaN in hi: alle Vergleiche false außer !=) ----

MKO_HD inline bool operator==(df64 a, df64 b) {
    return a.hi == b.hi && a.lo == b.lo;
}
MKO_HD inline bool operator!=(df64 a, df64 b) {
    return !(a == b);
}
MKO_HD inline bool operator<(df64 a, df64 b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
MKO_HD inline bool operator>(df64 a, df64 b) {
    return b < a;
}
MKO_HD inline bool operator<=(df64 a, df64 b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
}
MKO_HD inline bool operator>=(df64 a, df64 b) {
    return b <= a;
}

// ---- Funktionen (Überladungen wie in scalar.hpp, ADR-0003) ----

// SQRTDWtoDW (Lefèvre, Louvet, Muller, Picot, Rideau 2023): relativer Fehler <= 25/8 u².
// Wie double: √0 = 0 (mit Vorzeichen), √(negativ) = NaN, √∞ = ∞, √NaN = NaN.
MKO_HD inline df64 msqrt(df64 a) {
    using namespace df64_detail;
    if (!(a.hi > 0.0f) || a.hi > FLT_MAX) {
        return df64::from_parts(sqrt_rn(a.hi), 0.0f);
    }
    const float sh = sqrt_rn(a.hi);
    const float rho1 = fma_rn(-sh, sh, a.hi); // a_hi − sh² (exakt)
    const float rho2 = add_rn(a.lo, rho1);
    const float sl = div_rn(rho2, add_rn(sh, sh));
    float zh, zl;
    fast_two_sum(sh, sl, zh, zl);
    return df64::from_parts(zh, zl);
}

namespace df64_detail {

// Ein Schritt des Horner-Schemas p · t + c mit dem Koeffizienten c = c_hi + c_lo.
MKO_HD inline df64 horner_step(df64 p, df64 t, float c_hi, float c_lo) {
    return p * t + df64::from_parts(c_hi, c_lo);
}

} // namespace df64_detail

// Sinus und Kosinus gemeinsam. Reduktion r = x − q·π/2 mit q = round(x·2/π) und π/2 als Summe
// dreier float (P0 + P1 + P2, Fehler < 2⁻⁷⁴), die Produkte q·P0 und q·P1 exakt (two_prod), dann
// Taylor-Polynome in t = r² für |r| <= π/4 (sin bis r¹⁵, cos bis r¹⁶; Abbruchfehler < 7e-17)
// im Horner-Schema und Zuordnung des Quadranten q mod 4. Koeffizienten 1/n! als Paar
// hi = float(1/n!), lo = float(1/n! − hi) aus exakten Brüchen (Python fractions). Absoluter
// Fehler wenige u² (gemessen <= 7,3e-15 für |x| <= 100, Test Df64.SinCosAccuracy). Für den
// Tracer (Pupillenazimut und Feldwinkel, |x| < 2π) genügt das; |x| >= 2²⁰, ±∞ und NaN ergeben
// NaN. sin 0 = 0 und cos 0 = 1 exakt (Achsstrahlen bleiben in der Meridionalebene).
// Kosten: ca. 20 df64-Multiplikationen und -Additionen pro Aufruf (collimated_ray ruft msin und
// mcos für Azimut und Feldwinkel; gegen 30 Flächen pro Strahl gering).
MKO_HD inline void msincos(df64 x, df64& s, df64& c) {
    using namespace df64_detail;
    const float ax = x.hi < 0.0f ? -x.hi : x.hi;
    if (!(ax < 1048576.0f)) {
        s = df64::from_parts(sqrt_rn(-ax - 1.0f), 0.0f); // NaN (auch für ∞ und NaN)
        c = s;
        return;
    }
    const float qf = rintf(mul_rn(x.hi, 0x1.45f306p-1f)); // round(x · 2/π)
    const int q = static_cast<int>(qf);
    // r = x − q·P0 − q·P1 − q·P2; two_prod liefert q·P0 und q·P1 exakt und normiert.
    float ph, pl;
    two_prod(qf, 0x1.921fb6p+0f, ph, pl);
    df64 r = x - df64::from_parts(ph, pl);
    two_prod(qf, -0x1.777a5cp-25f, ph, pl);
    r = r - df64::from_parts(ph, pl);
    r = r - df64(mul_rn(qf, -0x1.ee59dap-50f));
    const df64 t = r * r;
    // sin r = r · (1 − t/3! + t²/5! − … − t⁷/15!)
    df64 ps = df64::from_parts(-0x1.ae7f3ep-41f, -0x1.ccee08p-67f);
    ps = horner_step(ps, t, 0x1.612462p-33f, -0x1.8af25ep-58f);
    ps = horner_step(ps, t, -0x1.ae6456p-26f, -0x1.fd5138p-52f);
    ps = horner_step(ps, t, 0x1.71de3ap-19f, 0x1.55b1ccp-45f);
    ps = horner_step(ps, t, -0x1.a01a02p-13f, 0x1.7f97fap-39f);
    ps = horner_step(ps, t, 0x1.111112p-7f, -0x1.dddddep-32f);
    ps = horner_step(ps, t, -0x1.555556p-3f, 0x1.555556p-28f);
    ps = horner_step(ps, t, 1.0f, 0.0f);
    const df64 sin_r = r * ps;
    // cos r = 1 − t/2! + t²/4! − … + t⁸/16!
    df64 pc = df64::from_parts(0x1.ae7f3ep-45f, 0x1.ccee08p-71f);
    pc = horner_step(pc, t, -0x1.93974ap-37f, -0x1.180f94p-62f);
    pc = horner_step(pc, t, 0x1.1eed8ep-29f, 0x1.ff1b12p-54f);
    pc = horner_step(pc, t, -0x1.27e4fcp-22f, 0x1.10ec14p-47f);
    pc = horner_step(pc, t, 0x1.a01a02p-16f, -0x1.7f97fap-42f);
    pc = horner_step(pc, t, -0x1.6c16c2p-10f, 0x1.27d27ep-35f);
    pc = horner_step(pc, t, 0x1.555556p-5f, -0x1.555556p-30f);
    pc = horner_step(pc, t, -0.5f, 0.0f);
    const df64 cos_r = horner_step(pc, t, 1.0f, 0.0f);
    switch (q & 3) {
    case 0:
        s = sin_r;
        c = cos_r;
        break;
    case 1:
        s = cos_r;
        c = -sin_r;
        break;
    case 2:
        s = -sin_r;
        c = -cos_r;
        break;
    default:
        s = -cos_r;
        c = sin_r;
        break;
    }
}

MKO_HD inline df64 msin(df64 x) {
    df64 s, c;
    msincos(x, s, c);
    return s;
}
MKO_HD inline df64 mcos(df64 x) {
    df64 s, c;
    msincos(x, s, c);
    return c;
}

#if defined(__CUDACC__)
// Warp-Shuffle für df64: zwei float-Shuffles (Reduktionen im Kernel, ADR-0007).
__device__ inline df64 shfl_down_sync(unsigned mask, df64 x, unsigned delta) {
    return df64::from_parts(__shfl_down_sync(mask, x.hi, delta),
                            __shfl_down_sync(mask, x.lo, delta));
}
#endif

} // namespace mko
