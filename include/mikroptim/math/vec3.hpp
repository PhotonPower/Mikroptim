#pragma once

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"

namespace mko {

template <class R>
struct Vec3 {
    R x, y, z;
};

template <class R>
MKO_HD constexpr Vec3<R> operator+(Vec3<R> a, Vec3<R> b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <class R>
MKO_HD constexpr Vec3<R> operator-(Vec3<R> a, Vec3<R> b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <class R>
MKO_HD constexpr Vec3<R> operator*(R s, Vec3<R> a) {
    return {s * a.x, s * a.y, s * a.z};
}

template <class R>
MKO_HD constexpr Vec3<R> operator*(Vec3<R> a, R s) {
    return s * a;
}

template <class R>
MKO_HD constexpr Vec3<R>& operator+=(Vec3<R>& a, Vec3<R> b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template <class R>
MKO_HD constexpr R dot(Vec3<R> a, Vec3<R> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <class R>
MKO_HD inline R norm(Vec3<R> a) {
    return msqrt(dot(a, a));
}

} // namespace mko
