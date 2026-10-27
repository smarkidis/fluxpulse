// include/fluxpulse/math/complex.hpp
#pragma once

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/real_math.hpp"

namespace fluxpulse {

struct Complex {
  Real re;
  Real im;

  FLUXPULSE_HD constexpr Complex() noexcept : re(0.0), im(0.0) {}
  FLUXPULSE_HD constexpr Complex(Real real, Real imag) noexcept : re(real), im(imag) {}
};

FLUXPULSE_HD constexpr Complex make_complex(Real re, Real im) noexcept { return Complex{re, im}; }
FLUXPULSE_HD constexpr Complex complex_zero() noexcept { return Complex{0.0, 0.0}; }
FLUXPULSE_HD constexpr Complex complex_one() noexcept { return Complex{1.0, 0.0}; }
FLUXPULSE_HD constexpr Complex complex_i() noexcept { return Complex{0.0, 1.0}; }

FLUXPULSE_HD constexpr Complex add(Complex a, Complex b) noexcept {
  return Complex{a.re + b.re, a.im + b.im};
}
FLUXPULSE_HD constexpr Complex sub(Complex a, Complex b) noexcept {
  return Complex{a.re - b.re, a.im - b.im};
}
FLUXPULSE_HD constexpr Complex neg(Complex a) noexcept { return Complex{-a.re, -a.im}; }
FLUXPULSE_HD constexpr Complex conj(Complex a) noexcept { return Complex{a.re, -a.im}; }
FLUXPULSE_HD constexpr Complex scale(Complex a, Real s) noexcept {
  return Complex{a.re * s, a.im * s};
}
FLUXPULSE_HD constexpr Complex mul(Complex a, Complex b) noexcept {
  return Complex{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

FLUXPULSE_HD constexpr Real abs2(Complex a) noexcept {
  return a.re * a.re + a.im * a.im;
}

FLUXPULSE_HD inline Real abs(Complex a) noexcept {
  return real_sqrt(abs2(a));
}

FLUXPULSE_HD inline Complex div(Complex a, Complex b) noexcept {
  const Real d = abs2(b);
  if (d <= kEpsilon) return complex_zero();
  const Complex num = mul(a, conj(b));
  return Complex{num.re / d, num.im / d};
}

FLUXPULSE_HD constexpr Complex add_scaled(Complex y, Complex x, Real a) noexcept {
  return add(y, scale(x, a));
}

FLUXPULSE_HD constexpr bool eq(Complex a, Complex b) noexcept {
  return (a.re == b.re) && (a.im == b.im);
}

}  // namespace fluxpulse