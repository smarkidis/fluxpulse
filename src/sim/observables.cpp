// src/sim/observables.cpp
#include "fluxpulse/sim/observables.hpp"

namespace fluxpulse {
namespace {

inline Complex cplx(Real re, Real im) { return Complex{re, im}; }

inline Complex conjc(const Complex& z) { return cplx(z.re, -z.im); }

inline Complex cadd(const Complex& a, const Complex& b) {
  return cplx(a.re + b.re, a.im + b.im);
}

inline Complex cmul(const Complex& a, const Complex& b) {
  return cplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

}  // namespace

Status apply_observable(const DenseOp& O, Span<const Complex> psi, Span<Complex> tmp) {
  const Dim n = O.dim();
  if (psi.size != n || tmp.size != n) {
    return Error(ErrorCode::kInvalidArgument, "apply_observable: dimension mismatch");
  }

  const Dim lda = O.lda();
  const Complex* A = O.data();

  // tmp = O * psi (column-major)
  for (Dim row = 0; row < n; ++row) {
    Complex acc = cplx(Real{0}, Real{0});
    for (Dim col = 0; col < n; ++col) {
      acc = cadd(acc, cmul(A[row + col * lda], psi[col]));
    }
    tmp[row] = acc;
  }

  return Ok();
}

Status expect_complex(const DenseOp& O,
                      Span<const Complex> psi,
                      Span<Complex> tmp,
                      Complex* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "expect_complex: out is null");

  const Dim n = O.dim();
  if (psi.size != n || tmp.size != n) {
    return Error(ErrorCode::kInvalidArgument, "expect_complex: dimension mismatch");
  }

  FLUXPULSE_RETURN_IF_ERROR(apply_observable(O, psi, tmp));

  // <psi|O|psi> = <psi|tmp>
  Complex acc = cplx(Real{0}, Real{0});
  for (Dim i = 0; i < n; ++i) {
    acc = cadd(acc, cmul(conjc(psi[i]), tmp[i]));
  }

  *out = acc;
  return Ok();
}

Status expect_real(const DenseOp& O, Span<const Complex> psi, Span<Complex> tmp, Real* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "expect_real: out is null");
  Complex v;
  FLUXPULSE_RETURN_IF_ERROR(expect_complex(O, psi, tmp, &v));
  *out = v.re;
  return Ok();
}

Status expect_many_real(Span<const DenseOp> observables,
                        Span<const Complex> psi,
                        Span<Complex> tmp,
                        Span<Real> out) {
  if (observables.size != out.size) {
    return Error(ErrorCode::kInvalidArgument, "expect_many_real: size mismatch");
  }

  for (Dim k = 0; k < observables.size; ++k) {
    Real v = Real{0};
    FLUXPULSE_RETURN_IF_ERROR(expect_real(observables[k], psi, tmp, &v));
    out[k] = v;
  }
  return Ok();
}

}  // namespace fluxpulse