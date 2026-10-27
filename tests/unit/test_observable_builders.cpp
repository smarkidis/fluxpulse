// tests/unit/test_observable_builders.cpp
#include "fluxpulse/sim/observable_builders.hpp"

#include <cassert>

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline void expect_complex_near(const fluxpulse::Complex& a,
                                fluxpulse::Real re,
                                fluxpulse::Real im,
                                fluxpulse::Real eps) {
  expect_near(a.re, re, eps);
  expect_near(a.im, im, eps);
}

inline const fluxpulse::Complex& at(const fluxpulse::DenseOp& A, int row, int col) {
  return A.data()[row + col * A.lda()];
}

}  // namespace

int main() {
  using fluxpulse::DenseOp;
  using fluxpulse::Dim;
  using fluxpulse::Real;

  constexpr Real kEps = Real{1e-12};

  // ---------------------------------------------------------------------------
  // 2-qubit dims: [2,2], little-endian basis: |q1 q0> = [|00>,|01>,|10>,|11>]
  // ---------------------------------------------------------------------------
  const std::vector<Dim> q2_dims{2, 2};

  // Z0: diag(+,-,+,-)
  {
    DenseOp Z0;
    fluxpulse::Status st = fluxpulse::sim::build_observable("Z0", q2_dims, &Z0);
    assert(st.ok());
    assert(Z0.dim() == 4);

    expect_complex_near(at(Z0, 0, 0), Real{+1}, Real{0}, kEps);
    expect_complex_near(at(Z0, 1, 1), Real{-1}, Real{0}, kEps);
    expect_complex_near(at(Z0, 2, 2), Real{+1}, Real{0}, kEps);
    expect_complex_near(at(Z0, 3, 3), Real{-1}, Real{0}, kEps);
  }

  // Z1: diag(+,+,-,-)
  {
    DenseOp Z1;
    fluxpulse::Status st = fluxpulse::sim::build_observable("Z1", q2_dims, &Z1);
    assert(st.ok());
    assert(Z1.dim() == 4);

    expect_complex_near(at(Z1, 0, 0), Real{+1}, Real{0}, kEps);
    expect_complex_near(at(Z1, 1, 1), Real{+1}, Real{0}, kEps);
    expect_complex_near(at(Z1, 2, 2), Real{-1}, Real{0}, kEps);
    expect_complex_near(at(Z1, 3, 3), Real{-1}, Real{0}, kEps);
  }

  // X0: couples |00><01| and |10><11|
  {
    DenseOp X0;
    fluxpulse::Status st = fluxpulse::sim::build_observable("X0", q2_dims, &X0);
    assert(st.ok());
    assert(X0.dim() == 4);

    expect_complex_near(at(X0, 0, 1), Real{1}, Real{0}, kEps);
    expect_complex_near(at(X0, 1, 0), Real{1}, Real{0}, kEps);
    expect_complex_near(at(X0, 2, 3), Real{1}, Real{0}, kEps);
    expect_complex_near(at(X0, 3, 2), Real{1}, Real{0}, kEps);
  }

  // X1: couples |00><10| and |01><11|
  {
    DenseOp X1;
    fluxpulse::Status st = fluxpulse::sim::build_observable("X1", q2_dims, &X1);
    assert(st.ok());
    assert(X1.dim() == 4);

    expect_complex_near(at(X1, 0, 2), Real{1}, Real{0}, kEps);
    expect_complex_near(at(X1, 2, 0), Real{1}, Real{0}, kEps);
    expect_complex_near(at(X1, 1, 3), Real{1}, Real{0}, kEps);
    expect_complex_near(at(X1, 3, 1), Real{1}, Real{0}, kEps);
  }

  // ZZ01: diag(+,-,-,+)
  {
    DenseOp ZZ;
    fluxpulse::Status st = fluxpulse::sim::build_observable("ZZ01", q2_dims, &ZZ);
    assert(st.ok());
    assert(ZZ.dim() == 4);

    expect_complex_near(at(ZZ, 0, 0), Real{+1}, Real{0}, kEps);
    expect_complex_near(at(ZZ, 1, 1), Real{-1}, Real{0}, kEps);
    expect_complex_near(at(ZZ, 2, 2), Real{-1}, Real{0}, kEps);
    expect_complex_near(at(ZZ, 3, 3), Real{+1}, Real{0}, kEps);
  }

  // Projector P:10 => |10> (big-endian) => index 2 in little-endian ordering.
  {
    DenseOp P;
    fluxpulse::Status st = fluxpulse::sim::build_observable("P:10", q2_dims, &P);
    assert(st.ok());
    assert(P.dim() == 4);

    expect_complex_near(at(P, 2, 2), Real{1}, Real{0}, kEps);
    // A few zeros
    expect_complex_near(at(P, 0, 0), Real{0}, Real{0}, kEps);
    expect_complex_near(at(P, 1, 1), Real{0}, Real{0}, kEps);
    expect_complex_near(at(P, 3, 3), Real{0}, Real{0}, kEps);
  }

  // Projector P1 (qubit 0): selects states with q0=1 => indices 1 and 3.
  {
    DenseOp P1;
    fluxpulse::Status st = fluxpulse::sim::build_observable("P1", q2_dims, &P1);
    assert(st.ok());
    assert(P1.dim() == 4);

    expect_complex_near(at(P1, 1, 1), Real{1}, Real{0}, kEps);
    expect_complex_near(at(P1, 3, 3), Real{1}, Real{0}, kEps);
    expect_complex_near(at(P1, 0, 0), Real{0}, Real{0}, kEps);
    expect_complex_near(at(P1, 2, 2), Real{0}, Real{0}, kEps);
  }

  // Projector P0_1 (qubit 1): selects states with q1=0 => indices 0 and 1.
  {
    DenseOp P0q1;
    fluxpulse::Status st = fluxpulse::sim::build_observable("P0_1", q2_dims, &P0q1);
    assert(st.ok());
    assert(P0q1.dim() == 4);

    expect_complex_near(at(P0q1, 0, 0), Real{1}, Real{0}, kEps);
    expect_complex_near(at(P0q1, 1, 1), Real{1}, Real{0}, kEps);
    expect_complex_near(at(P0q1, 2, 2), Real{0}, Real{0}, kEps);
    expect_complex_near(at(P0q1, 3, 3), Real{0}, Real{0}, kEps);
  }

  // Identity
  {
    DenseOp I;
    fluxpulse::Status st = fluxpulse::sim::build_observable("I", q2_dims, &I);
    assert(st.ok());
    assert(I.dim() == 4);

    expect_complex_near(at(I, 0, 0), Real{1}, Real{0}, kEps);
    expect_complex_near(at(I, 1, 1), Real{1}, Real{0}, kEps);
    expect_complex_near(at(I, 2, 2), Real{1}, Real{0}, kEps);
    expect_complex_near(at(I, 3, 3), Real{1}, Real{0}, kEps);
    expect_complex_near(at(I, 0, 1), Real{0}, Real{0}, kEps);
  }

  // 'X' short form must fail for multi-qubit dims
  {
    DenseOp X;
    fluxpulse::Status st = fluxpulse::sim::build_observable("X", q2_dims, &X);
    assert(!st.ok());
  }

  // Unknown observable must fail
  {
    DenseOp A;
    fluxpulse::Status st = fluxpulse::sim::build_observable("bad_obs_name", q2_dims, &A);
    assert(!st.ok());
  }

  // ---------------------------------------------------------------------------
  // Single-mode dims: N operator
  // ---------------------------------------------------------------------------
  {
    const std::vector<Dim> d{5};
    DenseOp N;
    fluxpulse::Status st = fluxpulse::sim::build_observable("N", d, &N);
    assert(st.ok());
    assert(N.dim() == 5);

    for (int i = 0; i < 5; ++i) {
      expect_complex_near(at(N, i, i), static_cast<Real>(i), Real{0}, kEps);
    }
    expect_complex_near(at(N, 0, 1), Real{0}, Real{0}, kEps);
  }

  return 0;
}
