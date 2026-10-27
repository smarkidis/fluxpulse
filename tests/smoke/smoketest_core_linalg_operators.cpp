// tests/smoketest_core_linalg_operators.cpp
//
// Smoke test for FluxPulse core + linalg + operators.
//
// No unit test framework required. Uses assert().
//
// Validates:
// - core: types/span/rng/timer
// - math: real_math/complex
// - linalg: vec/op/dense_op + blas_gemv wrapper
// - operators: pauli/ladder/tensor (kron + embedding, little-endian convention)
//
// Notes:
// - This test avoids relying on implicit Span<T> -> Span<const T> conversions by explicitly
//   creating const spans via make_span<const T>(...).

#include <cassert>
#include <cstdio>
#include <cstdint>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/rng.hpp"
#include "fluxpulse/core/timer.hpp"

#include "fluxpulse/math/real_math.hpp"
#include "fluxpulse/math/complex.hpp"

#include "fluxpulse/linalg/vec.hpp"
#include "fluxpulse/linalg/op.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/linalg/blas_gemv.hpp"

#include "fluxpulse/operators/pauli.hpp"
#include "fluxpulse/operators/ladder.hpp"
#include "fluxpulse/operators/tensor.hpp"

using namespace fluxpulse;

static inline bool near(Real a, Real b, Real tol) noexcept {
  const Real d = (a > b) ? (a - b) : (b - a);
  return d <= tol;
}

static inline bool complex_near(Complex a, Complex b, Real tol) noexcept {
  return near(a.re, b.re, tol) && near(a.im, b.im, tol);
}

static void test_span_basic() {
  int arr[4] = {10, 20, 30, 40};
  Span<int> s = make_span(arr, 4);
  assert(s.size == 4);
  assert(!s.empty());
  assert(s[0] == 10);
  assert(s[3] == 40);
  s[2] = 99;
  assert(arr[2] == 99);

  Span<const int> cs = make_span<const int>(arr, 4);
  assert(cs[2] == 99);
}

static void test_real_math() {
  assert(near(real_sqrt(4.0), 2.0, 1e-12));
  assert(near(real_abs(-3.5), 3.5, 1e-12));
  assert(near(real_exp(0.0), 1.0, 1e-12));
  assert(near(real_log(1.0), 0.0, 1e-12));
  assert(near(real_sin(0.0), 0.0, 1e-12));
  assert(near(real_cos(0.0), 1.0, 1e-12));
}

static void test_complex_basic() {
  const Complex a{2.0, 3.0};
  const Complex b{-1.0, 0.5};

  const Complex s = add(a, b);
  assert(complex_near(s, Complex{1.0, 3.5}, 1e-12));

  const Complex p = mul(a, b); // (2+3i)*(-1+0.5i) = -3.5 - 2i
  assert(complex_near(p, Complex{-3.5, -2.0}, 1e-12));

  assert(near(abs2(Complex{3.0, 4.0}), 25.0, 1e-12));
  assert(near(abs(Complex{3.0, 4.0}), 5.0, 1e-12));
}

static void test_rng_repro() {
  Rng r1; r1.init_splitmix64();
  Rng r2; r2.init_splitmix64();

  const Seed base = 1234567ULL;
  const TrajectoryId tid = 42ULL;

  const Seed s = derive_trajectory_seed(base, tid);
  r1.seed(s);
  r2.seed(s);

  for (int i = 0; i < 10; ++i) {
    const std::uint64_t a = r1.next_u64();
    const std::uint64_t b = r2.next_u64();
    assert(a == b);
  }

  for (int i = 0; i < 1000; ++i) {
    const Real u = r1.uniform01();
    assert(u >= 0.0);
    assert(u < 1.0);
  }
}

struct DummyState { std::uint64_t x; };
static void dummy_seed(void* ctx, Seed s) noexcept {
  static_cast<DummyState*>(ctx)->x = static_cast<std::uint64_t>(s);
}
static std::uint64_t dummy_next_u64(void* ctx) noexcept {
  DummyState* st = static_cast<DummyState*>(ctx);
  st->x = st->x * 6364136223846793005ULL + 1ULL;
  return st->x;
}

static void test_rng_external_backend() {
  DummyState st{0u};
  const RngBackend backend{&dummy_seed, &dummy_next_u64};

  Rng r;
  r.init_external(&backend, &st);
  assert(r.is_initialized());

  r.seed(7);
  const std::uint64_t a = r.next_u64();
  r.seed(7);
  const std::uint64_t b = r.next_u64();
  assert(a == b);
}

static void test_timer_basic() {
  Timer t;
  t.reset();
  volatile int sink = 0;
  for (int i = 0; i < 100000; ++i) sink = sink + i; // avoid deprecated volatile compound assignment
  const Real s = t.elapsed_seconds();
  const std::uint64_t ns = t.elapsed_ns();
  assert(s >= 0.0);
  assert(ns > 0u);
  (void)sink;
}

static void test_vec_kernels() {
  Vec a(4), b(4), out(4);

  a.set_zero();
  b.set_zero();
  a[0] = Complex{1.0, 0.0};
  a[1] = Complex{0.0, 1.0};
  a[2] = Complex{2.0, -1.0};
  a[3] = Complex{-3.0, 0.5};

  ConstVecView a_c = make_span<const Complex>(a.data(), a.size());

  vec_copy(b.view(), a_c);
  for (Index i = 0; i < 4; ++i) assert(complex_near(a[i], b[i], 1e-12));

  // out = 2*a + (-i)*b   (implemented via axpy)
  vec_set(out.view(), complex_zero());
  ConstVecView b_c = make_span<const Complex>(b.data(), b.size());
  vec_axpy_inplace(out.view(), Complex{2.0, 0.0}, a_c);
  vec_axpy_inplace(out.view(), Complex{0.0, -1.0}, b_c);

  for (Index i = 0; i < 4; ++i) {
    const Complex expected = add(mul(Complex{2.0, 0.0}, a[i]),
                                 mul(Complex{0.0, -1.0}, b[i]));
    assert(complex_near(out[i], expected, 1e-12));
  }

  // dotc(a,a) should be purely real = norm2
  const Complex d = vec_dotc(a_c, a_c);
  const Real n2 = vec_norm2(a_c);
  assert(near(d.re, n2, 1e-10));
  assert(near(d.im, 0.0, 1e-10));

  // normalize
  Vec c(4);
  vec_copy(c.view(), a_c);
  const Real n = vec_normalize_inplace(c.view());
  assert(n > 0.0);
  ConstVecView c_c = make_span<const Complex>(c.data(), c.size());
  const Real n2_after = vec_norm2(c_c);
  assert(near(n2_after, 1.0, 1e-10));
}

static void test_dense_op_basic_apply() {
  DenseOp A(2);
  A.set_zero();
  A.set(0,0, Complex{1.0, 0.0});
  A.set(0,1, Complex{2.0, 0.0});
  A.set(1,0, Complex{3.0, 0.0});
  A.set(1,1, Complex{4.0, 0.0});

  // Verify column-major storage location sanity: data()[col*lda + row]
  const Complex* d = A.data();
  assert(complex_near(d[0], Complex{1.0,0.0}, 1e-12)); // (0,0)
  assert(complex_near(d[1], Complex{3.0,0.0}, 1e-12)); // (1,0)
  assert(complex_near(d[2], Complex{2.0,0.0}, 1e-12)); // (0,1)
  assert(complex_near(d[3], Complex{4.0,0.0}, 1e-12)); // (1,1)

  Vec x(2), y(2);
  x[0] = Complex{1.0, 0.0};
  x[1] = Complex{1.0, 0.0};
  y.set_zero();

  ConstVecView x_c = make_span<const Complex>(x.data(), x.size());

  // y = A*x = [3,7]
  A.apply(y.view(), x_c);
  assert(complex_near(y[0], Complex{3.0, 0.0}, 1e-12));
  assert(complex_near(y[1], Complex{7.0, 0.0}, 1e-12));

  // y = 2*A*x + 0.5*y => starting from y=[3,7], becomes 2.5*[3,7] = [7.5,17.5]
  A.gemv(y.view(), x_c, Complex{2.0, 0.0}, Complex{0.5, 0.0});
  assert(complex_near(y[0], Complex{7.5, 0.0}, 1e-12));
  assert(complex_near(y[1], Complex{17.5, 0.0}, 1e-12));
}

static void test_pauli_ops() {
  DenseOp X = pauli_x();
  DenseOp Z = pauli_z();
  DenseOp I = pauli_i();

  Vec ket0(2), ket1(2), out(2);
  ket0.set_zero(); ket1.set_zero();
  ket0[0] = Complex{1.0, 0.0};
  ket1[1] = Complex{1.0, 0.0};

  ConstVecView ket0_c = make_span<const Complex>(ket0.data(), ket0.size());
  ConstVecView ket1_c = make_span<const Complex>(ket1.data(), ket1.size());

  out.set_zero();
  X.apply(out.view(), ket0_c);
  assert(complex_near(out[0], Complex{0.0,0.0}, 1e-12));
  assert(complex_near(out[1], Complex{1.0,0.0}, 1e-12));

  out.set_zero();
  Z.apply(out.view(), ket1_c);
  assert(complex_near(out[1], Complex{-1.0,0.0}, 1e-12));

  out.set_zero();
  I.apply(out.view(), ket0_c);
  assert(complex_near(out[0], Complex{1.0,0.0}, 1e-12));

  DenseOp sm = sigma_minus();
  out.set_zero();
  sm.apply(out.view(), ket1_c);
  assert(complex_near(out[0], Complex{1.0,0.0}, 1e-12));
}

static void test_ladder_ops() {
  const Dim N = 5;
  DenseOp a = annihilation(N);
  DenseOp ad = creation(N);
  DenseOp num = number_op(N);

  Vec ket2(N), out(N);
  ket2.set_zero();
  ket2[2] = Complex{1.0, 0.0};

  ConstVecView ket2_c = make_span<const Complex>(ket2.data(), ket2.size());

  out.set_zero();
  a.apply(out.view(), ket2_c);
  assert(complex_near(out[1], Complex{real_sqrt(2.0), 0.0}, 1e-12));
  for (Dim i = 0; i < N; ++i) if (i != 1) assert(complex_near(out[i], complex_zero(), 1e-12));

  out.set_zero();
  ad.apply(out.view(), ket2_c);
  assert(complex_near(out[3], Complex{real_sqrt(3.0), 0.0}, 1e-12));
  for (Dim i = 0; i < N; ++i) if (i != 3) assert(complex_near(out[i], complex_zero(), 1e-12));

  out.set_zero();
  num.apply(out.view(), ket2_c);
  assert(complex_near(out[2], Complex{2.0, 0.0}, 1e-12));
  for (Dim i = 0; i < N; ++i) if (i != 2) assert(complex_near(out[i], complex_zero(), 1e-12));
}

static void test_tensor_kron_and_embed_qubits() {
  // Little-endian: |q1 q0> indices: |00>=0, |01>=1, |10>=2, |11>=3
  const DenseOp X = pauli_x();
  const DenseOp I = pauli_i();

  // kron(I, X) flips q0; kron(X, I) flips q1.
  DenseOp Ix = kron(I, X);
  DenseOp Xi = kron(X, I);

  Vec ket00(4), out(4);
  ket00.set_zero();
  ket00[0] = Complex{1.0, 0.0};

  ConstVecView ket00_c = make_span<const Complex>(ket00.data(), ket00.size());

  out.set_zero();
  Ix.apply(out.view(), ket00_c);
  assert(complex_near(out[1], Complex{1.0,0.0}, 1e-12));
  for (Dim i = 0; i < 4; ++i) if (i != 1) assert(complex_near(out[i], complex_zero(), 1e-12));

  out.set_zero();
  Xi.apply(out.view(), ket00_c);
  assert(complex_near(out[2], Complex{1.0,0.0}, 1e-12));
  for (Dim i = 0; i < 4; ++i) if (i != 2) assert(complex_near(out[i], complex_zero(), 1e-12));

  DenseOp X0 = embed_qubit_op(X, 0, 2);
  DenseOp X1 = embed_qubit_op(X, 1, 2);

  out.set_zero();
  X0.apply(out.view(), ket00_c);
  assert(complex_near(out[1], Complex{1.0,0.0}, 1e-12));

  out.set_zero();
  X1.apply(out.view(), ket00_c);
  assert(complex_near(out[2], Complex{1.0,0.0}, 1e-12));
}

static void test_tensor_embed_local_nonqubit() {
  std::vector<Dim> dims{2, 3, 2};
  const Dim D = 2 * 3 * 2;

  DenseOp L(3);
  L.set_zero();
  L.set(0,0, Complex{10.0,0.0});
  L.set(1,1, Complex{20.0,0.0});
  L.set(2,2, Complex{30.0,0.0});

  DenseOp F = embed_single(L, dims, 1);
  assert(F.dim() == D);

  Vec psi(D), out(D);
  psi.set_zero();
  // digits (d0,d1,d2) = (1,2,0) -> index = 1 + 2*2 + 6*0 = 5
  psi[5] = Complex{1.0,0.0};

  ConstVecView psi_c = make_span<const Complex>(psi.data(), psi.size());

  out.set_zero();
  F.apply(out.view(), psi_c);

  assert(complex_near(out[5], Complex{30.0,0.0}, 1e-12));
  for (Dim i = 0; i < D; ++i) if (i != 5) assert(complex_near(out[i], complex_zero(), 1e-12));
}

int main() {
  std::printf("FluxPulse smoketest_core_linalg_operators: starting...\n");

  test_span_basic();
  test_real_math();
  test_complex_basic();
  test_rng_repro();
  test_rng_external_backend();
  test_timer_basic();

  test_vec_kernels();
  test_dense_op_basic_apply();

  test_pauli_ops();
  test_ladder_ops();
  test_tensor_kron_and_embed_qubits();
  test_tensor_embed_local_nonqubit();

  std::printf("FluxPulse smoketest_core_linalg_operators: ALL TESTS PASSED.\n");
  return 0;
}
