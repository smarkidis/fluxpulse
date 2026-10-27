// tests/smoketest_rng_timer.cpp
//
// Smoke test for RNG and Timer utilities.
//
// Validates:
// - SplitMix64 RNG reproducibility with derive_trajectory_seed()
// - uniform01 bounds and basic behavior
// - external RNG backend plumbing (RngBackend vtable)
// - Timer monotonicity and basic elapsed_* methods
//
// No unit test framework required. Uses assert().

#include <cassert>
#include <cstdio>
#include <cstdint>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/core/rng.hpp"
#include "fluxpulse/core/timer.hpp"

using namespace fluxpulse;

static void test_rng_splitmix64_repro() {
  std::printf("[rng] splitmix64 reproducibility...\n");

  Rng r1; r1.init_splitmix64();
  Rng r2; r2.init_splitmix64();

  const Seed base = 0x123456789ABCDEF0ULL;
  const TrajectoryId tid = 777ULL;

  const Seed s = derive_trajectory_seed(base, tid);
  r1.seed(s);
  r2.seed(s);

  for (int i = 0; i < 100; ++i) {
    const std::uint64_t a = r1.next_u64();
    const std::uint64_t b = r2.next_u64();
    assert(a == b);
  }
}

static void test_rng_uniform01_bounds() {
  std::printf("[rng] uniform01 bounds...\n");

  Rng r; r.init_splitmix64();
  r.seed(1234);

  for (int i = 0; i < 100000; ++i) {
    const Real u = r.uniform01();
    assert(u >= 0.0);
    assert(u < 1.0);
  }

  for (int i = 0; i < 10000; ++i) {
    const Real u = r.uniform01_open();
    assert(u > 0.0);
    assert(u < 1.0);
  }
}

static void test_rng_uniform_u64_range() {
  std::printf("[rng] uniform_u64 range + inclusivity...\n");

  Rng r; r.init_splitmix64();
  r.seed(999);

  const std::uint64_t lo = 10;
  const std::uint64_t hi = 25;

  bool saw_lo = false;
  bool saw_hi = false;

  for (int i = 0; i < 200000; ++i) {
    const std::uint64_t x = r.uniform_u64(lo, hi);
    assert(x >= lo);
    assert(x <= hi);
    if (x == lo) saw_lo = true;
    if (x == hi) saw_hi = true;
    if (saw_lo && saw_hi) break;
  }

  // Not mathematically guaranteed, but extremely likely with 2e5 draws.
  assert(saw_lo);
  assert(saw_hi);
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
  std::printf("[rng] external backend plumbing...\n");

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

  const std::uint64_t c = r.next_u64();
  assert(c != b);
}

static void test_timer_monotonic() {
  std::printf("[timer] monotonic elapsed time...\n");

  Timer t;
  t.reset();

  volatile std::uint64_t sink = 0;
  for (std::uint64_t i = 0; i < 500000ULL; ++i) sink = sink + i;

  const Real s1 = t.elapsed_seconds();
  const std::uint64_t ns1 = t.elapsed_ns();

  for (std::uint64_t i = 0; i < 500000ULL; ++i) sink = sink + i;

  const Real s2 = t.elapsed_seconds();
  const std::uint64_t ns2 = t.elapsed_ns();

  assert(s1 >= 0.0);
  assert(s2 >= s1);
  assert(ns1 > 0u);
  assert(ns2 >= ns1);

  (void)sink;
}

static void test_timer_now_ns() {
  std::printf("[timer] now_ns basic...\n");

  const std::uint64_t a = now_ns();
  volatile int sink = 0;
  for (int i = 0; i < 200000; ++i) sink = sink + i;
  const std::uint64_t b = now_ns();
  assert(b >= a);
  (void)sink;
}

int main() {
  std::printf("FluxPulse smoketest_rng_timer: starting...\n");

  test_rng_splitmix64_repro();
  test_rng_uniform01_bounds();
  test_rng_uniform_u64_range();
  test_rng_external_backend();

  test_timer_monotonic();
  test_timer_now_ns();

  std::printf("FluxPulse smoketest_rng_timer: ALL TESTS PASSED.\n");
  return 0;
}
