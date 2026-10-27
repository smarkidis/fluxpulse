// tests/unit/test_ensemble_accumulator.cpp
#include "fluxpulse/mpi/ensemble_accumulator.hpp"

#include <cassert>
#include <cmath>

#include "fluxpulse/core/span.hpp"
#include "fluxpulse/sim/result.hpp"

namespace {

inline void expect_near(fluxpulse::Real a,
                        fluxpulse::Real b,
                        fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

}  // namespace

int main() {
  using fluxpulse::Dim;
  using fluxpulse::Real;
  using fluxpulse::Result;
  using fluxpulse::Span;
  using fluxpulse::mpi::EnsembleAccumulator;

  constexpr Real kEps = Real{1e-12};

  // ---------------------------------------------------------------------------
  // Construct / reset / clear
  // ---------------------------------------------------------------------------
  {
    EnsembleAccumulator acc(2, 3);

    assert(acc.observable_count() == 2);
    assert(acc.sample_count() == 3);
    assert(acc.flat_size() == 6);
    assert(acc.trajectories() == 0);

    {
      const EnsembleAccumulator& cacc = acc;
      const Span<const Real> s = cacc.sum();
      const Span<const Real> s2 = cacc.sum_sq();
      assert(s.size == 6);
      assert(s2.size == 6);
      for (Dim i = 0; i < s.size; ++i) {
        expect_near(s[i], Real{0}, kEps);
        expect_near(s2[i], Real{0}, kEps);
      }
    }

    const Real x[] = {
        Real{1}, Real{2}, Real{3},
        Real{4}, Real{5}, Real{6},
    };
    acc.AccumulateTrajectory(Span<const Real>(x, 6));
    assert(acc.trajectories() == 1);

    acc.Clear();
    assert(acc.trajectories() == 0);

    {
      const EnsembleAccumulator& cacc = acc;
      const Span<const Real> s = cacc.sum();
      const Span<const Real> s2 = cacc.sum_sq();
      for (Dim i = 0; i < s.size; ++i) {
        expect_near(s[i], Real{0}, kEps);
        expect_near(s2[i], Real{0}, kEps);
      }
    }

    acc.Reset(1, 4);
    assert(acc.observable_count() == 1);
    assert(acc.sample_count() == 4);
    assert(acc.flat_size() == 4);
    assert(acc.trajectories() == 0);
  }

  // ---------------------------------------------------------------------------
  // Accumulate one trajectory
  // Layout: [obs0_t0, obs0_t1, obs1_t0, obs1_t1]
  // ---------------------------------------------------------------------------
  {
    EnsembleAccumulator acc(2, 2);

    const Real x[] = {
        Real{1}, Real{2},
        Real{3}, Real{4},
    };
    acc.AccumulateTrajectory(Span<const Real>(x, 4));

    assert(acc.trajectories() == 1);

    const EnsembleAccumulator& cacc = acc;
    const Span<const Real> s = cacc.sum();
    const Span<const Real> s2 = cacc.sum_sq();

    expect_near(s[0], Real{1}, kEps);
    expect_near(s[1], Real{2}, kEps);
    expect_near(s[2], Real{3}, kEps);
    expect_near(s[3], Real{4}, kEps);

    expect_near(s2[0], Real{1}, kEps);
    expect_near(s2[1], Real{4}, kEps);
    expect_near(s2[2], Real{9}, kEps);
    expect_near(s2[3], Real{16}, kEps);
  }

  // ---------------------------------------------------------------------------
  // Merge accumulators
  // ---------------------------------------------------------------------------
  {
    EnsembleAccumulator a(1, 3);
    EnsembleAccumulator b(1, 3);

    const Real x0[] = {Real{1}, Real{2}, Real{3}};
    const Real x1[] = {Real{2}, Real{4}, Real{6}};
    const Real x2[] = {Real{3}, Real{6}, Real{9}};

    a.AccumulateTrajectory(Span<const Real>(x0, 3));
    a.AccumulateTrajectory(Span<const Real>(x1, 3));
    b.AccumulateTrajectory(Span<const Real>(x2, 3));

    assert(a.trajectories() == 2);
    assert(b.trajectories() == 1);

    a.Merge(b);
    assert(a.trajectories() == 3);

    const EnsembleAccumulator& ca = a;
    const Span<const Real> s = ca.sum();
    const Span<const Real> s2 = ca.sum_sq();

    expect_near(s[0], Real{6}, kEps);
    expect_near(s[1], Real{12}, kEps);
    expect_near(s[2], Real{18}, kEps);

    expect_near(s2[0], Real{1 + 4 + 9}, kEps);
    expect_near(s2[1], Real{4 + 16 + 36}, kEps);
    expect_near(s2[2], Real{9 + 36 + 81}, kEps);
  }

  // ---------------------------------------------------------------------------
  // Merge via ConstLocalEnsembleStatsView
  // ---------------------------------------------------------------------------
  {
    EnsembleAccumulator a(1, 2);
    EnsembleAccumulator b(1, 2);

    const Real xa[] = {Real{1}, Real{3}};
    const Real xb[] = {Real{5}, Real{7}};

    a.AccumulateTrajectory(Span<const Real>(xa, 2));
    b.AccumulateTrajectory(Span<const Real>(xb, 2));

    const EnsembleAccumulator& cb = b;
    a.Merge(cb.view());

    assert(a.trajectories() == 2);

    const EnsembleAccumulator& ca = a;
    const Span<const Real> s = ca.sum();
    const Span<const Real> s2 = ca.sum_sq();

    expect_near(s[0], Real{6}, kEps);
    expect_near(s[1], Real{10}, kEps);

    expect_near(s2[0], Real{1 + 25}, kEps);
    expect_near(s2[1], Real{9 + 49}, kEps);
  }

  // ---------------------------------------------------------------------------
  // FinalizeInto(Result)
  //
  // Two trajectories, shape = (2 observables, 2 times)
  //
  // traj A:
  //   obs0 = [1, 3]
  //   obs1 = [2, 4]
  //
  // traj B:
  //   obs0 = [3, 5]
  //   obs1 = [4, 8]
  //
  // means:
  //   obs0 = [2, 4]
  //   obs1 = [3, 6]
  //
  // population variances:
  //   obs0 = [1, 1]
  //   obs1 = [1, 4]
  //
  // stderr = sqrt(var / N), N=2
  // ---------------------------------------------------------------------------
  {
    EnsembleAccumulator acc(2, 2);

    const Real xa[] = {
        Real{1}, Real{3},
        Real{2}, Real{4},
    };
    const Real xb[] = {
        Real{3}, Real{5},
        Real{4}, Real{8},
    };

    acc.AccumulateTrajectory(Span<const Real>(xa, 4));
    acc.AccumulateTrajectory(Span<const Real>(xb, 4));

    Result r;
    const bool ok = acc.FinalizeInto(&r);
    assert(ok);

    assert(r.num_observables() == 2);
    assert(r.num_times() == 2);

    expect_near(r.expect_at(0, 0), Real{2}, kEps);
    expect_near(r.expect_at(0, 1), Real{4}, kEps);
    expect_near(r.expect_at(1, 0), Real{3}, kEps);
    expect_near(r.expect_at(1, 1), Real{6}, kEps);

    assert(r.expectations_var.size() == 4);
    assert(r.expectations_stderr.size() == 4);

    expect_near(r.expectations_var[0], Real{1}, kEps);
    expect_near(r.expectations_var[1], Real{1}, kEps);
    expect_near(r.expectations_var[2], Real{1}, kEps);
    expect_near(r.expectations_var[3], Real{4}, kEps);

    const Real inv_sqrt2 =
        static_cast<Real>(1.0 / std::sqrt(2.0));

    expect_near(r.expectations_stderr[0], inv_sqrt2, kEps);
    expect_near(r.expectations_stderr[1], inv_sqrt2, kEps);
    expect_near(r.expectations_stderr[2], inv_sqrt2, kEps);
    expect_near(r.expectations_stderr[3], Real{2} * inv_sqrt2, kEps);
  }

  // ---------------------------------------------------------------------------
  // FinalizeInto should fail for empty ensemble
  // ---------------------------------------------------------------------------
  {
    EnsembleAccumulator acc(1, 1);
    Result r;
    const bool ok = acc.FinalizeInto(&r);
    assert(!ok);
  }

  return 0;
}