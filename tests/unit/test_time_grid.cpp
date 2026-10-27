// tests/unit/test_time_grid.cpp
#include "fluxpulse/control/time_grid.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

}  // namespace

int main() {
  using fluxpulse::Dim;
  using fluxpulse::Real;
  using fluxpulse::TimeGrid;

  constexpr Real kEps = Real{1e-12};

  // ---- Uniform grid: basic properties ----
  {
    const Real t0 = Real{1.5};
    const Real dt = Real{0.1};
    const Dim n = 6;

    TimeGrid g = TimeGrid::Uniform(t0, dt, n);

    // validate
    {
      const fluxpulse::Status st = g.validate();
      assert(st.ok());
    }

    assert(g.is_uniform());
    assert(!g.is_explicit());
    assert(g.size() == n);
    assert(g.num_intervals() == (n - 1));

    expect_near(g.t(0), Real{1.5}, kEps);
    expect_near(g.t(1), Real{1.6}, kEps);
    expect_near(g.t(5), Real{2.0}, kEps);

    for (Dim i = 0; i + 1 < g.size(); ++i) {
      expect_near(g.dt_interval(i), dt, kEps);
    }

    expect_near(g.t_begin(), Real{1.5}, kEps);
    expect_near(g.t_end(), Real{2.0}, kEps);
    expect_near(g.duration(), Real{0.5}, kEps);

    // explicit list should be empty for uniform grids
    {
      const auto s = g.explicit_tlist();
      assert(s.size == 0);
      assert(s.data == nullptr);
    }
  }

  // ---- Uniform grid: invalid cases ----
  {
    TimeGrid g = TimeGrid::Uniform(Real{0}, Real{0}, 10);
    const fluxpulse::Status st = g.validate();
    assert(!st.ok());
  }
  {
    TimeGrid g = TimeGrid::Uniform(Real{0}, Real{0.1}, 0);
    const fluxpulse::Status st = g.validate();
    assert(!st.ok());
  }

  // ---- Explicit grid: basic properties ----
  {
    std::vector<Real> tlist;
    tlist.push_back(Real{0.0});
    tlist.push_back(Real{0.1});
    tlist.push_back(Real{0.4});
    tlist.push_back(Real{1.0});

    TimeGrid g = TimeGrid::Explicit(std::move(tlist));

    // validate
    {
      const fluxpulse::Status st = g.validate();
      assert(st.ok());
    }

    assert(!g.is_uniform());
    assert(g.is_explicit());
    assert(g.size() == 4);
    assert(g.num_intervals() == 3);

    expect_near(g.t(0), Real{0.0}, kEps);
    expect_near(g.t(1), Real{0.1}, kEps);
    expect_near(g.t(2), Real{0.4}, kEps);
    expect_near(g.t(3), Real{1.0}, kEps);

    expect_near(g.dt_interval(0), Real{0.1}, kEps);
    expect_near(g.dt_interval(1), Real{0.3}, kEps);
    expect_near(g.dt_interval(2), Real{0.6}, kEps);

    expect_near(g.t_begin(), Real{0.0}, kEps);
    expect_near(g.t_end(), Real{1.0}, kEps);
    expect_near(g.duration(), Real{1.0}, kEps);

    // explicit list should match original content
    {
      const auto s = g.explicit_tlist();
      assert(s.size == 4);
      assert(s.data != nullptr);
      expect_near(s[0], Real{0.0}, kEps);
      expect_near(s[1], Real{0.1}, kEps);
      expect_near(s[2], Real{0.4}, kEps);
      expect_near(s[3], Real{1.0}, kEps);
    }
  }

  // ---- Explicit grid: invalid cases ----
  {
    // empty tlist
    TimeGrid g = TimeGrid::Explicit(std::vector<Real>{});
    const fluxpulse::Status st = g.validate();
    assert(!st.ok());
  }
  {
    // not strictly increasing (equal)
    TimeGrid g = TimeGrid::Explicit(std::vector<Real>{Real{0.0}, Real{0.1}, Real{0.1}});
    const fluxpulse::Status st = g.validate();
    assert(!st.ok());
  }
  {
    // not strictly increasing (decreasing)
    TimeGrid g = TimeGrid::Explicit(std::vector<Real>{Real{0.0}, Real{0.2}, Real{0.1}});
    const fluxpulse::Status st = g.validate();
    assert(!st.ok());
  }

  {
    const auto g = TimeGrid::Uniform(Real{0}, Real{0.1}, -1);
    assert(!g.validate().ok());
  }

  return 0;
}
