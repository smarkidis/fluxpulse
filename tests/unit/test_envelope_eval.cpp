// tests/unit/test_envelope_eval.cpp
#include "fluxpulse/control/envelope.hpp"

#include <cassert>
#include <vector>

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

}  // namespace

int main() {
  using fluxpulse::Envelope;
  using fluxpulse::Real;
  using fluxpulse::Segment;
  using fluxpulse::TimeGrid;

  constexpr Real kEps = Real{1e-12};

  // ---- Empty envelope ----
  {
    Envelope e;
    Envelope::Cursor c;
    Envelope::reset(&c);

    expect_near(e.eval(Real{0.0}), Real{0.0}, kEps);
    expect_near(e.eval_monotonic(Real{0.0}, &c), Real{0.0}, kEps);
  }

  // ---- PiecewiseConstant (ZOH) with gaps + boundary behavior ----
  {
    Envelope e;
    // Two constant segments with a gap between them:
    // [0.0, 1.0) -> 2.0
    // [2.0, 3.0) -> -1.0
    e.add(Segment::PiecewiseConstant(Real{0.0}, Real{1.0}, Real{2.0}));
    e.add(Segment::PiecewiseConstant(Real{2.0}, Real{3.0}, Real{-1.0}));

    const fluxpulse::Status st = e.finalize();
    assert(st.ok());

    // Inside first segment
    expect_near(e.eval(Real{0.0}), Real{2.0}, kEps);
    expect_near(e.eval(Real{0.5}), Real{2.0}, kEps);
    expect_near(e.eval(Real{0.999999}), Real{2.0}, Real{1e-6});

    // Boundary: t == 1.0 is excluded (half-open)
    expect_near(e.eval(Real{1.0}), Real{0.0}, kEps);

    // Gap
    expect_near(e.eval(Real{1.5}), Real{0.0}, kEps);

    // Inside second segment
    expect_near(e.eval(Real{2.0}), Real{-1.0}, kEps);
    expect_near(e.eval(Real{2.5}), Real{-1.0}, kEps);

    // Boundary: t == 3.0 excluded
    expect_near(e.eval(Real{3.0}), Real{0.0}, kEps);

    // Monotonic evaluation should match eval(t) for increasing t
    Envelope::Cursor c;
    Envelope::reset(&c);
    const Real tvals[] = {Real{-1.0}, Real{0.0}, Real{0.2}, Real{1.0}, Real{1.7},
                          Real{2.0}, Real{2.9}, Real{3.0}, Real{4.0}};
    for (Real t : tvals) {
      const Real a = e.eval(t);
      const Real b = e.eval_monotonic(t, &c);
      expect_near(a, b, kEps);
    }
  }

  // ---- PiecewiseLinear: interpolation correctness + boundaries ----
  {
    Envelope e;
    // Linear ramp [0, 2): 0 -> 1
    e.add(Segment::PiecewiseLinear(Real{0.0}, Real{2.0}, Real{0.0}, Real{1.0}));

    const fluxpulse::Status st = e.finalize();
    assert(st.ok());

    expect_near(e.eval(Real{-0.1}), Real{0.0}, kEps);
    expect_near(e.eval(Real{0.0}), Real{0.0}, kEps);
    expect_near(e.eval(Real{1.0}), Real{0.5}, kEps);
    expect_near(e.eval(Real{1.5}), Real{0.75}, kEps);
    // t == 2.0 excluded
    expect_near(e.eval(Real{2.0}), Real{0.0}, kEps);
  }

  // ---- finalize() sorts segments, and rejects overlaps ----
  {
    Envelope e;
    // Add out of order; finalize should sort and accept if non-overlapping.
    e.add(Segment::PiecewiseConstant(Real{2.0}, Real{3.0}, Real{5.0}));
    e.add(Segment::PiecewiseConstant(Real{0.0}, Real{1.0}, Real{1.0}));

    const fluxpulse::Status st = e.finalize();
    assert(st.ok());

    expect_near(e.eval(Real{0.5}), Real{1.0}, kEps);
    expect_near(e.eval(Real{2.5}), Real{5.0}, kEps);
  }
  {
    Envelope e;
    // Overlap: [0,2) and [1,3)
    e.add(Segment::PiecewiseConstant(Real{0.0}, Real{2.0}, Real{1.0}));
    e.add(Segment::PiecewiseConstant(Real{1.0}, Real{3.0}, Real{2.0}));

    const fluxpulse::Status st = e.finalize();
    assert(!st.ok());
  }

  // ---- eval_on_grid() monotonic path on uniform grid ----
  {
    Envelope e;
    // [0, 0.3) -> 1.0, [0.3, 0.6) -> -2.0
    e.add(Segment::PiecewiseConstant(Real{0.0}, Real{0.3}, Real{1.0}));
    e.add(Segment::PiecewiseConstant(Real{0.3}, Real{0.6}, Real{-2.0}));
    const fluxpulse::Status st = e.finalize();
    assert(st.ok());

    TimeGrid g = TimeGrid::Uniform(Real{0.0}, Real{0.1}, 8);  // t = 0..0.7
    assert(g.validate().ok());

    Envelope::Cursor c;
    Envelope::reset(&c);

    // t=0.0,0.1,0.2 -> 1
    expect_near(e.eval_on_grid(0, g, &c), Real{1.0}, kEps);
    expect_near(e.eval_on_grid(1, g, &c), Real{1.0}, kEps);
    expect_near(e.eval_on_grid(2, g, &c), Real{1.0}, kEps);

    // t=0.3,0.4,0.5 -> -2
    expect_near(e.eval_on_grid(3, g, &c), Real{-2.0}, kEps);
    expect_near(e.eval_on_grid(4, g, &c), Real{-2.0}, kEps);
    expect_near(e.eval_on_grid(5, g, &c), Real{-2.0}, kEps);

    // t=0.6,0.7 -> 0 (segment is [0.3,0.6))
    expect_near(e.eval_on_grid(6, g, &c), Real{0.0}, kEps);
    expect_near(e.eval_on_grid(7, g, &c), Real{0.0}, kEps);
  }

  return 0;
}