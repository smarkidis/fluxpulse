// tests/unit/test_pulse_program_coeff.cpp
#include "fluxpulse/control/pulse_program.hpp"

#include <cassert>
#include <vector>

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

}  // namespace

int main() {
  using fluxpulse::ChannelId;
  using fluxpulse::Envelope;
  using fluxpulse::Pulse;
  using fluxpulse::PulseProgram;
  using fluxpulse::Real;
  using fluxpulse::Segment;
  using fluxpulse::TimeGrid;

  constexpr Real kEps = Real{1e-12};

  // ---- validate(): grid + unique channels + finalize envelopes ----
  {
    PulseProgram prog(TimeGrid::Uniform(Real{0.0}, Real{0.1}, 11));
    // drive_x: [0.0, 0.5) -> 1.0
    Envelope ex;
    ex.add(Segment::PiecewiseConstant(Real{0.0}, Real{0.5}, Real{1.0}));

    // drive_y: [0.2, 0.4) -> -2.0
    Envelope ey;
    ey.add(Segment::PiecewiseConstant(Real{0.2}, Real{0.4}, Real{-2.0}));

    prog.add_pulse(Pulse(ChannelId::DriveX(), std::move(ex)));
    prog.add_pulse(Pulse(ChannelId::DriveY(), std::move(ey)));

    const fluxpulse::Status st = prog.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ---- validate(): duplicate channel should fail ----
  {
    PulseProgram prog(TimeGrid::Uniform(Real{0.0}, Real{0.1}, 11));

    Envelope e1;
    e1.add(Segment::PiecewiseConstant(Real{0.0}, Real{0.5}, Real{1.0}));
    Envelope e2;
    e2.add(Segment::PiecewiseConstant(Real{0.5}, Real{1.0}, Real{2.0}));

    prog.add_pulse(Pulse(ChannelId::DriveX(), std::move(e1)));
    prog.add_pulse(Pulse(ChannelId::DriveX(), std::move(e2)));  // duplicate

    const fluxpulse::Status st = prog.validate(/*finalize_envelopes=*/true);
    assert(!st.ok());
  }

  // ---- coeff(ch, t): correct values + gaps ----
  {
    PulseProgram prog(TimeGrid::Uniform(Real{0.0}, Real{0.1}, 11));

    Envelope ex;
    // [0.0, 0.3) -> 1.5 ; [0.6, 0.8) -> -1.0
    ex.add(Segment::PiecewiseConstant(Real{0.0}, Real{0.3}, Real{1.5}));
    ex.add(Segment::PiecewiseConstant(Real{0.6}, Real{0.8}, Real{-1.0}));

    prog.add_pulse(Pulse(ChannelId::DriveX(), std::move(ex)));

    assert(prog.validate(true).ok());

    expect_near(prog.coeff(ChannelId::DriveX(), Real{-1.0}), Real{0.0}, kEps);
    expect_near(prog.coeff(ChannelId::DriveX(), Real{0.0}), Real{1.5}, kEps);
    expect_near(prog.coeff(ChannelId::DriveX(), Real{0.29}), Real{1.5}, Real{1e-2});
    // boundary half-open: t==0.3 not included
    expect_near(prog.coeff(ChannelId::DriveX(), Real{0.3}), Real{0.0}, kEps);

    // gap [0.3,0.6)
    expect_near(prog.coeff(ChannelId::DriveX(), Real{0.5}), Real{0.0}, kEps);

    // second segment
    expect_near(prog.coeff(ChannelId::DriveX(), Real{0.6}), Real{-1.0}, kEps);
    expect_near(prog.coeff(ChannelId::DriveX(), Real{0.79}), Real{-1.0}, Real{1e-2});
    // boundary: t==0.8 not included
    expect_near(prog.coeff(ChannelId::DriveX(), Real{0.8}), Real{0.0}, kEps);

    // missing channel should return 0
    expect_near(prog.coeff(ChannelId::DriveY(), Real{0.1}), Real{0.0}, kEps);
  }

  // ---- coeff_on_grid(): monotonic cursor matches coeff(t) on uniform grid ----
  {
    TimeGrid grid = TimeGrid::Uniform(Real{0.0}, Real{0.1}, 11);  // t = 0..1.0
    PulseProgram prog(grid);

    Envelope ex;
    // [0.2, 0.6) -> 2.0
    ex.add(Segment::PiecewiseConstant(Real{0.2}, Real{0.6}, Real{2.0}));

    Envelope ez;
    // linear ramp [0.0, 0.5): 0 -> 1
    ez.add(Segment::PiecewiseLinear(Real{0.0}, Real{0.5}, Real{0.0}, Real{1.0}));

    prog.add_pulse(Pulse(ChannelId::DriveX(), std::move(ex)));
    prog.add_pulse(Pulse(ChannelId::DriveZ(), std::move(ez)));

    assert(prog.validate(true).ok());

    PulseProgram::Cursor cursor = prog.make_cursor();

    for (fluxpulse::Dim i = 0; i < grid.size(); ++i) {
      const Real t = grid.t(i);

      const Real ax = prog.coeff(ChannelId::DriveX(), t);
      const Real bx = prog.coeff_on_grid(ChannelId::DriveX(), i, &cursor);
      expect_near(ax, bx, kEps);

      const Real az = prog.coeff(ChannelId::DriveZ(), t);
      const Real bz = prog.coeff_on_grid(ChannelId::DriveZ(), i, &cursor);
      expect_near(az, bz, kEps);
    }
  }

  // ---- coeff_pulse_on_grid(): direct pulse index path ----
  {
    TimeGrid grid = TimeGrid::Uniform(Real{0.0}, Real{0.1}, 6);  // t=0..0.5
    PulseProgram prog(grid);

    Envelope ex;
    ex.add(Segment::PiecewiseConstant(Real{0.0}, Real{0.2}, Real{3.0}));  // t=0,0.1 -> 3.0

    prog.add_pulse(Pulse(ChannelId::DriveX(), std::move(ex)));
    assert(prog.validate(true).ok());

    PulseProgram::Cursor cursor = prog.make_cursor();

    // Pulse index 0
    expect_near(prog.coeff_pulse_on_grid(0, 0, &cursor), Real{3.0}, kEps);
    expect_near(prog.coeff_pulse_on_grid(0, 1, &cursor), Real{3.0}, kEps);
    // t=0.2 excluded => 0
    expect_near(prog.coeff_pulse_on_grid(0, 2, &cursor), Real{0.0}, kEps);
  }

  return 0;
}