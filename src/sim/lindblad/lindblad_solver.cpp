// src/sim/lindblad/lindblad_solver.cpp
#include "fluxpulse/sim/lindblad/lindblad_solver.hpp"

#include <cassert>
#include <cmath>
#include <limits>

namespace fluxpulse {
namespace lindblad {
namespace {

// -------------------- Complex helpers (POD Complex: {re,im}) --------------------

inline Complex c_sub(Complex a, Complex b) {
  return Complex{a.re - b.re, a.im - b.im};
}
inline Complex c_mul(Complex a, Complex b) {
  // (a.re + i a.im) (b.re + i b.im)
  return Complex{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
inline Complex c_conj(Complex a) {
  return Complex{a.re, Real{0} - a.im};
}
inline Complex c_mul_minus_i(Complex a) {
  // (-i) * (a.re + i a.im) = a.im - i a.re
  return Complex{a.im, Real{0} - a.re};
}
inline Complex c_zero() { return Complex{Real{0}, Real{0}}; }

// -------------------- Indexing --------------------

inline size_t idx_cm(Dim n, Index row, Index col) {
  return static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n);
}

// -------------------- Dense matrix ops (column-major, n x n) --------------------

void mat_zero(Complex* out, Dim n) {
  const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < N; ++i) out[i] = c_zero();
}

void mat_copy(const Complex* src, Complex* dst, Dim n) {
  const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < N; ++i) dst[i] = src[i];
}

void mat_axpy(Complex* y, const Complex* x, Real a, Dim n) {
  // y += a*x
  const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < N; ++i) {
    y[i].re += a * x[i].re;
    y[i].im += a * x[i].im;
  }
}

void mat_scale(Complex* x, Real a, Dim n) {
  const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < N; ++i) {
    x[i].re *= a;
    x[i].im *= a;
  }
}

// out = A * B (both n x n, column-major)
void mat_mul(const Complex* A, const Complex* B, Complex* out, Dim n) {
  mat_zero(out, n);
  const size_t nn = static_cast<size_t>(n);
  for (size_t j = 0; j < nn; ++j) {
    for (size_t k = 0; k < nn; ++k) {
      const Complex bkj = B[k + j * nn];
      if (bkj.re == Real{0} && bkj.im == Real{0}) continue;
      for (size_t i = 0; i < nn; ++i) {
        const Complex aik = A[i + k * nn];
        const Complex prod = c_mul(aik, bkj);
        Complex& oij = out[i + j * nn];
        oij.re += prod.re;
        oij.im += prod.im;
      }
    }
  }
}

// out = A^† (conjugate transpose) of A
void mat_dagger(const DenseOp& A, DenseOp* out) {
  const Dim n = A.dim();
  out->resize(n);
  for (Index col = 0; col < n; ++col) {
    for (Index row = 0; row < n; ++row) {
      // out(row,col) = conj(A(col,row))
      out->set(row, col, c_conj(A.at(col, row)));
    }
  }
}

// out = A * B (DenseOp operands)
void dense_mul(const DenseOp& A, const DenseOp& B, DenseOp* out) {
  const Dim n = A.dim();
  out->resize(n);
  Complex* od = out->data();
  std::vector<Complex> tmp(static_cast<size_t>(n) * static_cast<size_t>(n));
  mat_mul(A.data(), B.data(), tmp.data(), n);
  const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < N; ++i) od[i] = tmp[i];
}

// trace(rho) for column-major rho
Complex mat_trace(const Complex* rho, Dim n) {
  Complex tr = c_zero();
  for (Index i = 0; i < n; ++i) {
    const Complex a = rho[idx_cm(n, i, i)];
    tr.re += a.re;
    tr.im += a.im;
  }
  return tr;
}

void mat_enforce_hermiticity(Complex* rho, Dim n) {
  // rho <- (rho + rho^†)/2
  const size_t nn = static_cast<size_t>(n);
  for (size_t j = 0; j < nn; ++j) {
    for (size_t i = 0; i <= j; ++i) {
      const Complex a = rho[i + j * nn];
      const Complex b = c_conj(rho[j + i * nn]);
      Complex sym;
      sym.re = Real{0.5} * (a.re + b.re);
      sym.im = Real{0.5} * (a.im + b.im);
      rho[i + j * nn] = sym;
      rho[j + i * nn] = c_conj(sym);
    }
  }
}

Dim compute_num_samples(Dim n_points, Dim stride) {
  if (n_points <= 0) return 0;
  if (stride <= 0) stride = 1;
  const Dim last = n_points - 1;
  Dim n = last / stride + 1;
  if ((last % stride) != 0) n += 1;
  return n;
}

// Build H(t) = H_drift + sum_k u_k(t) * H_control[k]
Status build_hamiltonian_at_t(const ModelBase& model,
                              const PulseProgram& program,
                              Real t,
                              LindbladWorkspace* ws,
                              DenseOp* Ht) {
  if (!ws || !Ht) return Error(ErrorCode::kInvalidArgument, "build_hamiltonian_at_t: null ptr");
  const Dim n = model.hilbert_dim;
  Ht->resize(n);
  Ht->set_zero();

  // Start from drift sum
  if (ws->H_drift.dim() == n) {
    const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
    for (size_t i = 0; i < N; ++i) {
      Ht->data()[i] = ws->H_drift.data()[i];
    }
  }

  // Add control terms
  for (const ControlTerm& ct : model.control_terms) {
    const Real u = program.coeff(ct.channel, t);
    if (u == Real{0}) continue;
    // Ht += u * ct.H
    const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
    const Complex* hd = ct.H.data();
    Complex* out = Ht->data();
    for (size_t i = 0; i < N; ++i) {
      out[i].re += u * hd[i].re;
      out[i].im += u * hd[i].im;
    }
  }

  return Ok();
}

// Compute Lindblad RHS:
//   drho = -i [H, rho] + Σ_j ( C rho C† - 1/2 {C†C, rho} )
Status lindblad_rhs(Real t,
                    const ModelBase& model,
                    const PulseProgram& program,
                    const Complex* rho,
                    Complex* drho,
                    LindbladWorkspace* ws) {
  if (!rho || !drho || !ws) return Error(ErrorCode::kInvalidArgument, "lindblad_rhs: null ptr");
  const Dim n = model.hilbert_dim;

  // Build H(t)
  DenseOp Ht;
  FLUXPULSE_RETURN_IF_ERROR(build_hamiltonian_at_t(model, program, t, ws, &Ht));

  // tmp1 = H rho
  mat_mul(Ht.data(), rho, ws->tmp1.data(), n);
  // tmp2 = rho H
  mat_mul(rho, Ht.data(), ws->tmp2.data(), n);

  // drho = -i (tmp1 - tmp2)
  const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < N; ++i) {
    const Complex comm = c_sub(ws->tmp1[i], ws->tmp2[i]);
    drho[i] = c_mul_minus_i(comm);
  }

  // Dissipator terms
  for (size_t j = 0; j < model.collapse_ops.size(); ++j) {
    const DenseOp& C = model.collapse_ops[j].C;
    const DenseOp& Cdg = ws->Cdag[j];
    const DenseOp& CdC = ws->CdC[j];

    // tmp1 = C rho
    mat_mul(C.data(), rho, ws->tmp1.data(), n);
    // tmp2 = tmp1 C† = C rho C†
    mat_mul(ws->tmp1.data(), Cdg.data(), ws->tmp2.data(), n);

    // drho += tmp2
    for (size_t i = 0; i < N; ++i) {
      drho[i].re += ws->tmp2[i].re;
      drho[i].im += ws->tmp2[i].im;
    }

    // tmp2 = CdC rho
    mat_mul(CdC.data(), rho, ws->tmp2.data(), n);
    // tmp3 = rho CdC
    mat_mul(rho, CdC.data(), ws->tmp3.data(), n);

    // drho -= 0.5*(tmp2 + tmp3)
    for (size_t i = 0; i < N; ++i) {
      drho[i].re -= Real{0.5} * (ws->tmp2[i].re + ws->tmp3[i].re);
      drho[i].im -= Real{0.5} * (ws->tmp2[i].im + ws->tmp3[i].im);
    }
  }

  return Ok();
}

Real expectation_tr_O_rho_real(const DenseOp& O, const Complex* rho, Dim n) {
  // Tr(O rho) = sum_{i,j} O_{i,j} rho_{j,i}
  Complex s = c_zero();
  const Complex* Od = O.data();
  const size_t nn = static_cast<size_t>(n);
  for (size_t j = 0; j < nn; ++j) {
    for (size_t i = 0; i < nn; ++i) {
      const Complex Oij = Od[i + j * nn];
      const Complex rji = rho[j + i * nn];
      const Complex prod = c_mul(Oij, rji);
      s.re += prod.re;
      s.im += prod.im;
    }
  }
  return s.re;
}

}  // namespace

Status LindbladWorkspace::init(const ModelBase& model) {
  FLUXPULSE_RETURN_IF_ERROR(model.validate());

  dim = model.hilbert_dim;

  // Precompute drift sum
  H_drift.resize(dim);
  H_drift.set_zero();
  if (!model.drift_terms.empty()) {
    const size_t N = static_cast<size_t>(dim) * static_cast<size_t>(dim);
    for (const DenseOp& Hd : model.drift_terms) {
      if (Hd.dim() != dim) return Error(ErrorCode::kInvalidArgument, "LindbladWorkspace: drift dim mismatch");
      for (size_t i = 0; i < N; ++i) {
        H_drift.data()[i].re += Hd.data()[i].re;
        H_drift.data()[i].im += Hd.data()[i].im;
      }
    }
  }

  // Precompute collapse helpers
  Cdag.clear();
  CdC.clear();
  Cdag.reserve(model.collapse_ops.size());
  CdC.reserve(model.collapse_ops.size());

  for (const CollapseOp& cop : model.collapse_ops) {
    if (cop.C.dim() != dim) return Error(ErrorCode::kInvalidArgument, "LindbladWorkspace: collapse dim mismatch");
    DenseOp cdg;
    mat_dagger(cop.C, &cdg);
    DenseOp cdc;
    dense_mul(cdg, cop.C, &cdc);
    Cdag.push_back(std::move(cdg));
    CdC.push_back(std::move(cdc));
  }

  // Scratch buffers
  const size_t NN = static_cast<size_t>(dim) * static_cast<size_t>(dim);
  rho.assign(NN, c_zero());
  k1.assign(NN, c_zero());
  k2.assign(NN, c_zero());
  k3.assign(NN, c_zero());
  k4.assign(NN, c_zero());
  rho_stage.assign(NN, c_zero());
  tmp1.assign(NN, c_zero());
  tmp2.assign(NN, c_zero());
  tmp3.assign(NN, c_zero());

  return Ok();
}

Status rho_from_psi(Span<const Complex> psi, std::vector<Complex>* out_rho) {
  if (!out_rho) return Error(ErrorCode::kInvalidArgument, "rho_from_psi: out_rho null");
  if (psi.size <= 0) return Error(ErrorCode::kInvalidArgument, "rho_from_psi: empty psi");
  const Dim n = psi.size;
  out_rho->assign(static_cast<size_t>(n) * static_cast<size_t>(n), c_zero());

  for (Index col = 0; col < n; ++col) {
    const Complex psi_col = psi[col];  // |psi> component
    const Complex psi_col_conj = c_conj(psi_col);
    for (Index row = 0; row < n; ++row) {
      // rho(row,col) = psi(row) * conj(psi(col))
      const Complex v = c_mul(psi[row], psi_col_conj);
      (*out_rho)[idx_cm(n, row, col)] = v;
    }
  }
  return Ok();
}

Status run_lindblad(const ModelBase& model,
                    const PulseProgram& program,
                    Span<const DenseOp> observables,
                    Span<const Complex> rho0,
                    const LindbladOptions& opt,
                    LindbladWorkspace* ws,
                    Result* out) {
  if (!ws || !out) return Error(ErrorCode::kInvalidArgument, "run_lindblad: null ptr");
  FLUXPULSE_RETURN_IF_ERROR(model.validate());

  // Validate program grid
  PulseProgram tmp_prog = program;
  FLUXPULSE_RETURN_IF_ERROR(tmp_prog.validate(/*finalize_envelopes=*/false));

  const Dim n = model.hilbert_dim;
  if (rho0.size != n * n) return Error(ErrorCode::kInvalidArgument, "run_lindblad: rho0 size mismatch");

  for (Dim k = 0; k < observables.size; ++k) {
    if (observables[k].dim() != n) return Error(ErrorCode::kInvalidArgument, "run_lindblad: observable dim mismatch");
  }

  if (opt.sample_stride <= 0) return Error(ErrorCode::kInvalidArgument, "run_lindblad: sample_stride must be >= 1");

  // Ensure workspace is initialized for this model
  if (ws->dim != n) {
    FLUXPULSE_RETURN_IF_ERROR(ws->init(model));
  }

  // Copy rho0
  mat_copy(rho0.data, ws->rho.data(), n);

  const Dim n_points = program.grid.size();
  const Dim stride = opt.sample_stride;
  const Dim n_samples = compute_num_samples(n_points, stride);

  out->resize(observables.size, n_samples);

  // Record helper
  Dim sample_pos = 0;
  auto record_at_grid = [&](Dim grid_i) -> Status {
    const Real t = program.grid.t(grid_i);
    out->tlist[static_cast<size_t>(sample_pos)] = t;

    for (Dim o = 0; o < observables.size; ++o) {
      const Real ex = expectation_tr_O_rho_real(observables[o], ws->rho.data(), n);
      out->expectations[static_cast<size_t>(o) * static_cast<size_t>(n_samples) + static_cast<size_t>(sample_pos)] = ex;
    }
    sample_pos += 1;
    return Ok();
  };

  FLUXPULSE_RETURN_IF_ERROR(record_at_grid(0));

  // RK4 over grid intervals
  for (Dim i = 0; i + 1 < n_points; ++i) {
    const Real t = program.grid.t(i);
    const Real t_next = program.grid.t(i + 1);
    const Real dt = t_next - t;

    // k1
    FLUXPULSE_RETURN_IF_ERROR(lindblad_rhs(t, model, program, ws->rho.data(), ws->k1.data(), ws));

    // Keep the stage input separate from scratch written by lindblad_rhs().
    mat_copy(ws->rho.data(), ws->rho_stage.data(), n);
    mat_axpy(ws->rho_stage.data(), ws->k1.data(), Real{0.5} * dt, n);

    // k2
    FLUXPULSE_RETURN_IF_ERROR(lindblad_rhs(t + Real{0.5} * dt, model, program, ws->rho_stage.data(), ws->k2.data(), ws));

    // rho_tmp = rho + dt/2 k2
    mat_copy(ws->rho.data(), ws->rho_stage.data(), n);
    mat_axpy(ws->rho_stage.data(), ws->k2.data(), Real{0.5} * dt, n);

    // k3
    FLUXPULSE_RETURN_IF_ERROR(lindblad_rhs(t + Real{0.5} * dt, model, program, ws->rho_stage.data(), ws->k3.data(), ws));

    // rho_tmp = rho + dt k3
    mat_copy(ws->rho.data(), ws->rho_stage.data(), n);
    mat_axpy(ws->rho_stage.data(), ws->k3.data(), dt, n);

    // k4
    FLUXPULSE_RETURN_IF_ERROR(lindblad_rhs(t + dt, model, program, ws->rho_stage.data(), ws->k4.data(), ws));

    // rho <- rho + dt/6 (k1 + 2k2 + 2k3 + k4)
    // tmp1 = k1 + 2k2 + 2k3 + k4
    mat_copy(ws->k1.data(), ws->tmp1.data(), n);
    mat_axpy(ws->tmp1.data(), ws->k2.data(), Real{2}, n);
    mat_axpy(ws->tmp1.data(), ws->k3.data(), Real{2}, n);
    mat_axpy(ws->tmp1.data(), ws->k4.data(), Real{1}, n);

    mat_axpy(ws->rho.data(), ws->tmp1.data(), dt / Real{6}, n);

    // Optional stabilization
    if (opt.enforce_hermiticity) {
      mat_enforce_hermiticity(ws->rho.data(), n);
    }
    if (opt.enforce_trace_one) {
      const Complex tr = mat_trace(ws->rho.data(), n);
      const Real tr_re = tr.re;
      if (std::fabs(static_cast<double>(tr_re)) > 1e-15) {
        mat_scale(ws->rho.data(), Real{1} / tr_re, n);
      }
    }

    const Dim ip1 = i + 1;
    if ((ip1 % stride) == 0) {
      FLUXPULSE_RETURN_IF_ERROR(record_at_grid(ip1));
    }
  }

  // Always include last point if stride doesn't hit it.
  const Dim last = n_points - 1;
  if ((last % stride) != 0) {
    FLUXPULSE_RETURN_IF_ERROR(record_at_grid(last));
  }

  if (sample_pos != n_samples) {
    return Error(ErrorCode::kInternal, "run_lindblad: sample count mismatch");
  }

  return Ok();
}

}  // namespace lindblad
}  // namespace fluxpulse
