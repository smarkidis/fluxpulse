// src/sim/observable_builders.cpp
#include "fluxpulse/sim/observable_builders.hpp"

#include <cassert>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/operators/tensor.hpp"

namespace fluxpulse {
namespace sim {
namespace {

inline bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  return s;
}

std::string lower_ascii(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool all_digits(std::string_view s) {
  for (char c : s) {
    if (!is_digit(c)) return false;
  }
  return !s.empty();
}

Status parse_u64(std::string_view s, uint64_t* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_u64: out is null");
  s = trim(s);
  if (!all_digits(s)) return Error(ErrorCode::kInvalidArgument, "Expected unsigned integer");
  uint64_t v = 0;
  for (char c : s) {
    const uint64_t d = static_cast<uint64_t>(c - '0');
    if (v > (std::numeric_limits<uint64_t>::max() - d) / 10u) {
      return Error(ErrorCode::kOutOfRange, "Integer overflow");
    }
    v = v * 10u + d;
  }
  *out = v;
  return Ok();
}

Status parse_index_list(std::string_view suffix,
                        size_t expected_count,
                        std::vector<Index>* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_index_list: out is null");
  out->clear();

  suffix = trim(suffix);
  if (suffix.empty()) {
    return Error(ErrorCode::kInvalidArgument, "Missing indices for Pauli product");
  }

  const bool digits_only = all_digits(suffix);
  if (digits_only && suffix.size() == expected_count) {
    // Ambiguity resolver: if we have exactly as many digits as letters,
    // treat each digit as a separate index (e.g., "ZZ01" -> [0,1]).
    out->reserve(expected_count);
    for (char c : suffix) {
      out->push_back(static_cast<Index>(c - '0'));
    }
    return Ok();
  }

  // Otherwise parse integer tokens separated by non-digit chars.
  out->reserve(expected_count);

  size_t i = 0;
  while (i < suffix.size()) {
    while (i < suffix.size() && !is_digit(suffix[i])) ++i;
    if (i >= suffix.size()) break;

    size_t j = i;
    while (j < suffix.size() && is_digit(suffix[j])) ++j;

    uint64_t v = 0;
    FLUXPULSE_RETURN_IF_ERROR(parse_u64(suffix.substr(i, j - i), &v));
    if (v > static_cast<uint64_t>(std::numeric_limits<Index>::max())) {
      return Error(ErrorCode::kOutOfRange, "Index too large");
    }
    out->push_back(static_cast<Index>(v));
    i = j;
  }

  if (out->size() != expected_count) {
    return Error(ErrorCode::kInvalidArgument, "Index count does not match Pauli string length");
  }
  return Ok();
}

DenseOp pauli_x2() {
  DenseOp A(2);
  A.set_zero();
  A.set(0, 1, complex_one());
  A.set(1, 0, complex_one());
  return A;
}

DenseOp pauli_y2() {
  DenseOp A(2);
  A.set_zero();
  // [ [0, -i],
  //   [i,  0] ]
  A.set(0, 1, Complex{Real{0}, Real{-1}});
  A.set(1, 0, Complex{Real{0}, Real{+1}});
  return A;
}

DenseOp pauli_z2() {
  DenseOp A(2);
  A.set_zero();
  A.set(0, 0, complex_one());
  A.set(1, 1, Complex{Real{-1}, Real{0}});
  return A;
}

DenseOp projector_basis(Dim d, Dim k) {
  DenseOp P(d);
  P.set_zero();
  assert(k >= 0 && k < d);
  P.set(static_cast<Index>(k), static_cast<Index>(k), complex_one());
  return P;
}

DenseOp number_op(Dim d) {
  DenseOp N(d);
  N.set_zero();
  for (Dim k = 0; k < d; ++k) {
    N.set(static_cast<Index>(k), static_cast<Index>(k), Complex{static_cast<Real>(k), Real{0}});
  }
  return N;
}

Status validate_dims_nonempty(const std::vector<Dim>& dims) {
  if (dims.empty()) return Error(ErrorCode::kInvalidArgument, "subsystem_dims must be non-empty");
  for (Dim d : dims) {
    if (d <= 0) return Error(ErrorCode::kInvalidArgument, "subsystem_dims contains non-positive dimension");
  }
  return Ok();
}

Dim prod_dims_checked(const std::vector<Dim>& dims) {
  // dims are small in Phase 1; tensor.hpp has prod_dims() too.
  return prod_dims(dims);
}

Status build_projector_from_digits_big_endian(std::string_view digits_be,
                                              const std::vector<Dim>& dims,
                                              DenseOp* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "build_projector_from_digits_big_endian: out is null");
  digits_be = trim(digits_be);

  const size_t n = dims.size();

  // Case A: digits-only and length == n => per-subsystem digits (single-digit each).
  if (all_digits(digits_be) && digits_be.size() == n) {
    uint64_t linear = 0;
    uint64_t stride = 1;
    // little-endian digits are reverse of big-endian string
    for (size_t i = 0; i < n; ++i) {
      const char c = digits_be[n - 1 - i];
      const uint64_t dig = static_cast<uint64_t>(c - '0');
      if (dig >= static_cast<uint64_t>(dims[i])) {
        return Error(ErrorCode::kOutOfRange, "P:<digits> digit out of range for subsystem dims");
      }
      linear += dig * stride;
      stride *= static_cast<uint64_t>(dims[i]);
    }

    const Dim D = prod_dims_checked(dims);
    DenseOp P(D);
    P.set_zero();
    P.set(static_cast<Index>(linear), static_cast<Index>(linear), complex_one());
    *out = std::move(P);
    return Ok();
  }

  // Case B: token list (e.g. "0,1,0,1") for per-subsystem digits.
  std::vector<uint64_t> tokens;
  {
    size_t i = 0;
    while (i < digits_be.size()) {
      while (i < digits_be.size() && !is_digit(digits_be[i])) ++i;
      if (i >= digits_be.size()) break;
      size_t j = i;
      while (j < digits_be.size() && is_digit(digits_be[j])) ++j;
      uint64_t v = 0;
      FLUXPULSE_RETURN_IF_ERROR(parse_u64(digits_be.substr(i, j - i), &v));
      tokens.push_back(v);
      i = j;
    }
  }

  if (!tokens.empty()) {
    if (tokens.size() != n) {
      return Error(ErrorCode::kInvalidArgument, "P:<...> token count must match number of subsystems");
    }
    uint64_t linear = 0;
    uint64_t stride = 1;
    for (size_t i = 0; i < n; ++i) {
      const uint64_t dig_be = tokens[i];
      const size_t le = n - 1 - i;
      if (dig_be >= static_cast<uint64_t>(dims[le])) {
        return Error(ErrorCode::kOutOfRange, "P:<...> token digit out of range for subsystem dims");
      }
      // tokens are big-endian: tokens[0] corresponds to subsystem n-1
      // so little-endian digit for subsystem le = tokens[i]
      linear += dig_be * stride;
      stride *= static_cast<uint64_t>(dims[le]);
    }

    const Dim D = prod_dims_checked(dims);
    DenseOp P(D);
    P.set_zero();
    P.set(static_cast<Index>(linear), static_cast<Index>(linear), complex_one());
    *out = std::move(P);
    return Ok();
  }

  // Case C: treat as linear index if it's just an integer like "3"
  if (all_digits(digits_be)) {
    uint64_t li = 0;
    FLUXPULSE_RETURN_IF_ERROR(parse_u64(digits_be, &li));

    uint64_t total = 1;
    for (Dim d : dims) {
      total *= static_cast<uint64_t>(d);
    }
    if (li >= total) return Error(ErrorCode::kOutOfRange, "P:<int> linear index out of range for dims product");

    DenseOp P(static_cast<Dim>(total));
    P.set_zero();
    P.set(static_cast<Index>(li), static_cast<Index>(li), complex_one());
    *out = std::move(P);
    return Ok();
  }

  return Error(ErrorCode::kInvalidArgument, "Unsupported projector syntax after 'P:'");
}

Status build_pauli_on_qubit(char axis, Index q, const std::vector<Dim>& dims, DenseOp* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "build_pauli_on_qubit: out is null");
  if (q < 0 || static_cast<size_t>(q) >= dims.size()) {
    return Error(ErrorCode::kOutOfRange, "Qubit index out of range");
  }
  if (dims[static_cast<size_t>(q)] != 2) {
    return Error(ErrorCode::kInvalidArgument, "Pauli observable requires subsystem dim == 2");
  }

  DenseOp P;
  switch (axis) {
    case 'x': P = pauli_x2(); break;
    case 'y': P = pauli_y2(); break;
    case 'z': P = pauli_z2(); break;
    default:
      return Error(ErrorCode::kInvalidArgument, "Unknown Pauli axis");
  }

  // If single-qubit system, return local directly for convenience.
  if (dims.size() == 1) {
    *out = std::move(P);
    return Ok();
  }

  // Embed into full space.
  // tensor.hpp provides embed_single(local,dims,qubit) via embed_local.
  *out = embed_single(P, dims, q);
  return Ok();
}

Status build_pauli_product(const std::string& letters,
                           const std::vector<Index>& qs,
                           const std::vector<Dim>& dims,
                           DenseOp* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "build_pauli_product: out is null");
  if (letters.size() != qs.size()) {
    return Error(ErrorCode::kInvalidArgument, "Pauli string letters and indices size mismatch");
  }
  if (letters.empty()) return Error(ErrorCode::kInvalidArgument, "Empty Pauli product");

  // Validate indices + dims
  for (Index q : qs) {
    if (q < 0 || static_cast<size_t>(q) >= dims.size()) {
      return Error(ErrorCode::kOutOfRange, "Pauli product qubit index out of range");
    }
    if (dims[static_cast<size_t>(q)] != 2) {
      return Error(ErrorCode::kInvalidArgument, "Pauli product requires subsystem dim == 2 for all targets");
    }
  }

  // If k == 1, defer to single-qubit builder.
  if (qs.size() == 1) {
    return build_pauli_on_qubit(static_cast<char>(std::tolower(static_cast<unsigned char>(letters[0]))),
                               qs[0], dims, out);
  }

  // Build full operator via kron across all subsystems (msb -> lsb), placing identity where not targeted.
  // For Phase 1 (N<=4), this is fine and avoids relying on special-case helpers.
  std::vector<DenseOp> locals(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    locals[i] = identity_op(dims[i]);
  }
  for (size_t k = 0; k < qs.size(); ++k) {
    const Index q = qs[k];
    const char ax = static_cast<char>(std::tolower(static_cast<unsigned char>(letters[k])));
    DenseOp P;
    if (ax == 'x') P = pauli_x2();
    else if (ax == 'y') P = pauli_y2();
    else if (ax == 'z') P = pauli_z2();
    else return Error(ErrorCode::kInvalidArgument, "Unknown Pauli axis in product");
    locals[static_cast<size_t>(q)] = std::move(P);
  }

  DenseOp full = identity1();
  // Build from least-significant subsystem (0) up to most-significant (N-1).
  // kron(A,B): A acts on more-significant subsystems, B on less-significant.
  // This loop ensures locals[i] is placed on subsystem i under little-endian ordering.
  for (size_t i = 0; i < dims.size(); ++i) {
    full = kron(locals[i], full);
  }
  *out = std::move(full);
  return Ok();
}

Status parse_pauli_letters_and_suffix(std::string_view s, std::string* out_letters, std::string_view* out_suffix) {
  if (!out_letters || !out_suffix) return Error(ErrorCode::kInvalidArgument, "parse_pauli_letters_and_suffix: null");
  out_letters->clear();
  s = trim(s);
  if (s.empty()) return Error(ErrorCode::kInvalidArgument, "Empty observable");

  size_t i = 0;
  while (i < s.size()) {
    const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
    if (c == 'x' || c == 'y' || c == 'z') {
      out_letters->push_back(c);
      ++i;
      continue;
    }
    break;
  }
  if (out_letters->empty()) return Error(ErrorCode::kInvalidArgument, "Not a Pauli observable");
  *out_suffix = s.substr(i);
  return Ok();
}

Status parse_projector_qubit_suffix(std::string_view suffix, Index* out_qubit) {
  if (!out_qubit) return Error(ErrorCode::kInvalidArgument, "parse_projector_qubit_suffix: out is null");
  suffix = trim(suffix);
  if (suffix.empty()) {
    *out_qubit = 0;
    return Ok();
  }

  // suffix may contain separators, parse first integer token.
  size_t i = 0;
  while (i < suffix.size() && !is_digit(suffix[i])) ++i;
  if (i >= suffix.size()) return Error(ErrorCode::kInvalidArgument, "Projector qubit suffix must include an integer");
  size_t j = i;
  while (j < suffix.size() && is_digit(suffix[j])) ++j;
  uint64_t v = 0;
  FLUXPULSE_RETURN_IF_ERROR(parse_u64(suffix.substr(i, j - i), &v));
  if (v > static_cast<uint64_t>(std::numeric_limits<Index>::max())) return Error(ErrorCode::kOutOfRange, "Index too large");
  *out_qubit = static_cast<Index>(v);
  return Ok();
}

}  // namespace

Status build_observable(const std::string& name,
                        const std::vector<Dim>& subsystem_dims,
                        DenseOp* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "build_observable: out is null");
  FLUXPULSE_RETURN_IF_ERROR(validate_dims_nonempty(subsystem_dims));

  const std::string_view sv = trim(std::string_view(name.data(), name.size()));
  if (sv.empty()) return Error(ErrorCode::kInvalidArgument, "Observable name is empty");

  const std::string low = lower_ascii(sv);

  // Identity
  if (low == "i" || low == "id" || low == "identity") {
    const Dim D = prod_dims_checked(subsystem_dims);
    DenseOp I(D);
    I.set_identity();
    *out = std::move(I);
    return Ok();
  }

  // Number operator for single-mode systems (oscillator/transmon)
  if (low == "n") {
    if (subsystem_dims.size() != 1) {
      return Error(ErrorCode::kInvalidArgument, "Observable 'N' is only supported for single-mode dims");
    }
    *out = number_op(subsystem_dims[0]);
    return Ok();
  }

  // Projectors
  if (low.size() >= 2 && low[0] == 'p') {
    // "P:..." full projector
    if (low.size() >= 2 && low[1] == ':') {
      return build_projector_from_digits_big_endian(sv.substr(2), subsystem_dims, out);
    }

    // "P0" / "P1" (qubit projector), optional qubit suffix.
    const char which = low[1];
    if (which != '0' && which != '1') {
      return Error(ErrorCode::kInvalidArgument, "Projector must be P0, P1, or P:<digits>");
    }

    Index q = 0;
    FLUXPULSE_RETURN_IF_ERROR(parse_projector_qubit_suffix(sv.substr(2), &q));
    if (q < 0 || static_cast<size_t>(q) >= subsystem_dims.size()) {
      return Error(ErrorCode::kOutOfRange, "Projector qubit index out of range");
    }
    if (subsystem_dims[static_cast<size_t>(q)] != 2) {
      return Error(ErrorCode::kInvalidArgument, "P0/P1 projector requires target subsystem dim == 2");
    }

    DenseOp Ploc = projector_basis(2, static_cast<Dim>(which - '0'));
    if (subsystem_dims.size() == 1) {
      *out = std::move(Ploc);
      return Ok();
    }
    *out = embed_single(Ploc, subsystem_dims, q);
    return Ok();
  }

  // Pauli single-qubit short form: "X", "Y", "Z" (only when single-qubit system).
  if ((low == "x" || low == "y" || low == "z")) {
    if (!(subsystem_dims.size() == 1 && subsystem_dims[0] == 2)) {
      return Error(ErrorCode::kInvalidArgument, "Use X0/Y0/Z0 for multi-qubit systems");
    }
    return build_pauli_on_qubit(low[0], /*q=*/0, subsystem_dims, out);
  }

  // General Pauli parsing: letters first, then indices.
  std::string letters;
  std::string_view suffix;
  Status st_parse = parse_pauli_letters_and_suffix(sv, &letters, &suffix);
  if (st_parse.ok()) {
    std::vector<Index> qs;
    FLUXPULSE_RETURN_IF_ERROR(parse_index_list(suffix, letters.size(), &qs));
    return build_pauli_product(letters, qs, subsystem_dims, out);
  }

  return Error(ErrorCode::kInvalidArgument, "Unknown observable name");
}

Status build_observables(const std::vector<std::string>& names,
                         const std::vector<Dim>& subsystem_dims,
                         std::vector<DenseOp>* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "build_observables: out is null");
  out->clear();
  out->reserve(names.size());
  for (const std::string& n : names) {
    DenseOp op;
    Status st = build_observable(n, subsystem_dims, &op);
    if (!st.ok()) return Annotate(st, "build_observables: " + n);
    out->push_back(std::move(op));
  }
  return Ok();
}

}  // namespace sim
}  // namespace fluxpulse
