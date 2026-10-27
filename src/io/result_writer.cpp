// src/io/result_writer.cpp
#include "fluxpulse/io/result_writer.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fluxpulse {
namespace io {
namespace {

inline std::string default_obs_name(size_t k) {
  std::ostringstream ss;
  ss << "obs" << k;
  return ss.str();
}

inline bool all_names_nonempty_unique(const std::vector<std::string>& names) {
  if (names.empty()) return false;
  for (const std::string& n : names) {
    if (n.empty()) return false;
  }
  for (size_t i = 0; i < names.size(); ++i) {
    for (size_t j = i + 1; j < names.size(); ++j) {
      if (names[i] == names[j]) return false;
    }
  }
  return true;
}

inline Status open_out(std::ofstream* f, const std::string& path) {
  f->open(path, std::ios::out | std::ios::binary);
  if (!f->good()) {
    return Error(ErrorCode::kIo, "Failed to open output file for writing");
  }
  return Ok();
}

inline void write_json_string(std::ostream& os, const std::string& s) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\b': os << "\\b"; break;
      case '\f': os << "\\f"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default:
        os << c;
        break;
    }
  }
  os << '"';
}

}  // namespace

Status write_result_json(const std::string& path, const Result& r) {
  const Dim nT = r.num_times();
  const Dim nO = r.num_observables();

  if (nT < 0 || nO < 0) {
    return Error(ErrorCode::kInvalidArgument, "write_result_json: invalid sizes");
  }
  if (static_cast<size_t>(nT) != r.tlist.size()) {
    return Error(ErrorCode::kInvalidArgument, "write_result_json: tlist size mismatch");
  }
  if (static_cast<size_t>(nO) != r.observable_names.size()) {
    return Error(ErrorCode::kInvalidArgument, "write_result_json: observable_names size mismatch");
  }
  if (r.expectations.size() != static_cast<size_t>(nT) * static_cast<size_t>(nO)) {
    return Error(ErrorCode::kInvalidArgument, "write_result_json: expectations size mismatch");
  }

  const bool has_var = r.has_expect_var();
  const bool has_stderr = r.has_expect_stderr();

  std::ofstream f;
  FLUXPULSE_RETURN_IF_ERROR(open_out(&f, path));

  const bool named_map = all_names_nonempty_unique(r.observable_names);

  f << "{\n";

  // tlist
  f << "  \"tlist\": [";
  for (Dim i = 0; i < nT; ++i) {
    if (i) f << ", ";
    f << r.tlist[static_cast<size_t>(i)];
  }
  f << "],\n";

  // observables
  f << "  \"observables\": [";
  for (Dim k = 0; k < nO; ++k) {
    if (k) f << ", ";
    const std::string& name = r.observable_names[static_cast<size_t>(k)];
    write_json_string(f, name.empty() ? default_obs_name(static_cast<size_t>(k)) : name);
  }
  f << "],\n";

  // Optional mean/var/stderr maps (only if names are usable)
  if (named_map) {
    // Mean map
    f << "  \"expectations\": {\n";
    for (Dim k = 0; k < nO; ++k) {
      const std::string& name = r.observable_names[static_cast<size_t>(k)];
      f << "    ";
      write_json_string(f, name);
      f << ": [";
      for (Dim i = 0; i < nT; ++i) {
        if (i) f << ", ";
        f << r.expect_at(k, i);
      }
      f << "]";
      if (k + 1 < nO) f << ",";
      f << "\n";
    }
    f << "  }";

    // Var map
    if (has_var) {
      f << ",\n  \"expect_var\": {\n";
      for (Dim k = 0; k < nO; ++k) {
        const std::string& name = r.observable_names[static_cast<size_t>(k)];
        f << "    ";
        write_json_string(f, name);
        f << ": [";
        for (Dim i = 0; i < nT; ++i) {
          if (i) f << ", ";
          f << r.expect_var_at(k, i);
        }
        f << "]";
        if (k + 1 < nO) f << ",";
        f << "\n";
      }
      f << "  }";
    }

    // Stderr map
    if (has_stderr) {
      f << ",\n  \"expect_stderr\": {\n";
      for (Dim k = 0; k < nO; ++k) {
        const std::string& name = r.observable_names[static_cast<size_t>(k)];
        f << "    ";
        write_json_string(f, name);
        f << ": [";
        for (Dim i = 0; i < nT; ++i) {
          if (i) f << ", ";
          f << r.expect_stderr_at(k, i);
        }
        f << "]";
        if (k + 1 < nO) f << ",";
        f << "\n";
      }
      f << "  }";
    }

    f << ",\n";
  }

  // Always provide row-major fallback list, with optional var/stderr arrays.
  f << "  \"data\": [\n";
  for (Dim k = 0; k < nO; ++k) {
    const std::string& name = r.observable_names[static_cast<size_t>(k)];
    f << "    {\"name\": ";
    write_json_string(f, name.empty() ? default_obs_name(static_cast<size_t>(k)) : name);

    // Mean
    f << ", \"values\": [";
    for (Dim i = 0; i < nT; ++i) {
      if (i) f << ", ";
      f << r.expect_at(k, i);
    }
    f << "]";

    // Var
    if (has_var) {
      f << ", \"var\": [";
      for (Dim i = 0; i < nT; ++i) {
        if (i) f << ", ";
        f << r.expect_var_at(k, i);
      }
      f << "]";
    }

    // Stderr
    if (has_stderr) {
      f << ", \"stderr\": [";
      for (Dim i = 0; i < nT; ++i) {
        if (i) f << ", ";
        f << r.expect_stderr_at(k, i);
      }
      f << "]";
    }

    f << "}";
    if (k + 1 < nO) f << ",";
    f << "\n";
  }
  f << "  ]";

  // Optional final pure state vector.
  //
  // This is populated for single-trajectory runs when the solver exposes a
  // well-defined pure state.  Multi-trajectory MCWF ensemble runs leave it
  // empty because an ensemble does not have a unique state vector.
  if (!r.final_state.empty()) {
    Real norm2 = Real{0};
    for (const Complex& z : r.final_state) {
      norm2 += z.re * z.re + z.im * z.im;
    }

    f << ",\n";
    f << "  \"final_state\": {\n";
    f << "    \"representation\": \"state_vector\",\n";
    f << "    \"basis_order\": \"little_endian\",\n";
    f << "    \"dim\": " << r.final_state.size() << ",\n";

    f << "    \"real\": [";
    for (size_t i = 0; i < r.final_state.size(); ++i) {
      if (i) f << ", ";
      f << r.final_state[i].re;
    }
    f << "],\n";

    f << "    \"imag\": [";
    for (size_t i = 0; i < r.final_state.size(); ++i) {
      if (i) f << ", ";
      f << r.final_state[i].im;
    }
    f << "],\n";

    f << "    \"norm2\": " << norm2 << "\n";
    f << "  }";
  }

  f << "\n";
  f << "}\n";

  if (!f.good()) {
    return Error(ErrorCode::kIo, "write_result_json: write failed");
  }
  return Ok();
}

Status write_result_csv(const std::string& path, const Result& r) {
  const Dim nT = r.num_times();
  const Dim nO = r.num_observables();

  if (static_cast<size_t>(nT) != r.tlist.size()) {
    return Error(ErrorCode::kInvalidArgument, "write_result_csv: tlist size mismatch");
  }
  if (static_cast<size_t>(nO) != r.observable_names.size()) {
    return Error(ErrorCode::kInvalidArgument, "write_result_csv: observable_names size mismatch");
  }
  if (r.expectations.size() != static_cast<size_t>(nT) * static_cast<size_t>(nO)) {
    return Error(ErrorCode::kInvalidArgument, "write_result_csv: expectations size mismatch");
  }

  const bool has_var = r.has_expect_var();
  const bool has_stderr = r.has_expect_stderr();

  std::ofstream f;
  FLUXPULSE_RETURN_IF_ERROR(open_out(&f, path));

  // Header
  f << "t";
  for (Dim k = 0; k < nO; ++k) {
    const std::string& name = r.observable_names[static_cast<size_t>(k)];
    const std::string col = name.empty() ? default_obs_name(static_cast<size_t>(k)) : name;

    f << "," << col;
    if (has_var) f << "," << col << "_var";
    if (has_stderr) f << "," << col << "_stderr";
  }
  f << "\n";

  // Rows
  for (Dim i = 0; i < nT; ++i) {
    f << r.tlist[static_cast<size_t>(i)];
    for (Dim k = 0; k < nO; ++k) {
      f << "," << r.expect_at(k, i);
      if (has_var) f << "," << r.expect_var_at(k, i);
      if (has_stderr) f << "," << r.expect_stderr_at(k, i);
    }
    f << "\n";
  }

  if (!f.good()) {
    return Error(ErrorCode::kIo, "write_result_csv: write failed");
  }
  return Ok();
}

}  // namespace io
}  // namespace fluxpulse
