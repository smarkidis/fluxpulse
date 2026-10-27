// include/fluxpulse/core/error.hpp
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace fluxpulse {

// Keep error handling explicit and exception-free.
// Intended mainly for I/O, config, and other non-hot-path code.
enum class ErrorCode : uint32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kIo = 2,
  kParse = 3,
  kNotFound = 4,
  kOutOfRange = 5,
  kUnimplemented = 6,
  kInternal = 7,
};

inline const char* error_code_name(ErrorCode code) {
  switch (code) {
    case ErrorCode::kOk: return "OK";
    case ErrorCode::kInvalidArgument: return "INVALID_ARGUMENT";
    case ErrorCode::kIo: return "IO";
    case ErrorCode::kParse: return "PARSE";
    case ErrorCode::kNotFound: return "NOT_FOUND";
    case ErrorCode::kOutOfRange: return "OUT_OF_RANGE";
    case ErrorCode::kUnimplemented: return "UNIMPLEMENTED";
    case ErrorCode::kInternal: return "INTERNAL";
    default: return "UNKNOWN";
  }
}

struct [[nodiscard]] Status {
  ErrorCode code = ErrorCode::kOk;
  std::string message;

  constexpr bool ok() const { return code == ErrorCode::kOk; }
};

[[nodiscard]] inline Status Ok() { return Status{}; }

[[nodiscard]] inline Status Error(ErrorCode code, std::string_view msg) {
  Status s;
  s.code = code;
  s.message.assign(msg.data(), msg.size());
  return s;
}

// Adds context to an existing error. If `st` is OK, it is returned unchanged.
[[nodiscard]] inline Status Annotate(Status st, std::string_view context) {
  if (st.ok() || context.empty()) {
    return st;
  }
  if (st.message.empty()) {
    st.message.assign(context.data(), context.size());
    return st;
  }
  std::string out;
  out.reserve(context.size() + 2 + st.message.size());
  out.append(context.data(), context.size());
  out.append(": ");
  out.append(st.message);
  st.message = std::move(out);
  return st;
}

// Convenience macro for propagating Status-returning calls without exceptions.
#define FLUXPULSE_RETURN_IF_ERROR(expr)        \
  do {                                         \
    ::fluxpulse::Status _st = (expr);          \
    if (!_st.ok()) {                           \
      return _st;                              \
    }                                          \
  } while (0)

}  // namespace fluxpulse