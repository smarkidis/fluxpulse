// src/io/json_reader.cpp
#include "fluxpulse/io/json_reader.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

namespace fluxpulse {
namespace {

struct Parser {
  std::string_view s;
  size_t i = 0;
  size_t* err_pos = nullptr;

  explicit Parser(std::string_view in, size_t* error_pos) : s(in), i(0), err_pos(error_pos) {}

  void set_error_pos() {
    if (err_pos) {
      *err_pos = i;
    }
  }

  bool eof() const { return i >= s.size(); }
  char peek() const { return eof() ? '\0' : s[i]; }
  char get() { return eof() ? '\0' : s[i++]; }

  void skip_ws() {
    while (!eof()) {
      const unsigned char c = static_cast<unsigned char>(s[i]);
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++i;
        continue;
      }
      break;
    }
  }

  Status error(const char* msg) {
    set_error_pos();
    return Error(ErrorCode::kParse, msg);
  }

  static void append_utf8(uint32_t cp, std::string* out) {
    if (cp <= 0x7Fu) {
      out->push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
      out->push_back(static_cast<char>(0xC0u | ((cp >> 6) & 0x1Fu)));
      out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
      out->push_back(static_cast<char>(0xE0u | ((cp >> 12) & 0x0Fu)));
      out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
      out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
      out->push_back(static_cast<char>(0xF0u | ((cp >> 18) & 0x07u)));
      out->push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
      out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
      out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
  }

  static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
  }

  Status parse_value(JsonValue* out) {
    skip_ws();
    if (eof()) return error("Unexpected end of input while parsing JSON value");

    const char c = peek();
    if (c == 'n') return parse_null(out);
    if (c == 't' || c == 'f') return parse_bool(out);
    if (c == '"') return parse_string(out);
    if (c == '[') return parse_array(out);
    if (c == '{') return parse_object(out);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);

    return error("Invalid JSON value");
  }

  Status parse_null(JsonValue* out) {
    if (s.size() - i < 4 || s.substr(i, 4) != "null") {
      return error("Invalid token (expected 'null')");
    }
    i += 4;
    out->clear();
    out->type = JsonType::kNull;
    return Ok();
  }

  Status parse_bool(JsonValue* out) {
    if (s.size() - i >= 4 && s.substr(i, 4) == "true") {
      i += 4;
      out->clear();
      out->type = JsonType::kBool;
      out->bool_value = true;
      return Ok();
    }
    if (s.size() - i >= 5 && s.substr(i, 5) == "false") {
      i += 5;
      out->clear();
      out->type = JsonType::kBool;
      out->bool_value = false;
      return Ok();
    }
    return error("Invalid token (expected 'true' or 'false')");
  }

  Status parse_string(JsonValue* out) {
    if (get() != '"') return error("Invalid string (expected '\"')");

    std::string result;
    result.reserve(16);

    while (!eof()) {
      char c = get();
      if (c == '"') {
        out->clear();
        out->type = JsonType::kString;
        out->string_value = std::move(result);
        return Ok();
      }
      if (static_cast<unsigned char>(c) < 0x20u) {
        return error("Invalid string (control character)");
      }
      if (c != '\\') {
        result.push_back(c);
        continue;
      }

      if (eof()) return error("Invalid string escape (unexpected end)");
      const char esc = get();
      switch (esc) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
          if (s.size() - i < 4) return error("Invalid \\u escape (too short)");
          uint32_t cp = 0;
          for (int k = 0; k < 4; ++k) {
            const int hd = hex_digit(s[i + static_cast<size_t>(k)]);
            if (hd < 0) return error("Invalid \\u escape (non-hex digit)");
            cp = (cp << 4) | static_cast<uint32_t>(hd);
          }
          i += 4;
          // Note: does not combine surrogate pairs; fine for config usage.
          append_utf8(cp, &result);
          break;
        }
        default:
          return error("Invalid string escape sequence");
      }
    }
    return error("Unterminated string");
  }

  Status parse_number(JsonValue* out) {
    const size_t start = i;

    if (peek() == '-') {
      ++i;
      if (eof()) return error("Invalid number");
    }

    if (peek() == '0') {
      ++i;
    } else if (peek() >= '1' && peek() <= '9') {
      while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) ++i;
    } else {
      return error("Invalid number");
    }

    if (!eof() && peek() == '.') {
      ++i;
      if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) return error("Invalid number fraction");
      while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) ++i;
    }

    if (!eof() && (peek() == 'e' || peek() == 'E')) {
      ++i;
      if (!eof() && (peek() == '+' || peek() == '-')) ++i;
      if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) return error("Invalid number exponent");
      while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) ++i;
    }

    const size_t end = i;
    const std::string_view num_view = s.substr(start, end - start);

    std::string tmp(num_view.data(), num_view.size());
    char* end_ptr = nullptr;
    const double v = std::strtod(tmp.c_str(), &end_ptr);
    if (end_ptr == tmp.c_str()) return error("Invalid number (conversion failed)");

    out->clear();
    out->type = JsonType::kNumber;
    out->number_value = v;
    return Ok();
  }

  Status parse_array(JsonValue* out) {
    if (get() != '[') return error("Invalid array (expected '[')");

    out->clear();
    out->type = JsonType::kArray;
    out->array_values.clear();

    skip_ws();
    if (!eof() && peek() == ']') {
      get();
      return Ok();
    }

    while (true) {
      JsonValue v;
      FLUXPULSE_RETURN_IF_ERROR(parse_value(&v));
      out->array_values.push_back(std::move(v));

      skip_ws();
      if (eof()) return error("Unterminated array");
      const char c = get();
      if (c == ',') {
        skip_ws();
        continue;
      }
      if (c == ']') return Ok();
      return error("Invalid array (expected ',' or ']')");
    }
  }

  Status parse_object(JsonValue* out) {
    if (get() != '{') return error("Invalid object (expected '{')");

    out->clear();
    out->type = JsonType::kObject;
    out->object_members.clear();

    skip_ws();
    if (!eof() && peek() == '}') {
      get();
      return Ok();
    }

    while (true) {
      skip_ws();
      if (eof() || peek() != '"') return error("Invalid object (expected string key)");

      JsonValue key_v;
      FLUXPULSE_RETURN_IF_ERROR(parse_string(&key_v));
      std::string key = std::move(key_v.string_value);

      skip_ws();
      if (eof() || get() != ':') return error("Invalid object (expected ':')");

      JsonValue val;
      FLUXPULSE_RETURN_IF_ERROR(parse_value(&val));

      // Reject duplicate keys for determinism.
      for (const auto& kv : out->object_members) {
        if (kv.first == key) return error("Duplicate key in object");
      }
      out->object_members.emplace_back(std::move(key), std::move(val));

      skip_ws();
      if (eof()) return error("Unterminated object");
      const char c = get();
      if (c == ',') {
        skip_ws();
        continue;
      }
      if (c == '}') return Ok();
      return error("Invalid object (expected ',' or '}')");
    }
  }
};

// ---- Accessor helpers ----

inline Status require_object_type(const JsonValue& obj) {
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  return Ok();
}
inline Status require_string_type(const JsonValue& v) {
  if (!v.is_string()) return Error(ErrorCode::kInvalidArgument, "JSON: expected string");
  return Ok();
}
inline Status require_bool_type(const JsonValue& v) {
  if (!v.is_bool()) return Error(ErrorCode::kInvalidArgument, "JSON: expected bool");
  return Ok();
}
inline Status require_number_type(const JsonValue& v) {
  if (!v.is_number()) return Error(ErrorCode::kInvalidArgument, "JSON: expected number");
  return Ok();
}

inline bool is_integral_double(double x) {
  if (!std::isfinite(x)) return false;
  const double fx = std::floor(x);
  return fx == x;
}

template <typename T>
Status number_to_unsigned_integral(const JsonValue& v, const char* what, T* out) {
  FLUXPULSE_RETURN_IF_ERROR(require_number_type(v));
  const double x = v.number_value;
  if (!is_integral_double(x) || x < 0.0) return Error(ErrorCode::kInvalidArgument, what);
  const double maxv = static_cast<double>(std::numeric_limits<T>::max());
  if (x > maxv) return Error(ErrorCode::kOutOfRange, what);
  *out = static_cast<T>(x);
  return Ok();
}

}  // namespace

// ---- Reader API ----

Status parse_json(std::string_view text, JsonValue* out, size_t* error_pos) {
  if (!out) {
    if (error_pos) *error_pos = 0;
    return Error(ErrorCode::kInvalidArgument, "parse_json: out is null");
  }
  out->clear();
  if (error_pos) *error_pos = 0;

  Parser p(text, error_pos);
  Status st = p.parse_value(out);
  if (!st.ok()) return st;

  p.skip_ws();
  if (!p.eof()) {
    p.set_error_pos();
    return Error(ErrorCode::kParse, "Trailing characters after JSON value");
  }
  return Ok();
}

Status read_json_file(const std::string& path, JsonValue* out, size_t* error_pos) {
  if (!out) {
    if (error_pos) *error_pos = 0;
    return Error(ErrorCode::kInvalidArgument, "read_json_file: out is null");
  }
  if (error_pos) *error_pos = 0;

  std::ifstream f(path, std::ios::in | std::ios::binary);
  if (!f.good()) return Error(ErrorCode::kIo, "Failed to open JSON file");

  std::ostringstream ss;
  ss << f.rdbuf();
  if (!f.good() && !f.eof()) return Error(ErrorCode::kIo, "Failed to read JSON file");

  const std::string text = ss.str();
  return parse_json(std::string_view(text.data(), text.size()), out, error_pos);
}

// ---- Object accessors ----

Status get_object(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_object: out is null");
  *out = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(require_object_type(obj));
  const JsonValue* v = obj.find(key);
  if (!v) return Error(ErrorCode::kNotFound, "JSON: key not found");
  if (!v->is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object member");
  *out = v;
  return Ok();
}

Status get_array(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_array: out is null");
  *out = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(require_object_type(obj));
  const JsonValue* v = obj.find(key);
  if (!v) return Error(ErrorCode::kNotFound, "JSON: key not found");
  if (!v->is_array()) return Error(ErrorCode::kInvalidArgument, "JSON: expected array member");
  *out = v;
  return Ok();
}

Status get_string(const JsonValue& obj, std::string_view key, std::string* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_string: out is null");
  out->clear();
  FLUXPULSE_RETURN_IF_ERROR(require_object_type(obj));
  const JsonValue* v = obj.find(key);
  if (!v) return Error(ErrorCode::kNotFound, "JSON: key not found");
  FLUXPULSE_RETURN_IF_ERROR(require_string_type(*v));
  *out = v->string_value;
  return Ok();
}

Status get_bool(const JsonValue& obj, std::string_view key, bool* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_bool: out is null");
  *out = false;
  FLUXPULSE_RETURN_IF_ERROR(require_object_type(obj));
  const JsonValue* v = obj.find(key);
  if (!v) return Error(ErrorCode::kNotFound, "JSON: key not found");
  FLUXPULSE_RETURN_IF_ERROR(require_bool_type(*v));
  *out = v->bool_value;
  return Ok();
}

Status get_number(const JsonValue& obj, std::string_view key, Real* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_number: out is null");
  *out = Real{0};
  FLUXPULSE_RETURN_IF_ERROR(require_object_type(obj));
  const JsonValue* v = obj.find(key);
  if (!v) return Error(ErrorCode::kNotFound, "JSON: key not found");
  FLUXPULSE_RETURN_IF_ERROR(require_number_type(*v));
  *out = static_cast<Real>(v->number_value);
  return Ok();
}

Status get_u64(const JsonValue& obj, std::string_view key, uint64_t* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_u64: out is null");
  *out = 0;
  FLUXPULSE_RETURN_IF_ERROR(require_object_type(obj));
  const JsonValue* v = obj.find(key);
  if (!v) return Error(ErrorCode::kNotFound, "JSON: key not found");
  return number_to_unsigned_integral<uint64_t>(*v, "JSON: expected non-negative integer (u64)", out);
}

Status get_dim(const JsonValue& obj, std::string_view key, Dim* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_dim: out is null");
  *out = 0;
  FLUXPULSE_RETURN_IF_ERROR(require_object_type(obj));
  const JsonValue* v = obj.find(key);
  if (!v) return Error(ErrorCode::kNotFound, "JSON: key not found");
  return number_to_unsigned_integral<Dim>(*v, "JSON: expected non-negative integer (Dim)", out);
}

// ---- Required wrappers ----

Status get_required_object(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  Status st = get_object(obj, key, out);
  if (!st.ok()) return (st.code == ErrorCode::kNotFound)
                        ? Error(ErrorCode::kInvalidArgument, "JSON: missing required object")
                        : st;
  return st;
}

Status get_required_array(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  Status st = get_array(obj, key, out);
  if (!st.ok()) return (st.code == ErrorCode::kNotFound)
                        ? Error(ErrorCode::kInvalidArgument, "JSON: missing required array")
                        : st;
  return st;
}

Status get_required_string(const JsonValue& obj, std::string_view key, std::string* out) {
  Status st = get_string(obj, key, out);
  if (!st.ok()) return (st.code == ErrorCode::kNotFound)
                        ? Error(ErrorCode::kInvalidArgument, "JSON: missing required string")
                        : st;
  return st;
}

Status get_required_bool(const JsonValue& obj, std::string_view key, bool* out) {
  Status st = get_bool(obj, key, out);
  if (!st.ok()) return (st.code == ErrorCode::kNotFound)
                        ? Error(ErrorCode::kInvalidArgument, "JSON: missing required bool")
                        : st;
  return st;
}

Status get_required_number(const JsonValue& obj, std::string_view key, Real* out) {
  Status st = get_number(obj, key, out);
  if (!st.ok()) return (st.code == ErrorCode::kNotFound)
                        ? Error(ErrorCode::kInvalidArgument, "JSON: missing required number")
                        : st;
  return st;
}

Status get_required_u64(const JsonValue& obj, std::string_view key, uint64_t* out) {
  Status st = get_u64(obj, key, out);
  if (!st.ok()) return (st.code == ErrorCode::kNotFound)
                        ? Error(ErrorCode::kInvalidArgument, "JSON: missing required u64")
                        : st;
  return st;
}

Status get_required_dim(const JsonValue& obj, std::string_view key, Dim* out) {
  Status st = get_dim(obj, key, out);
  if (!st.ok()) return (st.code == ErrorCode::kNotFound)
                        ? Error(ErrorCode::kInvalidArgument, "JSON: missing required Dim")
                        : st;
  return st;
}

// ---- Direct type extraction ----

Status as_bool(const JsonValue& v, bool* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "as_bool: out is null");
  *out = false;
  FLUXPULSE_RETURN_IF_ERROR(require_bool_type(v));
  *out = v.bool_value;
  return Ok();
}

Status as_number(const JsonValue& v, Real* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "as_number: out is null");
  *out = Real{0};
  FLUXPULSE_RETURN_IF_ERROR(require_number_type(v));
  *out = static_cast<Real>(v.number_value);
  return Ok();
}

Status as_u64(const JsonValue& v, uint64_t* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "as_u64: out is null");
  *out = 0;
  return number_to_unsigned_integral<uint64_t>(v, "JSON: expected non-negative integer (u64)", out);
}

Status as_dim(const JsonValue& v, Dim* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "as_dim: out is null");
  *out = 0;
  return number_to_unsigned_integral<Dim>(v, "JSON: expected non-negative integer (Dim)", out);
}

Status as_string(const JsonValue& v, std::string* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "as_string: out is null");
  out->clear();
  FLUXPULSE_RETURN_IF_ERROR(require_string_type(v));
  *out = v.string_value;
  return Ok();
}

}  // namespace fluxpulse