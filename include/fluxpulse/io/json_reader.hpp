// include/fluxpulse/io/json_reader.hpp
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// Minimal JSON AST + reader interface.
// Intended for configuration files (small/medium JSON).
//
// Conventions:
// - Strict JSON (no comments, no trailing commas).
// - Numbers stored as double; accessors convert to Real/integers.
// - Objects store members in insertion order (vector of pairs), lookup is linear.
// - No exceptions: all failures return Status.
enum class JsonType : uint8_t {
  kNull = 0,
  kBool = 1,
  kNumber = 2,
  kString = 3,
  kArray = 4,
  kObject = 5,
};

// A simple JSON value representation.
// This is not optimized for memory; it is optimized for clarity and determinism.
struct JsonValue {
  JsonType type = JsonType::kNull;

  // Payload (only one is meaningful depending on type).
  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::vector<JsonValue> array_values;
  std::vector<std::pair<std::string, JsonValue>> object_members;

  void clear() {
    type = JsonType::kNull;
    bool_value = false;
    number_value = 0.0;
    string_value.clear();
    array_values.clear();
    object_members.clear();
  }

  // Type checks
  bool is_null() const { return type == JsonType::kNull; }
  bool is_bool() const { return type == JsonType::kBool; }
  bool is_number() const { return type == JsonType::kNumber; }
  bool is_string() const { return type == JsonType::kString; }
  bool is_array() const { return type == JsonType::kArray; }
  bool is_object() const { return type == JsonType::kObject; }

  // Object lookup (returns nullptr if not found or not an object).
  const JsonValue* find(std::string_view key) const {
    if (type != JsonType::kObject) {
      return nullptr;
    }
    for (const auto& kv : object_members) {
      const std::string& k = kv.first;
      if (k.size() == key.size() && std::string_view(k.data(), k.size()) == key) {
        return &kv.second;
      }
    }
    return nullptr;
  }

  JsonValue* find_mut(std::string_view key) {
    if (type != JsonType::kObject) {
      return nullptr;
    }
    for (auto& kv : object_members) {
      std::string& k = kv.first;
      if (k.size() == key.size() && std::string_view(k.data(), k.size()) == key) {
        return &kv.second;
      }
    }
    return nullptr;
  }
};

// ---- Reader API ----

// Parse JSON text into JsonValue.
// On parse failure, returns Status(kParse, "...") and optionally sets error_pos to a byte offset.
Status parse_json(std::string_view text, JsonValue* out, size_t* error_pos);

// Read file into memory and parse JSON.
// On failure, returns Status(kIo, ...) or Status(kParse, ...).
Status read_json_file(const std::string& path, JsonValue* out, size_t* error_pos);

// ---- Accessor helpers (object member extraction) ----
//
// Convention:
// - get_* returns kNotFound if key is missing.
// - get_required_* returns kInvalidArgument if key is missing.
// - type mismatch returns kInvalidArgument.
// - integer conversion checks for non-negative and integral (no fractional) values.

// Optional accessors
Status get_bool(const JsonValue& obj, std::string_view key, bool* out);
Status get_number(const JsonValue& obj, std::string_view key, Real* out);
Status get_u64(const JsonValue& obj, std::string_view key, uint64_t* out);
Status get_dim(const JsonValue& obj, std::string_view key, Dim* out);
Status get_string(const JsonValue& obj, std::string_view key, std::string* out);

// Returns pointer to member value (no copying). Fails if missing or wrong type.
Status get_object(const JsonValue& obj, std::string_view key, const JsonValue** out);
Status get_array(const JsonValue& obj, std::string_view key, const JsonValue** out);

// Required accessors (must exist)
Status get_required_bool(const JsonValue& obj, std::string_view key, bool* out);
Status get_required_number(const JsonValue& obj, std::string_view key, Real* out);
Status get_required_u64(const JsonValue& obj, std::string_view key, uint64_t* out);
Status get_required_dim(const JsonValue& obj, std::string_view key, Dim* out);
Status get_required_string(const JsonValue& obj, std::string_view key, std::string* out);

Status get_required_object(const JsonValue& obj, std::string_view key, const JsonValue** out);
Status get_required_array(const JsonValue& obj, std::string_view key, const JsonValue** out);

// Direct type extraction (useful when iterating arrays)
Status as_bool(const JsonValue& v, bool* out);
Status as_number(const JsonValue& v, Real* out);
Status as_u64(const JsonValue& v, uint64_t* out);
Status as_dim(const JsonValue& v, Dim* out);
Status as_string(const JsonValue& v, std::string* out);

}  // namespace fluxpulse