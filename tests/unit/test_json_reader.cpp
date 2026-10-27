// tests/unit/test_json_reader.cpp
#include "fluxpulse/io/json_reader.hpp"

#include <cassert>
#include <string>
#include <string_view>

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline void expect_ok(const fluxpulse::Status& st) { assert(st.ok()); }
inline void expect_fail(const fluxpulse::Status& st) { assert(!st.ok()); }

}  // namespace

int main() {
  using fluxpulse::Dim;
  using fluxpulse::JsonType;
  using fluxpulse::JsonValue;
  using fluxpulse::Real;

  constexpr Real kEps = Real{1e-12};

  // ---- Parse primitive values ----
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("null", &v, &pos));
    assert(v.type == JsonType::kNull);
  }
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json(" true ", &v, &pos));
    assert(v.is_bool());
    assert(v.bool_value == true);
  }
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("false", &v, &pos));
    assert(v.is_bool());
    assert(v.bool_value == false);
  }
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json(" 3.25 ", &v, &pos));
    assert(v.is_number());
    expect_near(static_cast<Real>(v.number_value), Real{3.25}, kEps);
  }
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("-2e3", &v, &pos));
    assert(v.is_number());
    expect_near(static_cast<Real>(v.number_value), Real{-2000.0}, Real{1e-9});
  }
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("\"hello\"", &v, &pos));
    assert(v.is_string());
    assert(v.string_value == "hello");
  }

  // ---- Parse string escapes ----
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("\"a\\\\b\\n\\t\\\"c\"", &v, &pos));
    assert(v.is_string());
    assert(v.string_value.size() >= 5);
    assert(v.string_value[0] == 'a');
    assert(v.string_value.find('\n') != std::string::npos);
    assert(v.string_value.find('\t') != std::string::npos);
    assert(v.string_value.find('"') != std::string::npos);
  }

  // ---- Parse arrays ----
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("[1, 2, 3]", &v, &pos));
    assert(v.is_array());
    assert(v.array_values.size() == 3);
    assert(v.array_values[0].is_number());
    assert(v.array_values[1].is_number());
    assert(v.array_values[2].is_number());
    expect_near(static_cast<Real>(v.array_values[2].number_value), Real{3.0}, kEps);
  }

  // ---- Parse objects + find() ----
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("{\"a\":1,\"b\":true,\"c\":\"x\"}", &v, &pos));
    assert(v.is_object());
    assert(v.object_members.size() == 3);

    const JsonValue* a = v.find("a");
    const JsonValue* b = v.find("b");
    const JsonValue* c = v.find("c");
    const JsonValue* d = v.find("d");

    assert(a && a->is_number());
    assert(b && b->is_bool());
    assert(c && c->is_string());
    assert(d == nullptr);

    expect_near(static_cast<Real>(a->number_value), Real{1.0}, kEps);
    assert(b->bool_value == true);
    assert(c->string_value == "x");
  }

  // ---- Duplicate keys rejected ----
  {
    JsonValue v;
    size_t pos = 0;
    const fluxpulse::Status st = fluxpulse::parse_json("{\"a\":1,\"a\":2}", &v, &pos);
    expect_fail(st);
    assert(pos > 0);
  }

  // ---- Trailing characters rejected ----
  {
    JsonValue v;
    size_t pos = 0;
    const fluxpulse::Status st = fluxpulse::parse_json("true false", &v, &pos);
    expect_fail(st);
    assert(pos > 0);
  }

  // ---- Accessors: get_required_* and optional get_* ----
  {
    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json(
        "{"
        "  \"x\": 1.5,"
        "  \"flag\": false,"
        "  \"name\": \"q0\","
        "  \"arr\": [1,2,3],"
        "  \"obj\": {\"k\": 7}"
        "}",
        &root, &pos));
    assert(root.is_object());

    Real x = Real{0};
    bool flag = true;
    std::string name;
    const JsonValue* arr = nullptr;
    const JsonValue* obj = nullptr;

    expect_ok(fluxpulse::get_required_number(root, "x", &x));
    expect_ok(fluxpulse::get_required_bool(root, "flag", &flag));
    expect_ok(fluxpulse::get_required_string(root, "name", &name));
    expect_ok(fluxpulse::get_required_array(root, "arr", &arr));
    expect_ok(fluxpulse::get_required_object(root, "obj", &obj));

    expect_near(x, Real{1.5}, kEps);
    assert(flag == false);
    assert(name == "q0");
    assert(arr && arr->is_array() && arr->array_values.size() == 3);
    assert(obj && obj->is_object());

    // Optional: missing key => kNotFound
    Real y = Real{0};
    const fluxpulse::Status st = fluxpulse::get_number(root, "missing", &y);
    assert(!st.ok());
    assert(st.code == fluxpulse::ErrorCode::kNotFound);

    // Nested access
    const JsonValue* k_v = obj->find("k");
    assert(k_v && k_v->is_number());
    uint64_t k_u64 = 0;
    expect_ok(fluxpulse::as_u64(*k_v, &k_u64));
    assert(k_u64 == 7u);
  }

  // ---- Integer conversion: u64 / Dim requires integral non-negative ----
  {
    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("{\"u\": 5, \"d\": 3}", &root, &pos));
    uint64_t u = 0;
    Dim d = 0;
    expect_ok(fluxpulse::get_required_u64(root, "u", &u));
    expect_ok(fluxpulse::get_required_dim(root, "d", &d));
    assert(u == 5u);
    assert(d == 3);
  }
  {
    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("{\"u\": -1}", &root, &pos));
    uint64_t u = 0;
    const fluxpulse::Status st = fluxpulse::get_u64(root, "u", &u);
    expect_fail(st);
  }
  {
    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("{\"u\": 1.25}", &root, &pos));
    uint64_t u = 0;
    const fluxpulse::Status st = fluxpulse::get_u64(root, "u", &u);
    expect_fail(st);
  }

  // ---- Direct as_* type mismatch ----
  {
    JsonValue v;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json("\"abc\"", &v, &pos));
    Real x = Real{0};
    const fluxpulse::Status st = fluxpulse::as_number(v, &x);
    expect_fail(st);
  }

  return 0;
}