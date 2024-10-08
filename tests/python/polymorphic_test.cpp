#include "proxsuite-nlp/python/polymorphic.hpp"
#include <eigenpy/std-vector.hpp>

using namespace proxsuite::nlp::python;

struct X {
  ~X() = default;
  virtual std::string name() const { return "I am X base"; }
  X() = default;

protected:
  X(const X &) = default;
};

struct Y : X {
  Y() : X() {}
  virtual std::string name() const override { return "I am Y, son of X"; }
};

struct Z final : Y {
  Z() : Y() {}
  std::string name() const override { return "I am Z, son of Y"; }
};

struct PyX final : X, PolymorphicWrapper<PyX, X> {
  std::string name() const override {
    if (bp::override f = get_override("name")) {
      return f();
    }
    return default_name();
  }
  std::string default_name() const { return X::name(); }
};

/// A callback class (extensible from Python) for the virtual
struct PyY final : Y, PolymorphicWrapper<PyY, Y> {
  std::string name() const override {
    if (bp::override f = get_override("name")) {
      return f();
    }
    return default_name();
  }
  std::string default_name() const { return Y::name(); }
};

namespace boost::python::objects {

template <> struct value_holder<PyX> : OwningNonOwningHolder<PyX> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};
template <> struct value_holder<PyY> : OwningNonOwningHolder<PyY> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

} // namespace boost::python::objects

using PolyX = xyz::polymorphic<X>;
using PolyY = xyz::polymorphic<Y>;
using VecPolyX = std::vector<PolyX>;

struct poly_use_base {
  template <class T> poly_use_base(const T &t) : x(t) {}

  PolyX x;
};

struct poly_store {
  poly_store(const PolyX &x) : x(x) {
    fmt::println("Called poly_store ctor: name is {:s}", x->name());
  }
  PolyX x;
};

struct vec_store {
  vec_store() = default;
  vec_store(const std::vector<PolyX> &x) : values(x) {}
  void add(const PolyX &x) { values.emplace_back(x); }
  const auto &get() const { return values; }

  auto get_copy() const { return values; }

private:
  VecPolyX values;
};

PolyX getY() { return PolyX(Y()); }
PolyX getZ() { return PolyX(Z()); }

PolyX poly_passthrough(const PolyX &x) {
  fmt::println("passthrough: got message {}", x->name());
  return x;
}

void call_method(const PolyX &x) {
  fmt::print("[{:s}] PolyX::name(): {:s}", __FUNCTION__, x->name());
  const bp::detail::wrapper_base *t =
      dynamic_cast<const bp::detail::wrapper_base *>(&(*x));
  if (t) {
    PyObject *o = bp::detail::wrapper_base_::get_owner(*t);
    PyTypeObject *type = o->ob_type;
    fmt::print(" | got bp::wrapper, Python type {:s}", type->tp_name);
  }
  fmt::println("");
}

static PolyX static_x = PolyX{Y()};

const PolyX &set_return(const PolyX &x) {
  static_x = x;
  return static_x;
}

BOOST_PYTHON_MODULE(MODULE_NAME) {
  register_polymorphic_to_python<PolyX>();
  register_polymorphic_to_python<PolyY>();

  bp::class_<PyX, boost::noncopyable>("X", bp::init<>())
      .def("name", &X::name, &PyX::default_name)
      .def(PolymorphicVisitor<PolyX>());
  bp::class_<PyY, bp::bases<X>, boost::noncopyable>("Y", bp::init<>())
      .def("name", &Y::name, &PyY::default_name)
      .def(PolymorphicVisitor<PolyX>())
      .def(PolymorphicVisitor<PolyY>());

  bp::class_<Z, bp::bases<Y>>("Z", bp::init<>())
      .def(PolymorphicVisitor<PolyX>())
      .def(PolymorphicVisitor<PolyY>());

  eigenpy::StdVectorPythonVisitor<VecPolyX>::expose(
      "VecPolyX",
      eigenpy::details::overload_base_get_item_for_std_vector<VecPolyX>());

  bp::def("getY", &getY);
  bp::def("getZ", &getZ);
  bp::def("poly_passthrough", &poly_passthrough, "x"_a);
  bp::def("call_method", &call_method, bp::args("x"));
  bp::def("set_return", &set_return,
          bp::return_value_policy<bp::reference_existing_object>());

  bp::class_<poly_use_base>("poly_use_base", bp::no_init)
      .def(bp::init<const Y &>(bp::args("self", "t")))
      .def(bp::init<const Z &>(bp::args("self", "t")))
      .def(bp::init<const PyX &>(bp::args("self", "t")))
      .def(bp::init<const PyY &>(bp::args("self", "t")))
      .def_readonly("x", &poly_use_base::x);

  bp::class_<poly_store>("poly_store", bp::init<const PolyX &>(("self"_a, "x")))
      .def_readonly("x", &poly_store::x);

  bp::class_<vec_store>("vec_store", bp::init<>())
      .def(bp::init<const std::vector<PolyX> &>(bp::args("self", "vec")))
      .def("add", &vec_store::add, ("self"_a, "x"))
      .def("get", &vec_store::get, ("self"_a),
           bp::return_internal_reference<>())
      .def("get_copy", &vec_store::get_copy, ("self"_a));
}
