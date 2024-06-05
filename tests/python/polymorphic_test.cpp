#include "proxsuite-nlp/python/polymorphic.hpp"
#include <eigenpy/std-vector.hpp>

using namespace proxsuite::nlp::python;

struct X {
  ~X() = default;
  virtual std::string name() const { return "X"; }
  X() = default;

protected:
  X(const X &) = default;
};

struct Y : X {
  Y() : X() {}
  std::string name() const override { return "Y"; }
};

struct Z final : Y {
  Z() : Y() {}
  std::string name() const override { return "Z (Y)"; }
};

struct PyX final : X, bp::wrapper<X> {
  std::string name() const override {
    if (bp::override f = get_override("name")) {
      return f();
    }
    return X::name();
  }
};

using PolyX = xyz::polymorphic<X>;
using PolyY = xyz::polymorphic<Y>;
using VecPolyX = std::vector<PolyX>;

struct poly_use_base {
  template <class T> poly_use_base(const T &t) : x(t) {}

  PolyX x;
};

struct X_wrap_store {
  X_wrap_store(const PyX &x) : x(x) {}
  PyX x;
};

struct poly_store {
  poly_store(const PolyX &x) : x(x) {
    std::cout << "Called poly_store ctor: name is " << x->name() << std::endl;
  }
  PolyX x;
};

VecPolyX create_vec_poly() { return {Y(), Y(), Z()}; }

PolyX getY() { return PolyX(Y()); }
PolyX getZ() { return PolyX(Z()); }

PolyX poly_passthrough(const PolyX &x) { return x; }

void call_method(const PolyX &x) {
  fmt::println("  - {:s} PolyX::name(): {:s}", __FUNCTION__, x->name());
  auto t = dynamic_cast<const bp::wrapper<X> *>(&(*x));
  if (t) {
    fmt::println("  - {:s} got a bp::wrapper<X>", __FUNCTION__);
  }
}

static PolyX static_x = PolyX{Y()};

const PolyX &set_return(const PolyX &x) {
  static_x = x;
  return static_x;
}

BOOST_PYTHON_MODULE(polymorphic_test) {
  register_polymorphic_to_python<PolyX>();
  register_polymorphic_to_python<PolyY>();

  bp::class_<PyX, boost::noncopyable>("X", bp::init<>())
      .def("name", &X::name)
      .def(PolymorphicVisitor<PolyX>());
  bp::class_<Y, bp::bases<X>>("Y", bp::init<>())
      .def(PolymorphicVisitor<PolyX>());

  bp::class_<Z, bp::bases<Y>>("Z", bp::init<>())
      .def(PolymorphicVisitor<PolyX>())
      .def(PolymorphicVisitor<PolyY>());

  eigenpy::StdVectorPythonVisitor<VecPolyX>::expose(
      "VecPolyX",
      eigenpy::details::overload_base_get_item_for_std_vector<VecPolyX>());

  bp::def("getY", &getY);
  bp::def("getZ", &getZ);
  bp::def("create_vec_poly", &create_vec_poly);
  bp::def("poly_passthrough", &poly_passthrough, "x"_a);
  bp::def("call_method", &call_method, bp::args("x"));
  bp::def("set_return", &set_return,
          bp::return_value_policy<bp::reference_existing_object>());

  bp::class_<X_wrap_store>("X_wrap_store", bp::init<const PyX &>("x"_a))
      .def_readwrite("x", &X_wrap_store::x);

  bp::class_<poly_use_base>("poly_use_base", bp::no_init)
      .def(bp::init<const Y &>(bp::args("self", "t")))
      .def(bp::init<const Z &>(bp::args("self", "t")))
      .def(bp::init<const PyX &>(bp::args("self", "t")))
      .def_readonly("x", &poly_use_base::x);

  bp::class_<poly_store>("poly_store", bp::init<const PolyX &>(("self"_a, "x")))
      .def_readonly("x", &poly_store::x);
}
