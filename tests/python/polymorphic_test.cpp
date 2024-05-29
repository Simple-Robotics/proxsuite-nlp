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

struct W final : Y {
  W() : Y() {}
  std::string name() const override { return "W (Y)"; }
};

using PolyX = xyz::polymorphic<X>;
using PolyY = xyz::polymorphic<Y>;
using VecPolyX = std::vector<PolyX>;

struct poly_use_base {
  template <class T> poly_use_base(const T &t) : x(t) {}

  PolyX x;
};

struct poly_store {
  poly_store(const PolyX &x) : x(x) {}
  PolyX x;
};

VecPolyX create_vec_poly() { return {Y(), Y(), W()}; }

PolyX getY() { return PolyX(Y()); }
PolyX getW() { return PolyX(W()); }

PolyX poly_passthrough(const PolyX &x) { return x; }

static PolyX static_x = PolyX{Y()};

const PolyX &get_const_y_poly_ref() { return static_x; }

const PolyX &set_return(PolyX x) {
  static_x = x;
  return static_x;
}

BOOST_PYTHON_MODULE(polymorphic_test) {
  register_polymorphic_to_python<PolyX>();
  register_polymorphic_to_python<PolyY>();

  bp::class_<X, boost::noncopyable>("X", bp::init<>()).def("name", &X::name);
  bp::class_<Y, bp::bases<X>>("Y", bp::init<>());

  bp::class_<W, bp::bases<Y>>("W", bp::init<>());

  eigenpy::StdVectorPythonVisitor<VecPolyX>::expose(
      "VecPolyX",
      eigenpy::details::overload_base_get_item_for_std_vector<VecPolyX>());

  bp::def("getY", &getY);
  bp::def("getW", &getW);
  bp::def("create_vec_poly", &create_vec_poly);
  bp::def("poly_passthrough", &poly_passthrough, "x"_a);
  bp::def("call_method", &call_method, bp::args("x"));
  bp::def("get_const_y_poly_ref", &get_const_y_poly_ref,
          bp::return_value_policy<bp::reference_existing_object>());
  bp::def("set_return", &set_return,
          bp::return_value_policy<bp::reference_existing_object>());

  bp::class_<poly_use_base>("poly_use_base", bp::no_init)
      .def(bp::init<const Y &>(bp::args("self", "t")))
      .def(bp::init<const W &>(bp::args("self", "t")))
      .add_property("x", bp::make_getter(&poly_use_base::x,
                                         bp::return_internal_reference<>()));

  bp::class_<poly_store>("poly_store", bp::init<const PolyX &>(("self"_a, "x")))
      .add_property("x", bp::make_getter(&poly_store::x,
                                         bp::return_internal_reference<>()));
}
