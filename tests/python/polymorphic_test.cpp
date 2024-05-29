#include "proxsuite-nlp/python/polymorphic.hpp"
#include <eigenpy/std-vector.hpp>

using namespace proxsuite::nlp::python;

struct X {
  ~X() = default;
  virtual std::string name() { return "X"; }
  X() = default;

protected:
  X(const X &) = default;
};

struct Y final : X {
  Y() : X() {}
  std::string name() override { return "Y"; }
};

struct Z final : X {
  Z() : X() {}
  std::string name() override { return "Z"; }
};

using PolyX = xyz::polymorphic<X>;
using VecPolyX = std::vector<PolyX>;

struct poly_use_base {
  template <class T> poly_use_base(const T &t) : x(t) {}

  PolyX x;
};

VecPolyX create_vec_poly() { return {Y(), Z(), Z()}; }

PolyX getY() { return PolyX(Y()); }
PolyX getZ() { return PolyX(Z()); }

PolyX poly_passthrough(const PolyX &x) { return x; }

static PolyX static_x = PolyX{Y()};

const PolyX &get_const_y_poly_ref() { return static_x; }

BOOST_PYTHON_MODULE(polymorphic_test) {
  register_polymorphic_to_python<PolyX>();

  bp::class_<X, boost::noncopyable>("X", bp::init<>()).def("name", &X::name);
  bp::class_<Y, bp::bases<X>>("Y", bp::init<>());
  bp::implicitly_convertible<Y, PolyX>();
  bp::class_<Z, bp::bases<X>>("Z", bp::init<>());
  bp::implicitly_convertible<Z, PolyX>();

  eigenpy::StdVectorPythonVisitor<VecPolyX>::expose(
      "VecPolyX",
      eigenpy::details::overload_base_get_item_for_std_vector<VecPolyX>());

  bp::def("getY", &getY);
  bp::def("getZ", &getZ);
  bp::def("create_vec_poly", &create_vec_poly);
  bp::def("poly_passthrough", &poly_passthrough, "x"_a);
  bp::def("get_const_y_poly_ref", &get_const_y_poly_ref,
          bp::return_value_policy<bp::reference_existing_object>());

  bp::class_<poly_use_base>("poly_use_base", bp::no_init)
      .def(bp::init<const Y &>(bp::args("self", "t")))
      .def(bp::init<const Z &>(bp::args("self", "t")))
      .add_property("x", bp::make_getter(&poly_use_base::x,
                                         bp::return_internal_reference<>()));
}
