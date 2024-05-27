#include "proxsuite-nlp/python/polymorphic.hpp"

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

struct poly_use_base {
  template <class T> poly_use_base(const T &t) : x(t) {}

  PolyX x;
};

PolyX getY() { return PolyX(Y()); }
PolyX getZ() { return PolyX(Z()); }

BOOST_PYTHON_MODULE(polymorphic_test) {
  register_polymorphic_to_python<PolyX>();

  bp::class_<X, boost::noncopyable>("X", bp::init<>()).def("name", &X::name);
  bp::class_<Y, bp::bases<X>>("Y", bp::init<>());
  bp::class_<Z, bp::bases<X>>("Z", bp::init<>());

  bp::def("getY", &getY);
  bp::def("getZ", &getZ);

  bp::class_<poly_use_base>("poly_use_base", bp::no_init)
      .def(bp::init<const Y &>(bp::args("self", "t")))
      .def(bp::init<const Z &>(bp::args("self", "t")))
      .add_property("x",
                    bp::make_getter(&poly_use_base::x, ReturnInternalPoly{}));
}
