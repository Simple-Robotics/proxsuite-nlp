#include "proxnlp/python/function.hpp"
#include "proxnlp/function-ops.hpp"

namespace proxnlp {
namespace python {
using context::C1Function;
using context::C2Function;
using context::ConstVectorRef;
using context::Function;
using context::Scalar;

void exposeFunctionOps();

void exposeFunctionTypes() {

  bp::class_<FunctionWrap, boost::noncopyable>(
      "BaseFunction", "Base class for functions.",
      bp::init<int, int, int>(bp::args("self", "nx", "ndx", "nr")))
      .def("__call__", bp::pure_virtual(&Function::operator()),
           bp::args("self", "x"), "Call the function.")
      .add_property("nx", &Function::nx, "Input dimension")
      .add_property("ndx", &Function::ndx, "Input tangent space dimension.")
      .add_property("nr", &Function::nr, "Function codimension.");

  context::MatFuncType C1Function::*compJac1 = &C1Function::computeJacobian;
  context::MatFuncRetType C1Function::*compJac2 = &C1Function::computeJacobian;

  bp::class_<C1FunctionWrap, bp::bases<Function>, boost::noncopyable>(
      "C1Function", "Base class for differentiable functions",
      bp::init<int, int, int>(bp::args("self", "nx", "ndx", "nr")))
      .def("computeJacobian", bp::pure_virtual(compJac1),
           bp::args("self", "x", "Jout"))
      .def("getJacobian", compJac2, bp::args("self", "x"),
           "Compute and return Jacobian.");

  bp::register_ptr_to_python<shared_ptr<C2Function>>();
  bp::class_<C2FunctionWrap, bp::bases<C1Function>, boost::noncopyable>(
      "C2Function", "Base class for twice-differentiable functions.",
      bp::init<int, int, int>(bp::args("self", "nx", "ndx", "nr")))
      .def("vectorHessianProduct", &C2Function::vectorHessianProduct,
           &C2FunctionWrap::default_vhp, bp::args("self", "x", "v", "Hout"))
      .def("getVHP", &C2FunctionWrap::getVHP, bp::args("self", "x", "v"),
           "Compute and return the vector-Hessian product.")
      .def(
          "__matmul__",
          +[](shared_ptr<C2Function> const &left,
              shared_ptr<C2Function> const &right) {
            return ::proxnlp::compose<Scalar>(left, right);
          },
          "Composition operator. This composes the first argument over the "
          "second one.");

  exposeFunctionOps();
}

void exposeFunctionOps() {
  using ComposeFunction = ComposeFunctionTpl<Scalar>;

  const char *compose_doc = "Composition of two functions. This returns the "
                            "composition of `f` over `g`.";
  bp::register_ptr_to_python<shared_ptr<ComposeFunction>>();
  bp::class_<ComposeFunction, bp::bases<C2Function>>("ComposeFunction",
                                                     compose_doc, bp::no_init)
      .def(bp::init<const shared_ptr<C2Function> &,
                    const shared_ptr<C2Function> &>(bp::args("self", "f", "g")))
      .add_property("left",
                    bp::make_function(&ComposeFunction::left,
                                      bp::return_internal_reference<>()),
                    "The left-hand side of the composition.")
      .add_property("right",
                    bp::make_function(&ComposeFunction::right,
                                      bp::return_internal_reference<>()),
                    "The right-hand side of the composition.");

  bp::def("compose", &::proxnlp::compose<context::Scalar>, bp::args("f", "g"),
          compose_doc);
}

} // namespace python
} // namespace proxnlp
