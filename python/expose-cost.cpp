
#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/cost-function.hpp"
#include "proxsuite-nlp/cost-sum.hpp"

#include "boost/python/operators.hpp"

namespace proxnlp {
namespace python {

using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::Cost;
using context::Manifold;
using context::MatrixRef;
using context::MatrixXs;
using context::Scalar;
using context::VectorRef;
using context::VectorXs;

struct CostWrapper : Cost, bp::wrapper<Cost> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Cost::Cost;

  Scalar call(const ConstVectorRef &x) const { return get_override("call")(x); }
  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    get_override("computeGradient")(x, out);
  }
  void computeHessian(const ConstVectorRef &x, MatrixRef out) const {
    get_override("computeHessian")(x, out);
  }
};

/// Expose specific cost functions
void exposeQuadraticCosts();

void exposeCost() {
  using CostPtr = shared_ptr<Cost>;
  bp::register_ptr_to_python<CostPtr>();

  void (Cost::*compGrad1)(const ConstVectorRef &, VectorRef) const =
      &Cost::computeGradient;
  void (Cost::*compHess1)(const ConstVectorRef &, MatrixRef) const =
      &Cost::computeHessian;
  VectorXs (Cost::*compGrad2)(const ConstVectorRef &) const =
      &Cost::computeGradient;
  MatrixXs (Cost::*compHess2)(const ConstVectorRef &) const =
      &Cost::computeHessian;

  bp::class_<CostWrapper, bp::bases<context::C2Function>, boost::noncopyable>(
      "CostFunctionBase", bp::no_init)
      .def(bp::init<int, int>(bp::args("self", "nx", "ndx")))
      .def(bp::init<const Manifold &>(bp::args("self", "manifold")))
      .def("call", bp::pure_virtual(&Cost::call), bp::args("self", "x"))
      .def("computeGradient", bp::pure_virtual(compGrad1),
           bp::args("self", "x", "gout"))
      .def("computeGradient", compGrad2, bp::args("self", "x"),
           "Compute and return the gradient.")
      .def("computeHessian", bp::pure_virtual(compHess1),
           bp::args("self", "x", "Hout"))
      .def("computeHessian", compHess2, bp::args("self", "x"),
           "Compute and return the Hessian.")
      // define non-member operators
      .def(
          "__add__",
          +[](CostPtr const &a, CostPtr const &b) {
            return a + b;
          }) // see cost_sum.hpp / returns CostSum<Scalar>
      .def(
          "__mul__", +[](CostPtr const &self, Scalar a) { return a * self; })
      .def(
          "__rmul__", +[](CostPtr const &self, Scalar a) { return a * self; })
      // see cost_sum.hpp / returns CostSum<Scalar>
      ;

  bp::class_<func_to_cost<Scalar>, bp::bases<Cost>>(
      "CostFromFunction",
      "Wrap a scalar-values C2 function into a cost function.", bp::no_init)
      .def(bp::init<const shared_ptr<context::C2Function> &>(
          bp::args("self", "func")));

  using CostSum = CostSumTpl<Scalar>;
  bp::register_ptr_to_python<shared_ptr<CostSum>>();
  bp::class_<CostSum, bp::bases<Cost>>(
      "CostSum", "Sum of cost functions.",
      bp::init<int, int, const std::vector<CostSum::BasePtr> &,
               const std::vector<Scalar> &>(
          bp::args("self", "nx", "ndx", "components", "weights")))
      .def(bp::init<int, int>(bp::args("self", "nx", "ndx")))
      .add_property("num_components", &CostSum::numComponents,
                    "Number of components.")
      .def_readonly("weights", &CostSum::weights_)
      .def("add_component", &CostSum::addComponent,
           ((bp::arg("self"), bp::arg("comp"), bp::arg("w") = 1.)),
           "Add a component to the cost.")
      // expose inplace operators
      .def(
          "__iadd__", +[](CostSum &a, CostSum const &b) { return a += b; })
      .def(
          "__iadd__", +[](CostSum &a, CostPtr const &b) { return a += b; })
      .def(
          "__imul__", +[](CostSum &a, Scalar b) { return a *= b; })
      // printing
      .def(bp::self * Scalar())
      .def(Scalar() * bp::self)
      .def(-bp::self)
      .def(bp::self_ns::str(bp::self));

  exposeQuadraticCosts();
}

} // namespace python
} // namespace proxnlp
