
#include "proxnlp/python/fwd.hpp"
#include "proxnlp/cost-function.hpp"
#include "proxnlp/cost-sum.hpp"

#include "proxnlp/modelling/costs/quadratic-residual.hpp"
#include "proxnlp/modelling/costs/squared-distance.hpp"

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

namespace internal {

struct CostWrapper : context::Cost, bp::wrapper<context::Cost> {
  PROXNLP_DYNAMIC_TYPEDEFS(context::Scalar);

  CostWrapper(const int nx, const int ndx) : context::Cost(nx, ndx) {}

  context::Scalar call(const ConstVectorRef &x) const {
    return get_override("call")(x);
  }
  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    get_override("computeGradient")(x, out);
  }
  void computeHessian(const ConstVectorRef &x, MatrixRef out) const {
    get_override("computeHessian")(x, out);
  }
};
} // namespace internal

void exposeCost() {
  using CostPtr = shared_ptr<context::Cost>;

  void (Cost::*compGrad1)(const ConstVectorRef &, VectorRef) const =
      &Cost::computeGradient;
  void (Cost::*compHess1)(const ConstVectorRef &, MatrixRef) const =
      &Cost::computeHessian;
  VectorXs (Cost::*compGrad2)(const ConstVectorRef &) const =
      &Cost::computeGradient;
  MatrixXs (Cost::*compHess2)(const ConstVectorRef &) const =
      &Cost::computeHessian;

  bp::class_<internal::CostWrapper, bp::bases<context::C2Function>,
             boost::noncopyable>(
      "CostFunctionBase", bp::init<int, int>(bp::args("self", "nx", "ndx")))
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
          "__mul__",
          +[](Scalar a, CostPtr const &b) {
            return a * b;
          }) // see cost_sum.hpp / returns CostSum<Scalar>
      ;

  bp::class_<func_to_cost<context::Scalar>, bp::bases<Cost>>(
      "CostFromFunction",
      "Wrap a scalar-values C2 function into a cost function.",
      bp::init<const context::C2Function &>(bp::args("self", "func")));

  using CostSum = CostSumTpl<context::Scalar>;
  bp::register_ptr_to_python<shared_ptr<CostSum>>();
  bp::class_<CostSum, bp::bases<Cost>>(
      "CostSum", "Sum of cost functions.",
      bp::init<int, int, const std::vector<CostSum::BasePtr> &,
               const std::vector<context::Scalar> &>(
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
      //   // printing
      .def(bp::self_ns::str(bp::self));

  /* Expose specific cost functions */

  bp::class_<QuadraticResidualCost<context::Scalar>, bp::bases<Cost>>(
      "QuadraticResidualCost", "Quadratic of a residual function",
      bp::init<const shared_ptr<context::C2Function> &, const ConstMatrixRef &,
               const ConstVectorRef &, context::Scalar>(
          (bp::arg("self"), bp::arg("residual"), bp::arg("weights"),
           bp::arg("slope"), bp::arg("constant") = 0.)))
      .def(bp::init<const shared_ptr<context::C2Function> &,
                    const ConstMatrixRef &, context::Scalar>(
          (bp::arg("self"), bp::arg("residual"), bp::arg("weights"),
           bp::arg("constant") = 0.)));

  bp::class_<QuadraticDistanceCost<context::Scalar>, bp::bases<Cost>>(
      "QuadraticDistanceCost",
      "Quadratic distance cost `(1/2)r.T * Q * r + b.T * r + c` on the "
      "manifold.",
      bp::init<const shared_ptr<Manifold> &, const VectorXs &,
               const MatrixXs &>(
          bp::args("self", "space", "target", "weights")))
      .def(bp::init<const shared_ptr<Manifold> &, const VectorXs &>(
          bp::args("self", "space", "target")))
      .def(bp::init<const shared_ptr<Manifold> &>(
          "Constructor which uses the neutral element of the space.",
          bp::args("self", "space")))
      .def("update_target",
           &QuadraticDistanceCost<context::Scalar>::updateTarget,
           bp::args("self", "new_target"));
}

} // namespace python
} // namespace proxnlp
