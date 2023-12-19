#include "proxsuite-nlp/python/fwd.hpp"

#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"
#include "proxsuite-nlp/modelling/costs/squared-distance.hpp"

namespace proxsuite {
namespace nlp {
namespace python {

using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::Cost;
using context::MatrixXs;
using context::Scalar;
using context::VectorXs;

void exposeQuadraticCosts() {
  using FunctionPtr = shared_ptr<context::C2Function>;
  using ManifoldPtr = shared_ptr<context::Manifold>;

  using QuadraticResidualCost = QuadraticResidualCostTpl<Scalar>;
  bp::class_<QuadraticResidualCost, bp::bases<Cost>>(
      "QuadraticResidualCost",
      "A cost which is a quadratic form :math:`\\frac 12 r(x)^\\top Wr(x) + "
      "b^\\top r(x) + c` of a residual function",
      bp::no_init)
      .def(bp::init<FunctionPtr, const ConstMatrixRef &, const ConstVectorRef &,
                    Scalar>((bp::arg("self"), bp::arg("residual"),
                             bp::arg("weights"), bp::arg("slope"),
                             bp::arg("constant") = 0.)))
      .def(bp::init<const shared_ptr<context::C2Function> &,
                    const ConstMatrixRef &, Scalar>(
          (bp::arg("self"), bp::arg("residual"), bp::arg("weights"),
           bp::arg("constant") = 0.)))
      .def_readonly("residual", &QuadraticResidualCost::residual_,
                    "The underlying function residual.")
      .def_readwrite(
          "gauss_newton", &QuadraticResidualCost::gauss_newton_,
          "Whether to use a Gauss-Newton approximation of the Hessian.");

  using QuadraticDistanceCost = QuadraticDistanceCostTpl<Scalar>;
  bp::class_<QuadraticDistanceCost, bp::bases<QuadraticResidualCost>>(
      "QuadraticDistanceCost",
      "Quadratic distance cost `(1/2)r.T * Q * r + b.T * r + c` on the "
      "manifold.",
      bp::init<ManifoldPtr, const VectorXs &, const MatrixXs &>(
          bp::args("self", "space", "target", "weights")))
      .def(bp::init<ManifoldPtr, const VectorXs &>(
          bp::args("self", "space", "target")))
      .def(bp::init<ManifoldPtr>(
          "Constructor which uses the neutral element of the space.",
          bp::args("self", "space")))
      .add_property("target", &QuadraticDistanceCost::getTarget,
                    &QuadraticDistanceCost::updateTarget);
}
} // namespace python
} // namespace nlp
} // namespace proxsuite
