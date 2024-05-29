#include "proxsuite-nlp/python/residuals.hpp"

#include "proxsuite-nlp/modelling/residuals/linear.hpp"
#include "proxsuite-nlp/modelling/residuals/state-residual.hpp"

namespace proxsuite {
namespace nlp {
namespace python {
using context::C2Function;
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::Manifold;
using context::MatrixXs;
using context::Scalar;
using context::Vector3s;
using context::VectorXs;

/// Expose some residual functions
void exposeResiduals() {

  using LinearFunction = LinearFunctionTpl<Scalar>;

  expose_function<LinearFunction>(
      "LinearFunction", "Residual f(x) = Ax + b.",
      bp::init<MatrixXs, VectorXs>(bp::args("self", "A", "b")))
      .def(bp::init<MatrixXs>(bp::args("self", "A")))
      .def_readwrite("A", &LinearFunction::mat, "Matrix :math:`A`.")
      .def_readwrite("b", &LinearFunction::b, "Intercept :math:`b`.");

  expose_function<ManifoldDifferenceToPoint<Scalar>>(
      "ManifoldDifferenceToPoint", "Difference vector x (-) x0.",
      bp::init<const polymorphic<Manifold> &, const ConstVectorRef &>(
          bp::args("self", "space", "target")))
      .add_property("target", &ManifoldDifferenceToPoint<Scalar>::target_);

  expose_function<LinearFunctionDifferenceToPoint<Scalar>>(
      "LinearFunctionDifferenceToPoint",
      "Linear function of the vector difference to a reference point.",
      bp::init<const polymorphic<Manifold> &, VectorXs, MatrixXs, VectorXs>(
          bp::args("self", "space", "target", "A", "b")));
}

} // namespace python
} // namespace nlp
} // namespace proxsuite
