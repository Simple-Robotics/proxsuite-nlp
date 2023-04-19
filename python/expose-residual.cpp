#include "proxnlp/python/fwd.hpp"

#include "proxnlp/modelling/residuals/linear.hpp"
#include "proxnlp/modelling/residuals/state-residual.hpp"
#include "proxnlp/modelling/residuals/rigid-transform-point.hpp"

namespace proxnlp {
namespace python {
using context::C2Function;
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::Manifold;
using context::MatrixXs;
using context::Scalar;
using context::Vector3s;
using context::VectorXs;

/// Expose a differentiable residual (subclass of C2FunctionTpl).
template <typename T, class Init>
auto expose_function(const char *name, const char *docstring, Init init) {
  return bp::class_<T, bp::bases<C2Function>>(name, docstring, init);
}

/// Expose some residual functions
void exposeResiduals() {

  using LinearFunction = LinearFunctionTpl<Scalar>;

  expose_function<LinearFunction>(
      "LinearFunction", "Residual f(x) = Ax + b.",
      bp::init<ConstMatrixRef, ConstVectorRef>(bp::args("self", "A", "b")))
      .def(bp::init<ConstMatrixRef>(bp::args("self", "A")))
      .def_readwrite("A", &LinearFunction::mat, "Matrix :math:`A`.")
      .def_readwrite("b", &LinearFunction::b, "Intercept :math:`b`.");

  expose_function<ManifoldDifferenceToPoint<Scalar>>(
      "ManifoldDifferenceToPoint", "Difference vector x (-) x0.",
      bp::init<const shared_ptr<Manifold> &, const ConstVectorRef &>(
          bp::args("self", "space", "target")))
      .add_property("target", &ManifoldDifferenceToPoint<Scalar>::target_);

  expose_function<LinearFunctionDifferenceToPoint<Scalar>>(
      "LinearFunctionDifferenceToPoint",
      "Linear function of the vector difference to a reference point.",
      bp::init<const shared_ptr<Manifold> &, VectorXs, MatrixXs, VectorXs>(
          bp::args("self", "space", "target", "A", "b")));

  using RigidTransformPointAction = RigidTransformationPointActionTpl<Scalar>;
  expose_function<RigidTransformPointAction>(
      "RigidTransformationPointAction",
      "A residual representing the action :math:`M\\cdot p = Rp + t` of a "
      "rigid "
      "transform :math:`M` on a 3D point :math:`p`.",
      bp::init<context::Vector3s>(bp::args("self", "point")))
      .def_readwrite("point", &RigidTransformPointAction::point_)
      .add_property("skew_matrix", &RigidTransformPointAction::skew_point);
}

} // namespace python
} // namespace proxnlp
