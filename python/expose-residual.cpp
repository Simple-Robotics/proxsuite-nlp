#include "proxnlp/python/fwd.hpp"

#include "proxnlp/modelling/residuals/linear.hpp"
#include "proxnlp/modelling/residuals/state-residual.hpp"

namespace proxnlp {
namespace python {

/// Expose a differentiable residual (subclass of C2FunctionTpl).
template <typename T, class Init>
bp::class_<T, bp::bases<context::C2Function>>
expose_function(const char *name, const char *docstring, Init init) {
  return bp::class_<T, bp::bases<context::C2Function>>(name, docstring, init);
}

/// Expose some residual functions
void exposeResiduals() {
  using context::ConstMatrixRef;
  using context::ConstVectorRef;
  using context::Manifold;
  using context::MatrixXs;
  using context::Scalar;
  using context::VectorXs;

  expose_function<LinearFunctionTpl<Scalar>>(
      "LinearFunction", "Residual f(x) = Ax + b.",
      bp::init<ConstMatrixRef, ConstVectorRef>(bp::args("self", "A", "b")))
      .def(bp::init<ConstMatrixRef>(bp::args("self", "A")));

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
}

} // namespace python
} // namespace proxnlp
