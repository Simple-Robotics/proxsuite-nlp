#include "proxnlp/python/fwd.hpp"

#include "proxnlp/modelling/residuals/linear.hpp"
#include "proxnlp/modelling/residuals/state-residual.hpp"

#include <boost/python/overloads.hpp>


namespace proxnlp
{
namespace python
{

  /// Expose a differentiable residual (subclass of C2FunctionTpl).
  template<typename T, class Init>
  bp::class_<T, bp::bases<context::C2Function>>
  expose_function(const char* name, const char* docstring, Init init)
  {
    return bp::class_<T, bp::bases<context::C2Function>>(
      name, docstring, init
    );
  }


  /// Expose some residual functions
  void exposeResiduals()
  {
    using context::Scalar;
    using context::VectorXs;
    using context::MatrixXs;
    using context::ConstVectorRef;
    using context::Manifold;

    expose_function<LinearFunction<Scalar>>(
      "LinearFunction", "Residual f(x) = Ax + b.",
      bp::init<MatrixXs, VectorXs>(bp::args("A", "b")));

    expose_function<ManifoldDifferenceToPoint<Scalar>>(
      "ManifoldDifferenceToPoint", "Difference vector x (-) x0.",
      bp::init<const Manifold&, const ConstVectorRef&>(bp::args("space", "target")))
      .add_property("target", &ManifoldDifferenceToPoint<Scalar>::m_target)
      ;

    expose_function<LinearFunctionDifferenceToPoint<Scalar>>(
      "LinearFunctionDifferenceToPoint", "Linear function of the vector difference to a reference point.",
      bp::init<const Manifold&, VectorXs, MatrixXs, VectorXs>(bp::args("space", "target", "A", "b"))
    );
  }

} // namespace python
} // namespace proxnlp
