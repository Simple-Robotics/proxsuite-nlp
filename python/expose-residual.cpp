#include "lienlp/python/fwd.hpp"

#include "lienlp/modelling/residuals/linear.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

#include <boost/python/overloads.hpp>


namespace lienlp
{
namespace python
{

  /// Expose a differentiable residual (subclass of DifferentiableFunctor).
  template<typename T, class Init>
  bp::class_<T, bp::bases<context::DFunctor_t>>
  expose_residual(const char* name, const char* docstring, Init init)
  {
    return bp::class_<T, bp::bases<context::DFunctor_t>>(
      name, docstring, init
    );
  }


  /// Expose some residual functions
  void exposeResiduals()
  {
    using context::VectorXs;
    using context::MatrixXs;
    using context::ConstVectorRef;
    using context::ManifoldType;

    expose_residual<LinearResidual<context::Scalar>>(
      "LinearResidual", "Residual f(x) = Ax + b.",
      bp::init<MatrixXs, VectorXs>(bp::args("A", "b")));

    expose_residual<StateResidual<context::Scalar>>(
      "StateResidual", "Difference vector x (-) x0.",
      bp::init<const ManifoldType&, const ConstVectorRef&>(bp::args("space", "target")));

    expose_residual<LinearStateResidual<context::Scalar>>(
      "LinearStateResidual", "Linear function of the vector difference to a reference point.",
      bp::init<const ManifoldType&, VectorXs, MatrixXs, VectorXs>(bp::args("space", "target", "A", "b"))
    );
  }

} // namespace python
} // namespace lienlp
