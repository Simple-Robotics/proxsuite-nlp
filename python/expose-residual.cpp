#include "lienlp/python/fwd.hpp"

#include "lienlp/modelling/residuals/linear.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

#include <boost/python/overloads.hpp>


namespace lienlp
{
namespace python
{
  namespace bp = boost::python;


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
    using context::ManifoldAbstract_t;

    expose_residual<LinearResidual<context::Scalar>>(
      "LinearResidual", "Residual f(x) = Ax + b.",
      bp::init<MatrixXs, VectorXs>(bp::args("A", "b")));

    expose_residual<StateResidual<context::Scalar>>(
      "StateResidual", "Difference vector x (-) x0.",
      bp::init<const ManifoldAbstract_t&, const ConstVectorRef&>(bp::args("space", "target")));
  }

} // namespace python
} // namespace lienlp
