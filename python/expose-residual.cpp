#include "lienlp/python/fwd.hpp"

#include "lienlp/modelling/residuals/linear.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

#include <boost/python/overloads.hpp>


namespace lienlp
{
namespace python
{

  /// Expose a differentiable residual (subclass of C2Function).
  template<typename T, class Init>
  bp::class_<T, bp::bases<context::C2Function_t>>
  expose_function(const char* name, const char* docstring, Init init)
  {
    return bp::class_<T, bp::bases<context::C2Function_t>>(
      name, docstring, init
    );
  }


  /// Expose some residual functions
  void exposeResiduals()
  {
    using context::VectorXs;
    using context::MatrixXs;
    using context::ConstVectorRef;
    using context::Manifold;

    expose_function<LinearFunction<context::Scalar>>(
      "LinearFunction", "Residual f(x) = Ax + b.",
      bp::init<MatrixXs, VectorXs>(bp::args("A", "b")));

    expose_function<StateResidual<context::Scalar>>(
      "StateResidual", "Difference vector x (-) x0.",
      bp::init<const Manifold&, const ConstVectorRef&>(bp::args("space", "target")));

    expose_function<LinearStateResidual<context::Scalar>>(
      "LinearStateResidual", "Linear function of the vector difference to a reference point.",
      bp::init<const Manifold&, VectorXs, MatrixXs, VectorXs>(bp::args("space", "target", "A", "b"))
    );
  }

} // namespace python
} // namespace lienlp
