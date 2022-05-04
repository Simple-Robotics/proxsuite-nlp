#include "proxnlp/python/fwd.hpp"

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/modelling/autodiff/finite-difference.hpp"


namespace proxnlp
{
namespace python
{

  /// Expose finite difference helpers.
  void expose_finite_differences()
  {
    using namespace autodiff;
    using context::Scalar;
    using context::Manifold;
    using context::Function;
    using context::C1Function;
    using context::C2Function;

    bp::enum_<FDLevel>("FDLevel", "Finite difference level.")
      .value("ToC1", FDLevel::TOC1)
      .value("ToC2", FDLevel::TOC2)
      ;

    bp::class_<finite_difference_wrapper<Scalar, FDLevel::TOC1>,
               bp::bases<C1Function>>(
      "FiniteDifferenceHelper",
      "Make a function into a differentiable function using"
      " finite differences.",
      bp::init<const Manifold&,
               const Function&,
               const Scalar>(bp::args("self", "func", "eps"))
      )
      ;

    bp::class_<finite_difference_wrapper<Scalar, TOC2>,
               bp::bases<C2Function>>(
      "FiniteDifferenceHelperC2",
      "Make a differentiable function into a twice-differentiable function using"
      " finite differences.",
      bp::init<const Manifold&,
               const C1Function&,
               const Scalar>(bp::args("self", "func", "eps"))
      )
      ;

  }

  void exposeAutodiff()
  {
    expose_finite_differences();
  }
  
} // namespace python
} // namespace proxnlp

