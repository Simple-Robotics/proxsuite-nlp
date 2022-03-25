#include "lienlp/python/fwd.hpp"
#include "lienlp/constraint-base.hpp"

#include "lienlp/modelling/constraints/equality-constraint.hpp"
#include "lienlp/modelling/constraints/negative-orthant.hpp"


namespace lienlp
{
namespace python
{

  namespace bp = boost::python;

  void exposeConstraint()
  {
    using context::Scalar;
    using context::Constraint_t;
    bp::class_<Constraint_t, boost::noncopyable>(
      "ConstraintSetBase", "Base class for constraint sets.",
      bp::no_init
    )
      .def("projection", &Constraint_t::projection, bp::args("self", "z"))
      .def("normalConeProjection", &Constraint_t::normalConeProjection, bp::args("self", "z"))
      .def("computeActiveSet", &Constraint_t::computeActiveSet, bp::args("self", "z", "out"))
      ;

    using Equality_t = EqualityConstraint<Scalar>;
    bp::class_<Equality_t, bp::bases<Constraint_t>>(
      "EqualityConstraint",
      "Cast a residual into an equality constraint",
      bp::init<const context::Residual_t&>()
    )
      ;

    using Inequality_t = NegativeOrthant<Scalar>;
    bp::class_<Inequality_t, bp::bases<Constraint_t>>(
      "NegativeOrthant",
      "Cast a residual into a negative inequality constraint h(x) \\leq 0",
      bp::init<const context::Residual_t&>()
    )
      ;
  }

}
} // namespace lienlp

