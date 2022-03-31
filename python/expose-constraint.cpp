#include "lienlp/python/fwd.hpp"
#include "lienlp/constraint-base.hpp"

#include "lienlp/modelling/constraints/equality-constraint.hpp"
#include "lienlp/modelling/constraints/negative-orthant.hpp"

/// TODO remove this include once functionality upstream
#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>


namespace lienlp
{
namespace python
{

  template<typename T>
  void exposeSpecificConstraint(const char* name, const char* docstring)
  {
    bp::class_<T, shared_ptr<T>, bp::bases<context::Constraint_t>>(
      name, docstring,
      bp::init<const context::C2Function_t&>()
    );
  }

  void exposeConstraints()
  {
    using context::Scalar;
    using context::Constraint_t;
    using ConstraintPtr = shared_ptr<Constraint_t>;
    bp::class_<Constraint_t, ConstraintPtr, boost::noncopyable>(
      "ConstraintSetBase", "Base class for constraint sets.",
      bp::no_init
    )
      .def("projection", &Constraint_t::projection, bp::args("self", "z"))
      .def("normalConeProjection", &Constraint_t::normalConeProjection, bp::args("self", "z"))
      .def("computeActiveSet", &Constraint_t::computeActiveSet, bp::args("self", "z", "out"))
      ;

    /* Expose constraint stack */
    namespace pp = pinocchio::python;
    pp::StdVectorPythonVisitor<std::vector<ConstraintPtr>, true>::expose("ConstraintVector");

    exposeSpecificConstraint<EqualityConstraint<Scalar>>(
      "EqualityConstraint",
      "Cast  function into an equality constraint");

    exposeSpecificConstraint<NegativeOrthant<Scalar>>(
      "NegativeOrthant",
      "Cast a function into a negative inequality constraint h(x) \\leq 0");
  }

}
} // namespace lienlp

