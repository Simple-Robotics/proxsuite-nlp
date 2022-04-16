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
    bp::class_<T, shared_ptr<T>, bp::bases<context::Constraint>>(
      name, docstring,
      bp::init<const context::C2Function&>()
    );
  }

  /// @todo Expose properly using pure_virtual, to allow overriding from Python
  void exposeConstraints()
  {
    using context::Scalar;
    using context::Constraint;
    using ConstraintPtr = shared_ptr<Constraint>;
    bp::class_<Constraint, ConstraintPtr, boost::noncopyable>(
      "ConstraintSetBase", "Base class for constraint sets or nonsmooth penalties.",
      bp::no_init
    )
      .def("projection", &Constraint::projection, bp::args("self", "z"))
      .def("normal_cone_proj", &Constraint::normalConeProjection, bp::args("self", "z"))
      .def("apply_jacobian", &Constraint::applyProjectionJacobian, bp::args("self", "z", "Jout"), "Apply the projection Jacobian.")
      .def("apply_normal_jacobian", &Constraint::applyNormalConeProjectionJacobian, bp::args("self", "z", "Jout"), "Apply the normal cone projection Jacobian.")
      .def("compute_active_set", &Constraint::computeActiveSet, bp::args("self", "z", "out"))
      .add_property("nx",  &Constraint::nx)
      .add_property("ndx", &Constraint::ndx)
      .add_property("nr",  &Constraint::nr)
      .def(bp::self == bp::self)
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

