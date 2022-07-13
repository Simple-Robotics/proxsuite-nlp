#include "proxnlp/python/fwd.hpp"
#include "proxnlp/constraint-base.hpp"

#include "proxnlp/modelling/constraints/equality-constraint.hpp"
#include "proxnlp/modelling/constraints/negative-orthant.hpp"

#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

namespace proxnlp {
namespace python {

template <typename T>
void exposeSpecificConstraintSet(const char *name, const char *docstring) {
  bp::class_<T, shared_ptr<T>, bp::bases<context::ConstraintSet>>(
      name, docstring, bp::init<>());
}

template <typename T>
context::Constraint make_constraint(const context::C2Function &f) {
  shared_ptr<context::ConstraintSet> s(new T());
  return context::Constraint(f, s);
}

/// @todo Expose properly using pure_virtual, to allow overriding from Python
void exposeConstraints() {
  using context::ConstraintSet;
  using context::Scalar;
  using ConstraintSetPtr = shared_ptr<ConstraintSet>;
  using context::Constraint;
  bp::class_<ConstraintSet, ConstraintSetPtr, boost::noncopyable>(
      "ConstraintSetBase",
      "Base class for constraint sets or nonsmooth penalties.", bp::no_init)
      .def("projection", &ConstraintSet::projection, bp::args("self", "z"))
      .def("normal_cone_proj", &ConstraintSet::normalConeProjection,
           bp::args("self", "z"))
      .def("apply_jacobian", &ConstraintSet::applyProjectionJacobian,
           bp::args("self", "z", "Jout"), "Apply the projection Jacobian.")
      .def("apply_normal_jacobian",
           &ConstraintSet::applyNormalConeProjectionJacobian,
           bp::args("self", "z", "Jout"),
           "Apply the normal cone projection Jacobian.")
      .def("compute_active_set", &ConstraintSet::computeActiveSet,
           bp::args("self", "z", "out"))
      .def(bp::self == bp::self);

  bp::class_<Constraint, shared_ptr<Constraint>>(
      "ConstraintObject", "Packs a constraint set together with a function.",
      bp::init<const context::C2Function &, ConstraintSetPtr>(
          bp::args("self", "func", "set")))
      .add_property("nr", &Constraint::nr, "Constraint dimension.")
      .def_readonly("set", &Constraint::m_set, "Constraint set.");

  /* Expose constraint stack */
  namespace pp = pinocchio::python;
  pp::StdVectorPythonVisitor<std::vector<shared_ptr<Constraint>>, true>::expose(
      "ConstraintVector");

  exposeSpecificConstraintSet<EqualityConstraint<Scalar>>(
      "EqualityConstraintSet", "Cast a function into an equality constraint");

  exposeSpecificConstraintSet<NegativeOrthant<Scalar>>(
      "NegativeOrthant",
      "Cast a function into a negative inequality constraint h(x) \\leq 0");

  bp::def("create_equality_constraint",
          &make_constraint<EqualityConstraint<Scalar>>,
          "Convenience function to create an equality constraint from a "
          "C2Function.");
  bp::def("create_inequality_constraint",
          &make_constraint<NegativeOrthant<Scalar>>,
          "Convenience function to create an inequality constraint from a "
          "C2Function.");

  bp::def("evaluateMoreauEnvelope",
          &evaluateMoreauEnvelope<Scalar>,
          bp::args("cstr_set", "zin", "zproj", "inv_mu"),
          "Evaluate the Moreau envelope with parameter :math:`\\mu`.");
}

} // namespace python
} // namespace proxnlp
