#include "proxnlp/python/fwd.hpp"
#include "proxnlp/constraint-base.hpp"

#include "proxnlp/modelling/constraints/equality-constraint.hpp"
#include "proxnlp/modelling/constraints/negative-orthant.hpp"

#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

namespace proxnlp {
namespace python {

namespace pp = pinocchio::python;

template <typename T>
void exposeSpecificConstraintSet(const char *name, const char *docstring) {
  bp::class_<T, shared_ptr<T>, bp::bases<context::ConstraintSet>>(
      name, docstring, bp::init<>(bp::args("self")));
}

template <typename T>
context::Constraint make_constraint(const shared_ptr<context::C2Function> &f) {
  return context::Constraint(f, std::make_shared<T>());
}

/// @todo Expose properly using pure_virtual, to allow overriding from Python
void exposeConstraints() {
  using context::C2Function;
  using context::Constraint;
  using context::ConstraintSet;
  using context::Scalar;
  using ConstraintSetPtr = shared_ptr<ConstraintSet>;

  bp::class_<ConstraintSet, boost::noncopyable>(
      "ConstraintSetBase",
      "Base class for constraint sets or nonsmooth penalties.", bp::no_init)
      .def("evaluate", &ConstraintSet::evaluate, bp::args("self", "z"),
           "Evaluate the constraint indicator function or nonsmooth penalty "
           "on the projection/prox map of :math:`z`.")
      .def("projection", &ConstraintSet::projection,
           bp::args("self", "z", "zout"))
      .def("normalConeProjection", &ConstraintSet::normalConeProjection,
           bp::args("self", "z", "zout"))
      .def("applyJacobian", &ConstraintSet::applyProjectionJacobian,
           bp::args("self", "z", "Jout"), "Apply the projection Jacobian.")
      .def("applyNormalJacobian",
           &ConstraintSet::applyNormalConeProjectionJacobian,
           bp::args("self", "z", "Jout"),
           "Apply the normal cone projection Jacobian.")
      .def("computeActiveSet", &ConstraintSet::computeActiveSet,
           bp::args("self", "z", "out"))
      .def(bp::self == bp::self);

  bp::class_<Constraint>(
      "ConstraintObject", "Packs a constraint set together with a function.",
      bp::init<shared_ptr<C2Function>, shared_ptr<ConstraintSet>>(
          bp::args("self", "func", "set")))
      .add_property("nr", &Constraint::nr, "Constraint dimension.")
      .def_readonly("func", &Constraint::func_, "Underlying function.")
      .def_readonly("set", &Constraint::set_, "Constraint set.");

  /* Expose constraint stack */
  pp::StdVectorPythonVisitor<std::vector<Constraint>, true>::expose(
      "StdVec_Constraint");

  exposeSpecificConstraintSet<EqualityConstraint<Scalar>>(
      "EqualityConstraintSet", "Cast a function into an equality constraint");

  exposeSpecificConstraintSet<NegativeOrthant<Scalar>>(
      "NegativeOrthant",
      "Cast a function into a negative inequality constraint h(x) \\leq 0");

  bp::def("createEqualityConstraint",
          &make_constraint<EqualityConstraint<Scalar>>,
          "Convenience function to create an equality constraint from a "
          "C2Function.");
  bp::def("createInequalityConstraint",
          &make_constraint<NegativeOrthant<Scalar>>,
          "Convenience function to create an inequality constraint from a "
          "C2Function.");

  bp::def("evaluateMoreauEnvelope", &evaluateMoreauEnvelope<Scalar>,
          bp::args("cstr_set", "zin", "zproj", "inv_mu"),
          "Evaluate the Moreau envelope with parameter :math:`\\mu`.");
}

} // namespace python
} // namespace proxnlp
