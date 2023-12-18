/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxsuite-nlp/python/fwd.hpp"

#include "proxsuite-nlp/modelling/constraints/equality-constraint.hpp"
#include "proxsuite-nlp/modelling/constraints/negative-orthant.hpp"
#include "proxsuite-nlp/modelling/constraints/box-constraint.hpp"
#include "proxsuite-nlp/modelling/constraints/l1-penalty.hpp"

#include <eigenpy/std-vector.hpp>

namespace proxnlp {
namespace python {

using context::C2Function;
using context::Constraint;
using context::ConstraintSet;
using context::ConstVectorRef;
using context::Scalar;
using eigenpy::StdVectorPythonVisitor;

template <typename T, typename Init>
auto exposeSpecificConstraintSet(const char *name, const char *docstring,
                                 Init &&init) {
  return bp::class_<T, bp::bases<context::ConstraintSet>>(name, docstring,
                                                          init);
}

template <typename T>
auto exposeSpecificConstraintSet(const char *name, const char *docstring) {
  // bp::class_<T, bp::bases<context::ConstraintSet>>(
  //     name, docstring, bp::init<>(bp::args("self")));
  return exposeSpecificConstraintSet<T>(name, docstring,
                                        bp::init<>(bp::args("self")));
}

template <typename T>
context::Constraint make_constraint(const shared_ptr<context::C2Function> &f) {
  return context::Constraint(f, std::make_shared<T>());
}

static void exposeConstraintTypes();

/// @todo Expose properly using pure_virtual, to allow overriding from Python
void exposeConstraints() {

  bp::register_ptr_to_python<shared_ptr<ConstraintSet>>();
  bp::class_<ConstraintSet, boost::noncopyable>(
      "ConstraintSetBase",
      "Base class for constraint sets or nonsmooth penalties.", bp::no_init)
      .def("evaluate", &ConstraintSet::evaluate, bp::args("self", "z"),
           "Evaluate the constraint indicator function or nonsmooth penalty "
           "on the projection/prox map of :math:`z`.")
      .def("projection", &ConstraintSet::projection,
           bp::args("self", "z", "zout"))
      .def(
          "projection",
          +[](const ConstraintSet &c, const ConstVectorRef &z) {
            context::VectorXs zout(z.size());
            c.projection(z, zout);
            return zout;
          },
          bp::args("self", "z"))
      .def("normalConeProjection", &ConstraintSet::normalConeProjection,
           bp::args("self", "z", "zout"))
      .def(
          "normalConeProjection",
          +[](const ConstraintSet &c, const ConstVectorRef &z) {
            context::VectorXs zout(z.size());
            c.normalConeProjection(z, zout);
            return zout;
          },
          bp::args("self", "z"))
      .def("applyProjectionJacobian", &ConstraintSet::applyProjectionJacobian,
           bp::args("self", "z", "Jout"), "Apply the projection Jacobian.")
      .def("applyNormalProjectionJacobian",
           &ConstraintSet::applyNormalConeProjectionJacobian,
           bp::args("self", "z", "Jout"),
           "Apply the normal cone projection Jacobian.")
      .def("computeActiveSet", &ConstraintSet::computeActiveSet,
           bp::args("self", "z", "out"))
      .def("evaluateMoreauEnvelope", &ConstraintSet::evaluateMoreauEnvelope,
           bp::args("self", "zin", "zproj"),
           "Evaluate the Moreau envelope with parameter :math:`\\mu`.")
      .def("setProxParameter", &ConstraintSet::setProxParameter,
           bp::args("self", "mu"), "Set proximal parameter.")
      .add_property("mu", &ConstraintSet::mu, "Current proximal parameter.")
      .def(bp::self == bp::self);

  bp::class_<Constraint>(
      "ConstraintObject", "Packs a constraint set together with a function.",
      bp::init<shared_ptr<C2Function>, shared_ptr<ConstraintSet>>(
          bp::args("self", "func", "set")))
      .add_property("nr", &Constraint::nr, "Constraint dimension.")
      .def_readonly("func", &Constraint::func_, "Underlying function.")
      .def_readonly("set", &Constraint::set_, "Constraint set.");

  StdVectorPythonVisitor<std::vector<context::Constraint>, true>::expose(
      "StdVec_ConstraintObject");

  exposeConstraintTypes();
}

static void exposeConstraintTypes() {
  exposeSpecificConstraintSet<EqualityConstraint<Scalar>>(
      "EqualityConstraintSet", "Cast a function into an equality constraint");

  exposeSpecificConstraintSet<NegativeOrthant<Scalar>>(
      "NegativeOrthant",
      "Cast a function into a negative inequality constraint h(x) \\leq 0");

  exposeSpecificConstraintSet<BoxConstraintTpl<Scalar>>(
      "BoxConstraint",
      "Box constraint of the form :math:`z \\in [z_\\min, z_\\max]`.",
      bp::init<context::ConstVectorRef, context::ConstVectorRef>(
          bp::args("self", "lower_limit", "upper_limit")))
      .def_readwrite("upper_limit", &BoxConstraintTpl<Scalar>::upper_limit)
      .def_readwrite("lower_limit", &BoxConstraintTpl<Scalar>::lower_limit);

  bp::def("createEqualityConstraint",
          &make_constraint<EqualityConstraint<Scalar>>,
          "Convenience function to create an equality constraint from a "
          "C2Function.");
  bp::def("createInequalityConstraint",
          &make_constraint<NegativeOrthant<Scalar>>,
          "Convenience function to create an inequality constraint from a "
          "C2Function.");

  using L1Penalty_t = NonsmoothPenaltyL1Tpl<Scalar>;
  exposeSpecificConstraintSet<L1Penalty_t>("NonsmoothPenaltyL1",
                                           "1-norm penalty function.");
}

} // namespace python
} // namespace proxnlp
