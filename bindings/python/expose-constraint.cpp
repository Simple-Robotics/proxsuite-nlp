/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxsuite-nlp/modelling/constraints.hpp"

#include "proxsuite-nlp/python/polymorphic.hpp"

#include <eigenpy/std-vector.hpp>

namespace proxsuite {
namespace nlp {
namespace python {

using context::C2Function;
using context::Constraint;
using context::ConstraintSet;
using context::ConstVectorRef;
using context::Scalar;
using eigenpy::StdVectorPythonVisitor;
using L1Penalty = NonsmoothPenaltyL1Tpl<Scalar>;
using ConstraintSetProduct = ConstraintSetProductTpl<Scalar>;
using BoxConstraint = BoxConstraintTpl<Scalar>;

template <typename T>
auto exposeSpecificConstraintSet(const char *name, const char *docstring) {
  return bp::class_<T, bp::bases<ConstraintSet>>(name, docstring, bp::no_init)
      .def(PolymorphicVisitor<polymorphic<ConstraintSet>>());
}

template <typename ConstraintType>
context::Constraint make_constraint(const shared_ptr<context::C2Function> &f) {
  return context::Constraint(f, ConstraintType{});
}

static void exposeConstraintTypes();

/// @todo Expose properly using pure_virtual, to allow overriding from Python
void exposeConstraints() {

  register_polymorphic_to_python<polymorphic<ConstraintSet>>();
  bp::class_<ConstraintSet, boost::noncopyable>(
      "ConstraintSetBase",
      "Base class for constraint sets or nonsmooth penalties.", bp::no_init)
      .def("evaluate", &ConstraintSet::evaluate, ("self"_a, "z"),
           "Evaluate the constraint indicator function or nonsmooth penalty "
           "on the projection/prox map of :math:`z`.")
      .def("projection", &ConstraintSet::projection, ("self"_a, "z", "zout"))
      .def(
          "projection",
          +[](const ConstraintSet &c, const ConstVectorRef &z) {
            context::VectorXs zout(z.size());
            c.projection(z, zout);
            return zout;
          },
          ("self"_a, "z"))
      .def("normalConeProjection", &ConstraintSet::normalConeProjection,
           ("self"_a, "z", "zout"))
      .def(
          "normalConeProjection",
          +[](const ConstraintSet &c, const ConstVectorRef &z) {
            context::VectorXs zout(z.size());
            c.normalConeProjection(z, zout);
            return zout;
          },
          ("self"_a, "z"))
      .def("applyProjectionJacobian", &ConstraintSet::applyProjectionJacobian,
           ("self"_a, "z", "Jout"), "Apply the projection Jacobian.")
      .def("applyNormalProjectionJacobian",
           &ConstraintSet::applyNormalConeProjectionJacobian,
           ("self"_a, "z", "Jout"),
           "Apply the normal cone projection Jacobian.")
      .def("computeActiveSet", &ConstraintSet::computeActiveSet,
           ("self"_a, "z", "out"))
      .def("evaluateMoreauEnvelope", &ConstraintSet::evaluateMoreauEnvelope,
           ("self"_a, "zin", "zproj"),
           "Evaluate the Moreau envelope with parameter :math:`\\mu`.")
      .def("setProxParameter", &ConstraintSet::setProxParameter,
           ("self"_a, "mu"), "Set proximal parameter.")
      .add_property("mu", &ConstraintSet::mu, "Current proximal parameter.")
      .def(bp::self == bp::self);

  bp::class_<Constraint>(
      "ConstraintObject", "Packs a constraint set together with a function.",
      bp::init<shared_ptr<C2Function>, const polymorphic<ConstraintSet> &>(
          ("self"_a, "func",
           "constraint_set"))[bp::with_custodian_and_ward<1, 3>()])
      .add_property("nr", &Constraint::nr, "Constraint dimension.")
      .def_readonly("func", &Constraint::func_, "Underlying function.")
      .def_readonly("set", &Constraint::set_, "Constraint set.");

  StdVectorPythonVisitor<std::vector<context::Constraint>, true>::expose(
      "StdVec_ConstraintObject");

  exposeConstraintTypes();
}

static void exposeConstraintTypes() {
  exposeSpecificConstraintSet<EqualityConstraintTpl<Scalar>>(
      "EqualityConstraintSet", "Cast a function into an equality constraint")
      .def(bp::init<>("self"_a));

  exposeSpecificConstraintSet<NegativeOrthantTpl<Scalar>>(
      "NegativeOrthant",
      "Cast a function into a negative inequality constraint h(x) \\leq 0")
      .def(bp::init<>("self"_a));

  exposeSpecificConstraintSet<BoxConstraint>(
      "BoxConstraint",
      "Box constraint of the form :math:`z \\in [z_\\min, z_\\max]`.")
      .def(bp::init<context::ConstVectorRef, context::ConstVectorRef>(
          ("self"_a, "lower_limit", "upper_limit")))
      .def_readwrite("upper_limit", &BoxConstraint::upper_limit)
      .def_readwrite("lower_limit", &BoxConstraint::lower_limit);

  bp::def("createEqualityConstraint",
          &make_constraint<EqualityConstraintTpl<Scalar>>,
          "Convenience function to create an equality constraint from a "
          "C2Function.");
  bp::def("createInequalityConstraint",
          &make_constraint<NegativeOrthantTpl<Scalar>>,
          "Convenience function to create an inequality constraint from a "
          "C2Function.");

  exposeSpecificConstraintSet<L1Penalty>("NonsmoothPenaltyL1",
                                         "1-norm penalty function.")
      .def(bp::init<>(("self"_a)));

  exposeSpecificConstraintSet<ConstraintSetProduct>(
      "ConstraintSetProduct", "Cartesian product of constraint sets.")
      .def(bp::init<std::vector<polymorphic<ConstraintSet>>,
                    std::vector<Eigen::Index>>(
          ("self"_a, "components",
           "blockSizes"))[with_custodian_and_ward_list_content<1, 2>()])
      .add_property("components",
                    bp::make_function(&ConstraintSetProduct::components,
                                      bp::return_internal_reference<>()))
      .add_property("blockSizes",
                    bp::make_function(&ConstraintSetProduct::blockSizes,
                                      bp::return_internal_reference<>()),
                    "Dimensions of each component of the cartesian product.");
}

} // namespace python
} // namespace nlp
} // namespace proxsuite
