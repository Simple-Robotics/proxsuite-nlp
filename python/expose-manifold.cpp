#include "proxnlp/python/manifold.hpp"

#include "proxnlp/modelling/spaces/vector-space.hpp"
#include "proxnlp/modelling/spaces/tangent-bundle.hpp"
#ifdef PROXNLP_WITH_PINOCCHIO
#include "proxnlp/modelling/spaces/pinocchio-groups.hpp"
#include "proxnlp/modelling/spaces/multibody.hpp"
#endif

namespace proxnlp {
namespace python {

namespace internal {
void exposeManifoldBase() {
  using context::ConstVectorRef;
  using context::Manifold;
  using context::MatrixRef;
  using context::MatrixXs;
  using context::Scalar;
  using context::VectorRef;
  using context::VectorXs;

  using BinaryFunTypeRet = VectorXs (Manifold::*)(const ConstVectorRef &,
                                                  const ConstVectorRef &) const;
  using BinaryFunType = void (Manifold::*)(
      const ConstVectorRef &, const ConstVectorRef &, VectorRef) const;
  using JacobianFunType = void (Manifold::*)(
      const ConstVectorRef &, const ConstVectorRef &, MatrixRef, int) const;

  bp::class_<Manifold, shared_ptr<Manifold>, boost::noncopyable>(
      "ManifoldAbstract", "Manifold abstract class.", bp::no_init)
      .add_property("nx", &Manifold::nx, "Manifold representation dimension.")
      .add_property("ndx", &Manifold::ndx, "Tangent space dimension.")
      .def("neutral", &Manifold::neutral, bp::args("self"),
           "Get the neutral point from the manifold (if a Lie group).")
      .def("rand", &Manifold::rand, bp::args("self"),
           "Sample a random point from the manifold.")
      .def("integrate", static_cast<BinaryFunType>(&Manifold::integrate),
           bp::args("self", "x", "v", "out"))
      .def("difference", static_cast<BinaryFunType>(&Manifold::difference),
           bp::args("self", "x0", "x1", "out"))
      .def("integrate", static_cast<BinaryFunTypeRet>(&Manifold::integrate),
           bp::args("self", "x", "v"))
      .def("difference", static_cast<BinaryFunTypeRet>(&Manifold::difference),
           bp::args("self", "x0", "x1"))
      .def("interpolate",
           (void(Manifold::*)(const ConstVectorRef &, const ConstVectorRef &,
                              const Scalar &, VectorRef)
                const)(&Manifold::interpolate),
           bp::args("self", "x0", "x1", "u", "out"))
      .def(
          "interpolate",
          (VectorXs(Manifold::*)(const ConstVectorRef &, const ConstVectorRef &,
                                 const Scalar &) const)(&Manifold::interpolate),
          bp::args("self", "x0", "x1", "u"),
          "Interpolate between two points on the manifold. Allocated version.")
      .def("Jintegrate", static_cast<JacobianFunType>(&Manifold::Jintegrate),
           bp::args("self", "x", "v", "Jout", "arg"),
           "Compute the Jacobian of the exp operator.")
      .def("Jdifference", static_cast<JacobianFunType>(&Manifold::Jdifference),
           bp::args("self", "x0", "x1", "Jout", "arg"),
           "Compute the Jacobian of the log operator.")
      .def(
          "Jintegrate",
          +[](const Manifold &m, const ConstVectorRef x,
              const ConstVectorRef &v, int arg) {
            MatrixXs Jout(m.ndx(), m.ndx());
            m.Jintegrate(x, v, Jout, arg);
            return Jout;
          },
          "Compute and return the Jacobian of the exp.")
      .def("JintegrateTransport", &Manifold::JintegrateTransport,
           bp::args("self", "x", "v", "J", "arg"),
           "Perform parallel transport of matrix J expressed at point x+v to "
           "point x.")
      .def(
          "Jdifference",
          +[](const Manifold &m, const ConstVectorRef x0,
              const ConstVectorRef &x1, int arg) {
            MatrixXs Jout(m.ndx(), m.ndx());
            m.Jdifference(x0, x1, Jout, arg);
            return Jout;
          },
          "Compute and return the Jacobian of the log.")
      .def("tangent_space", &Manifold::tangentSpace, bp::args("self"),
           "Returns an object representing the tangent space to this manifold.")
      .def(
          "__mul__", +[](const shared_ptr<Manifold> &a,
                         const shared_ptr<Manifold> &b) { return a * b; })
      .def(
          "__mul__",
          +[](const shared_ptr<Manifold> &a,
              const CartesianProductTpl<Scalar> &b) { return a * b; })
      .def(
          "__rmul__",
          +[](const shared_ptr<Manifold> &a,
              const CartesianProductTpl<Scalar> &b) { return a * b; });
}

} // namespace internal

/// Expose the tangent bundle of a manifold type @p M.
template <typename M>
bp::class_<TangentBundleTpl<M>, bp::bases<context::Manifold>>
exposeTangentBundle(const char *name, const char *docstring) {
  using OutType = TangentBundleTpl<M>;
  return bp::class_<OutType, bp::bases<context::Manifold>>(
             name, docstring, bp::init<M>(bp::args("self", "base")))
      .add_property(
          "base",
          bp::make_function(
              &OutType::getBaseSpace,
              bp::return_value_policy<bp::reference_existing_object>()),
          "Get the base space.");
}

/// Expose the tangent bundle with an additional constructor.
template <typename M, class Init>
bp::class_<TangentBundleTpl<M>, bp::bases<context::Manifold>>
exposeTangentBundle(const char *name, const char *docstring, Init init) {
  return exposeTangentBundle<M>(name, docstring).def(init);
}

#ifdef PROXNLP_WITH_PINOCCHIO
/// Expose a Pinocchio Lie group with a specified name, docstring,
/// and no-arg default constructor.
template <typename LieGroup>
void exposeLieGroup(const char *name, const char *docstring) {
  using BoundType = PinocchioLieGroup<LieGroup>;
  bp::class_<BoundType, bp::bases<context::Manifold>>(
      name, docstring, bp::init<>(bp::args("self")));
}
#endif

void exposeManifolds() {
  using context::Manifold;
  using context::Scalar;

  internal::exposeManifoldBase();

  /* Basic vector space */
  bp::class_<VectorSpaceTpl<Scalar>, bp::bases<Manifold>>(
      "VectorSpace", "Basic Euclidean vector space.",
      bp::init<const int>(bp::args("self", "dim")));

  bp::class_<CartesianProductTpl<Scalar>, bp::bases<Manifold>>(
      "CartesianProduct", "Cartesian product of two or more manifolds.",
      bp::init<const std::vector<shared_ptr<Manifold>> &>(
          bp::args("self", "spaces")))
      .def(bp::init<shared_ptr<Manifold>, shared_ptr<Manifold>>(
          bp::args("self", "left", "right")))
      .def("getComponent", &CartesianProductTpl<Scalar>::getComponent,
           bp::return_internal_reference<>(), bp::args("self", "i"),
           "Get the i-th component of the Cartesian product.")
      .def("addComponent", &CartesianProductTpl<Scalar>::addComponent,
           bp::args("self", "c"), "Add a component to the Cartesian product.")
      .add_property("num_components",
                    &CartesianProductTpl<Scalar>::numComponents,
                    "Get the number of components in the Cartesian product.")
      .def("split", &CartesianProductTpl<Scalar>::split, bp::args("self", "x"),
           "Takes an point on the product manifold and splits it up between "
           "the two base manifolds.")
      .def("split_vector", &CartesianProductTpl<Scalar>::split_vector,
           bp::args("self", "v"),
           "Takes a tangent vector on the product manifold and splits it up.")
      .def("merge", &CartesianProductTpl<Scalar>::merge, bp::args("self", "xs"),
           "Define a point on the manifold by merging two points of the "
           "component manifolds.")
      .def("merge_vector", &CartesianProductTpl<Scalar>::merge_vector,
           bp::args("self", "vs"),
           "Define a tangent vector on the manifold by merging two points of "
           "the component manifolds.")
      .def(
          "__mul__", +[](const CartesianProductTpl<Scalar> &a,
                         const shared_ptr<Manifold> &b) { return a * b; });

#ifdef PROXNLP_WITH_PINOCCHIO
  namespace pin = pinocchio;
  using VectorSpace = pin::VectorSpaceOperationTpl<Eigen::Dynamic, Scalar>;
  bp::class_<PinocchioLieGroup<VectorSpace>, bp::bases<Manifold>>(
      "EuclideanSpace", "Pinocchio's n-dimensional Euclidean vector space.",
      bp::init<int>(bp::args("self", "dim")));

  exposeLieGroup<pin::VectorSpaceOperationTpl<1, Scalar>>(
      "R", "One-dimensional Euclidean space AKA real number line.");
  exposeLieGroup<pin::VectorSpaceOperationTpl<2, Scalar>>(
      "R2", "Two-dimensional Euclidean space.");
  exposeLieGroup<pin::VectorSpaceOperationTpl<3, Scalar>>(
      "R3", "Three-dimensional Euclidean space.");
  exposeLieGroup<pin::VectorSpaceOperationTpl<4, Scalar>>(
      "R4", "Four-dimensional Euclidean space.");
  exposeLieGroup<pin::SpecialOrthogonalOperationTpl<2, Scalar>>(
      "SO2", "SO(2) special orthogonal group.");
  exposeLieGroup<pin::SpecialOrthogonalOperationTpl<3, Scalar>>(
      "SO3", "SO(3) special orthogonal group.");
  exposeLieGroup<pin::SpecialEuclideanOperationTpl<2, Scalar>>(
      "SE2", "SE(2) special Euclidean group.");
  exposeLieGroup<pin::SpecialEuclideanOperationTpl<3, Scalar>>(
      "SE3", "SE(3) special Euclidean group.");

  using SO2 = PinocchioLieGroup<pin::SpecialOrthogonalOperationTpl<2, Scalar>>;
  using SO3 = PinocchioLieGroup<pin::SpecialOrthogonalOperationTpl<3, Scalar>>;
  using SE2 = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<2, Scalar>>;
  using SE3 = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<3, Scalar>>;

  /* Expose tangent bundles */

  exposeTangentBundle<SO2>("TSO2", "Tangent bundle of the SO(2) group.")
      .def(bp::init<>(bp::args("self")));
  exposeTangentBundle<SO3>("TSO3", "Tangent bundle of the SO(3) group.")
      .def(bp::init<>(bp::args("self")));
  exposeTangentBundle<SE2>("TSE2", "Tangent bundle of the SE(2) group.")
      .def(bp::init<>(bp::args("self")));
  exposeTangentBundle<SE3>("TSE3", "Tangent bundle of the SE(3) group.")
      .def(bp::init<>(bp::args("self")));

  /* Groups associated w/ Pinocchio models */
  using Multibody = MultibodyConfiguration<Scalar>;
  using Model = pin::ModelTpl<Scalar>;
  bp::class_<Multibody, bp::bases<Manifold>>(
      "MultibodyConfiguration", "Configuration group of a multibody",
      bp::init<const Model &>(bp::args("self", "model")))
      .add_property(
          "model",
          bp::make_function(
              &Multibody::getModel,
              bp::return_value_policy<bp::reference_existing_object>()),
          "Return the Pinocchio model instance.");

  bp::class_<MultibodyPhaseSpace<Scalar>, bp::bases<Manifold>>(
      "MultibodyPhaseSpace",
      "Tangent space of the multibody configuration group.",
      bp::init<const Model &>(bp::args("self", "model")))
      .add_property(
          "model",
          bp::make_function(
              &MultibodyPhaseSpace<Scalar>::getModel,
              bp::return_value_policy<bp::reference_existing_object>()),
          "Return the Pinocchio model instance.")
      .add_property(
          "base",
          +[](const MultibodyPhaseSpace<Scalar> &m) {
            return m.getBaseSpace();
          }, // decay lambda to function ptr
          "Get the base space.");
#endif
}

} // namespace python
} // namespace proxnlp
