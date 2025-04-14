#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
#include <pinocchio/fwd.hpp>
#include "proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp"
#include "proxsuite-nlp/modelling/spaces/multibody.hpp"
#endif

#include "proxsuite-nlp/python/polymorphic.hpp"
#include "proxsuite-nlp/manifold-base.hpp"
#include "proxsuite-nlp/modelling/spaces/cartesian-product.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"
#include "proxsuite-nlp/modelling/spaces/tangent-bundle.hpp"

#include <eigenpy/std-vector.hpp>

#include <vector>

namespace proxsuite {
namespace nlp {
namespace python {
using context::ConstVectorRef;
using context::Manifold;
using context::MatrixRef;
using context::Scalar;
using context::VectorRef;
using PolyManifold = polymorphic<Manifold>;
using CartesianProduct = CartesianProductTpl<Scalar>;

void exposeManifoldBase() {
  using context::MatrixXs;
  using context::VectorXs;
  register_polymorphic_to_python<PolyManifold>();

  using BinaryFunTypeRet = VectorXs (Manifold::*)(const ConstVectorRef &,
                                                  const ConstVectorRef &) const;
  using BinaryFunType = void (Manifold::*)(
      const ConstVectorRef &, const ConstVectorRef &, VectorRef) const;
  using JacobianFunType = void (Manifold::*)(
      const ConstVectorRef &, const ConstVectorRef &, MatrixRef, int) const;

  bp::class_<Manifold, boost::noncopyable>(
      "ManifoldAbstract", "Manifold abstract class.", bp::no_init)
      .add_property("nx", &Manifold::nx, "Manifold representation dimension.")
      .add_property("ndx", &Manifold::ndx, "Tangent space dimension.")
      .def("neutral", &Manifold::neutral, "self"_a,
           "Get the neutral point from the manifold (if a Lie group).")
      .def("rand", &Manifold::rand, "self"_a,
           "Sample a random point from the manifold.")
      .def("isNormalized", &Manifold::isNormalized, ("self"_a, "x"),
           "Check if the input vector :math:`x` is a viable element of the "
           "manifold.")
      .def<BinaryFunType>("integrate", &Manifold::integrate,
                          ("self"_a, "x", "v", "out"))
      .def<BinaryFunType>("difference", &Manifold::difference,
                          ("self"_a, "x0", "x1", "out"))
      .def<BinaryFunTypeRet>("integrate", &Manifold::integrate,
                             ("self"_a, "x", "v"))
      .def<BinaryFunTypeRet>("difference", &Manifold::difference,
                             ("self"_a, "x0", "x1"))
      .def("interpolate",
           (void (Manifold::*)(const ConstVectorRef &, const ConstVectorRef &,
                               const Scalar &, VectorRef)
                const)(&Manifold::interpolate),
           ("self"_a, "x0", "x1", "u", "out"))
      .def("interpolate",
           (VectorXs (Manifold::*)(const ConstVectorRef &,
                                   const ConstVectorRef &, const Scalar &)
                const)(&Manifold::interpolate),
           ("self"_a, "x0", "x1", "u"),
           "Interpolate between two points on the manifold. Allocated version.")
      .def<JacobianFunType>("Jintegrate", &Manifold::Jintegrate,
                            ("self"_a, "x", "v", "Jout", "arg"),
                            "Compute the Jacobian of the exp operator.")
      .def<JacobianFunType>("Jdifference", &Manifold::Jdifference,
                            ("self"_a, "x0", "x1", "Jout", "arg"),
                            "Compute the Jacobian of the log operator.")
      .def(
          "Jintegrate",
          +[](const Manifold &m, const ConstVectorRef x,
              const ConstVectorRef &v, int arg) {
            MatrixXs Jout(m.ndx(), m.ndx());
            m.Jintegrate(x, v, Jout, arg);
            return Jout;
          },
          ("self"_a, "x", "v", "arg"),
          "Compute and return the Jacobian of the exp.")
      .def("JintegrateTransport", &Manifold::JintegrateTransport,
           ("self"_a, "x", "v", "J", "arg"),
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
          ("self"_a, "x0", "x1", "arg"),
          "Compute and return the Jacobian of the log.")
      .def("tangent_space", &Manifold::tangentSpace, bp::args("self"),
           "Returns an object representing the tangent space to this manifold.")
      .def(
          "__mul__",
          +[](const PolyManifold &a, const PolyManifold &b) { return a * b; })
      .def(
          "__mul__", +[](const PolyManifold &a,
                         const CartesianProduct &b) { return a * b; })
      .def(
          "__rmul__", +[](const PolyManifold &a, const CartesianProduct &b) {
            return a * b;
          });

  eigenpy::StdVectorPythonVisitor<std::vector<PolyManifold>>::expose(
      "StdVec_Manifold",
      eigenpy::details::overload_base_get_item_for_std_vector<
          std::vector<PolyManifold>>());
}

/// Expose the tangent bundle of a manifold type @p M.
template <typename M>
bp::class_<TangentBundleTpl<M>, bp::bases<Manifold>>
exposeTangentBundle(const char *name, const char *docstring) {
  using OutType = TangentBundleTpl<M>;
  return bp::class_<OutType, bp::bases<Manifold>>(
             name, docstring, bp::init<M>(("self"_a, "base")))
      .add_property("base",
                    bp::make_function(&OutType::getBaseSpace,
                                      bp::return_internal_reference<>()),
                    "Get the base space.")
      .def(PolymorphicVisitor<PolyManifold>());
}

/// Expose the tangent bundle with an additional constructor.
template <typename M, class Init>
bp::class_<TangentBundleTpl<M>, bp::bases<Manifold>>
exposeTangentBundle(const char *name, const char *docstring, Init init) {
  return exposeTangentBundle<M>(name, docstring).def(init);
}

void exposeCartesianProduct();

#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
/// Expose a Pinocchio Lie group with a specified name, docstring,
/// and no-arg default constructor.
template <typename LieGroup>
void exposeLieGroup(const char *name, const char *docstring) {
  bp::class_<PinocchioLieGroup<LieGroup>, bp::bases<Manifold>> cl(
      name, docstring, bp::init<>("self"_a));
  cl.def(PolymorphicVisitor<PolyManifold>());
  cl.enable_pickling_(true);
}

void exposePinocchioSpaces() {
  namespace pin = pinocchio;
  using pin::ModelTpl;
  using pin::SpecialEuclideanOperationTpl;
  using pin::SpecialOrthogonalOperationTpl;
  using pin::VectorSpaceOperationTpl;

  using DynSizeEuclideanSpace = VectorSpaceOperationTpl<Eigen::Dynamic, Scalar>;
  bp::class_<PinocchioLieGroup<DynSizeEuclideanSpace>, bp::bases<Manifold>>(
      "EuclideanSpace", "Pinocchio's n-dimensional Euclidean vector space.",
      bp::no_init)
      .def(bp::init<int>(("self"_a, "dim")))
      .def(PolymorphicVisitor<PolyManifold>());

  exposeLieGroup<VectorSpaceOperationTpl<1, Scalar>>(
      "R", "One-dimensional Euclidean space AKA real number line.");
  exposeLieGroup<VectorSpaceOperationTpl<2, Scalar>>(
      "R2", "Two-dimensional Euclidean space.");
  exposeLieGroup<VectorSpaceOperationTpl<3, Scalar>>(
      "R3", "Three-dimensional Euclidean space.");
  exposeLieGroup<VectorSpaceOperationTpl<4, Scalar>>(
      "R4", "Four-dimensional Euclidean space.");
  exposeLieGroup<SpecialOrthogonalOperationTpl<2, Scalar>>(
      "SO2", "SO(2) special orthogonal group.");
  exposeLieGroup<SpecialOrthogonalOperationTpl<3, Scalar>>(
      "SO3", "SO(3) special orthogonal group.");
  exposeLieGroup<SpecialEuclideanOperationTpl<2, Scalar>>(
      "SE2", "SE(2) special Euclidean group.");
  exposeLieGroup<SpecialEuclideanOperationTpl<3, Scalar>>(
      "SE3", "SE(3) special Euclidean group.");

  using SO2 = PinocchioLieGroup<SpecialOrthogonalOperationTpl<2, Scalar>>;
  using SO3 = PinocchioLieGroup<SpecialOrthogonalOperationTpl<3, Scalar>>;
  using SE2 = PinocchioLieGroup<SpecialEuclideanOperationTpl<2, Scalar>>;
  using SE3 = PinocchioLieGroup<SpecialEuclideanOperationTpl<3, Scalar>>;

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
  using Model = ModelTpl<Scalar>;
  bp::class_<Multibody, bp::bases<Manifold>>(
      "MultibodyConfiguration", "Configuration group of a multibody",
      bp::init<const Model &>(bp::args("self", "model")))
      .add_property("model",
                    bp::make_function(&Multibody::getModel,
                                      bp::return_internal_reference<>()),
                    "Return the Pinocchio model instance.")
      .def(PolymorphicVisitor<PolyManifold>())
      .enable_pickling_(true);

  using MultiPhase = MultibodyPhaseSpace<Scalar>;
  bp::class_<MultiPhase, bp::bases<Manifold>>(
      "MultibodyPhaseSpace",
      "Tangent space of the multibody configuration group.",
      bp::init<const Model &>(("self"_a, "model")))
      .add_property("model",
                    bp::make_function(&MultiPhase::getModel,
                                      bp::return_internal_reference<>()),
                    "Return the Pinocchio model instance.")
      .add_property("base", bp::make_function(
                                +[](const MultiPhase &m) -> const Multibody & {
                                  return m.getBaseSpace();
                                },
                                bp::return_internal_reference<>()))
      .def(PolymorphicVisitor<PolyManifold>())
      .enable_pickling_(true);
}
#endif

void exposeManifolds() {

  exposeManifoldBase();

  /* Basic vector space */
  bp::class_<VectorSpaceTpl<Scalar>, bp::bases<Manifold>>(
      "VectorSpace", "Basic Euclidean vector space.", bp::no_init)
      .def(bp::init<const int>(("self"_a, "dim")))
      .def(PolymorphicVisitor<PolyManifold>())
      .enable_pickling_(true);

  exposeCartesianProduct();
#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
  exposePinocchioSpaces();
#endif
}

} // namespace python
} // namespace nlp
} // namespace proxsuite
