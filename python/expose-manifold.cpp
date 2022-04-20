#include "proxnlp/python/manifold.hpp"


#include "proxnlp/modelling/spaces/tangent-bundle.hpp"
#ifdef WITH_PINOCCHIO
  #include "proxnlp/modelling/spaces/pinocchio-groups.hpp"
  #include "proxnlp/modelling/spaces/multibody.hpp"
#endif


namespace proxnlp
{
namespace python
{

  /// Expose the tangent bundle of a manifold type @p M.
  template<typename M>
  bp::class_<TangentBundleTpl<M>, bp::bases<context::Manifold>>
  exposeTangentBundle(const char* name, const char* docstring)
  {
    using OutType = TangentBundleTpl<M>;
    return bp::class_<OutType, bp::bases<context::Manifold>>(
        name, docstring, bp::init<M>(bp::args("self", "base"))
      )
      .add_property("base",
                    bp::make_function(&OutType::getBaseSpace, bp::return_value_policy<bp::reference_existing_object>()),
                    "Get the base space.")
      ;
  }

  /// Expose the tangent bundle with an additional constructor.
  template<typename M, class Init>
  bp::class_<TangentBundleTpl<M>, bp::bases<context::Manifold>>
  exposeTangentBundle(const char* name, const char* docstring, Init init)
  {
    return exposeTangentBundle<M>(name, docstring)
      .def(init)
      ;
  }

#ifdef WITH_PINOCCHIO
  /// Expose a Pinocchio Lie group with a specified name and docstring,
  /// but a no-arg default constructor.
  template<typename LieGroup>
  void exposeLieGroup(const char* name, const char* docstring)
  {
    using Manifold = PinocchioLieGroup<LieGroup>;
    bp::class_<Manifold, bp::bases<context::Manifold>>(
      name, docstring, bp::init<>()
    );
  }
#endif

  void exposeManifold()
  {
    using context::Scalar;
    using context::Manifold;

    internal::exposeBaseManifold();

#ifdef WITH_PINOCCHIO
    namespace pin = pinocchio;
    using VectorSpace = pin::VectorSpaceOperationTpl<Eigen::Dynamic, Scalar>;
    bp::class_<PinocchioLieGroup<VectorSpace>, bp::bases<Manifold>>(
      "EuclideanSpace", "n-dimensional Euclidean vector space.",
      bp::init<int>(bp::args("dim"))
    );

    exposeLieGroup<pin::VectorSpaceOperationTpl<1,Scalar>>("R", "One-dimensional Euclidean space AKA real number line.");
    exposeLieGroup<pin::VectorSpaceOperationTpl<2,Scalar>>("R2", "Two-dimensional Euclidean space.");
    exposeLieGroup<pin::VectorSpaceOperationTpl<3,Scalar>>("R3", "Three-dimensional Euclidean space.");
    exposeLieGroup<pin::VectorSpaceOperationTpl<4,Scalar>>("R4", "Four-dimensional Euclidean space.");
    exposeLieGroup<pin::SpecialOrthogonalOperationTpl<2,Scalar>>("SO2", "SO(2) special orthogonal group.");
    exposeLieGroup<pin::SpecialOrthogonalOperationTpl<3,Scalar>>("SO3", "SO(3) special orthogonal group.");
    exposeLieGroup<pin::SpecialEuclideanOperationTpl<2,Scalar>> ("SE2", "SE(2) special Euclidean group.");
    exposeLieGroup<pin::SpecialEuclideanOperationTpl<3,Scalar>> ("SE3", "SE(3) special Euclidean group.");

    using SO2 = PinocchioLieGroup<pin::SpecialOrthogonalOperationTpl<2,Scalar>>;
    using SO3 = PinocchioLieGroup<pin::SpecialOrthogonalOperationTpl<3,Scalar>>;
    using SE2 = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<2,Scalar>>;
    using SE3 = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<3,Scalar>>;

    /* Expose tangent bundles */

    exposeTangentBundle<SO2>("TSO2", "Tangent bundle of the SO(2) group.").def(bp::init<>());
    exposeTangentBundle<SO3>("TSO3", "Tangent bundle of the SO(3) group.").def(bp::init<>());
    exposeTangentBundle<SE2>("TSE2", "Tangent bundle of the SE(2) group.").def(bp::init<>());
    exposeTangentBundle<SE3>("TSE3", "Tangent bundle of the SE(3) group.").def(bp::init<>());


    /* Groups associated w/ Pinocchio models */
    using Multibody_t = MultibodyConfiguration<Scalar>;
    using Model_t = pin::ModelTpl<Scalar>;
    bp::class_<Multibody_t, bp::bases<Manifold>>(
      "MultibodyConfiguration", "Configuration group of a multibody",
      bp::init<const Model_t&>(bp::args("self", "model"))
    )
      .add_property("model",
                    bp::make_function(&MultibodyPhaseSpace<Scalar>::getModel,
                                      bp::return_value_policy<bp::reference_existing_object>()),
                    "Return the Pinocchio model instance.");

    bp::class_<MultibodyPhaseSpace<Scalar>, bp::bases< Manifold >>(
      "MultibodyPhaseSpace", "Tangent space of the multibody configuration group.",
      bp::init<const Model_t&>(bp::args("model"))
    )
      .add_property("model",
                    bp::make_function(&MultibodyPhaseSpace<Scalar>::getModel,
                                      bp::return_value_policy<bp::reference_existing_object>()),
                    "Return the Pinocchio model instance.")
      .add_property("base",
                    +[](const MultibodyPhaseSpace<Scalar>& m) { return m.getBaseSpace(); },
                    "Get the base space.");
#endif

  }

} // namespace python
} // namespace proxnlp
