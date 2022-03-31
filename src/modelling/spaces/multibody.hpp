#pragma once

#include "lienlp/modelling/spaces/pinocchio-groups.hpp"
#include "lienlp/modelling/spaces/tangent-bundle.hpp"


namespace lienlp
{

  /** @brief      The tangent bundle of a multibody configuration group.
   *  @details    This is not a typedef, since we provide a constructor for the class.
   *              Any point on the manifold is of the form \f$x = (q,v) \f$, where
   *              \f$q \in \mathcal{Q} \f$ is a configuration.
   */
  template<typename Scalar, int Options=0>
  struct StateMultibody : TangentBundleTpl<MultibodyConfiguration<Scalar, Options>>
  {
    using ConfigSpace = MultibodyConfiguration<Scalar, Options>;
    using ModelType = typename ConfigSpace::ModelType;

    const ModelType& getModel() { return this->m_base.getModel(); }

    StateMultibody(const ModelType& model)
      : TangentBundleTpl<ConfigSpace>(ConfigSpace(model))
      {}
  };

} // namespace lienlp
