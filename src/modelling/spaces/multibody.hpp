#pragma once

#include "lienlp/modelling/spaces/pinocchio-groups.hpp"
#include "lienlp/modelling/spaces/tangent-bundle.hpp"


namespace lienlp {

  /// @brief      A convenient alias for the tangent bundle of a multibody configuration group.
  /// @details    This cannot not a typedef, since we provide a constructor for the class.
  template<typename Scalar, int Options=0>
  struct StateMultibody : TangentBundle<MultibodyConfiguration<Scalar, Options>>
  {
    using ConfigSpace = MultibodyConfiguration<Scalar, Options>;
    StateMultibody<Scalar, Options>(const pinocchio::ModelTpl<Scalar, Options>& model)
      : TangentBundle<ConfigSpace>(ConfigSpace(model)) {}
  };

} // namespace lienlp
