#pragma once

#include "lienlp/manifold-base.hpp"

namespace lienlp
{

  /**
   * @brief     Tangent bundle of a manifold M. This construction is recursive.
   */
  template<class Base>
  struct TangentBundle<Base> : public ManifoldTpl<TangentBundle<Base>>
  {
    TangentBundle<Base>(Base base) : baseManifold(base) {}; 
  protected:
    Base baseManifold;
  };

  template<class M>
  struct traits<TangentBundle<M>>
  {
    using base_traits = traits<M>;
    using Scalar = typename base_traits::Scalar;
    enum {
      NQ = Eigen::Dynamic,
      NV = Eigen::Dynamic
    };
  }

}
