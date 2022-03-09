#pragma once

#include "lienlp/merit-function-base.hpp"


namespace lienlp {
  
  /**
   * Primal-dual Augmented Lagrangian function, extending
   * the function from Gill & Robinson (2012) to inequality constraints.
   * 
   */
  template<class M>
  struct PDALFunction : MeritFunctionTpl<M>
  {
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)

    using Cost_t = CostFunction<M>;
    Cost_t& m_func;
  };

}