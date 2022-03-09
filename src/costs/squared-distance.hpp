#pragma once

#include "lienlp/costs/squared-residual.hpp"
#include "lienlp/constraints/state-constraint.hpp"

namespace lienlp{
  
  template<class M>
  using WeightedSquareDistanceCost = QuadResidualCost<StateResidual<M>>;

} // namespace lienlp
