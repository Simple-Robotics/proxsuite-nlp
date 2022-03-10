#pragma once

#include "lienlp/costs/squared-residual.hpp"
#include "lienlp/constraints/state-constraint.hpp"

namespace lienlp{
  
  template<class Scalar>
  using WeightedSquareDistanceCost = QuadResidualCost<StateResidual<Scalar>>;

} // namespace lienlp
