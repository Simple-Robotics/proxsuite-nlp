#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/modelling/costs/squared-residual.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

namespace lienlp{

  template<typename M>
  struct QuadDistanceCost : QuadraticResidualCost<typename M::Scalar>
  {
    using Scalar = typename M::Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using ResidualType = StateResidual<M>;
    using Base = QuadraticResidualCost<Scalar>;
    using Base::m_residual;

    QuadDistanceCost(M& manifold)
      : Base(std::make_shared<ResidualType>(manifold, manifold.zero()),
             MatrixXs::Identity(manifold.ndx(), manifold.ndx()))
      {}

    void updateTarget(const ConstVectorRef& x)
    {
      std::static_pointer_cast<ResidualType>(m_residual)->m_target = x;
    }
  };

} // namespace lienlp
