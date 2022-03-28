#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/modelling/costs/squared-residual.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

namespace lienlp{

  template<typename _Scalar>
  struct QuadDistanceCost : QuadraticResidualCost<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using FunctorType = StateResidual<Scalar>;
    using M = ManifoldAbstract<Scalar>;
    using Base = QuadraticResidualCost<Scalar>;
    using Base::m_residual;

    QuadDistanceCost(const M& manifold, const ConstMatrixRef& weights)
      : Base(std::make_shared<FunctorType>(manifold, manifold.neutral()), weights)
      {}

    QuadDistanceCost(const M& manifold)
      : QuadDistanceCost(manifold, MatrixXs::Identity(manifold.ndx(), manifold.ndx()))
      {}

    void updateTarget(const ConstVectorRef& x)
    {
      std::static_pointer_cast<FunctorType>(m_residual)->m_target = x;
    }
  };

} // namespace lienlp
