#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/modelling/costs/quadratic-residual.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

namespace lienlp{

  /** @brief    Weighted quadratic distance \f$\frac{1}{2}\|x\ominus \bar{x}\|_W\f$ on a manifold.
   * 
   *  @details  This function subclasses from QuadraticResidualCost and
   *            provides a convenient constructor. It uses StateResidual under the hood
   *            as the input residual for the parent.
   *            This struct also exposes a method to update the target point.
   */
  template<typename _Scalar>
  struct QuadDistanceCost : QuadraticResidualCost<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using FunctorType = StateResidual<Scalar>;
    using M = ManifoldAbstract<Scalar>;
    using Base = QuadraticResidualCost<Scalar>;
    using Base::m_residual;
    using Base::m_weights;

    QuadDistanceCost(const M& manifold,
                     const ConstVectorRef& target,
                     const ConstMatrixRef& weights)
      : Base(std::make_shared<FunctorType>(manifold, target), weights)
      {}

    QuadDistanceCost(const M& manifold, const ConstVectorRef& target)
      : QuadDistanceCost(manifold, target, MatrixXs::Identity(manifold.ndx(), manifold.ndx()))
      {}

    void updateTarget(const ConstVectorRef& x)
    {
      std::static_pointer_cast<FunctorType>(m_residual)->m_target = x;
    }
  };

} // namespace lienlp
