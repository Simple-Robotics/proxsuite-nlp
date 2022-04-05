#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/modelling/costs/quadratic-residual.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

namespace lienlp
{

  /** @brief    Weighted quadratic distance \f$\frac{1}{2}\|x\ominus \bar{x}\|^2_W\f$ on a manifold.
   * 
   *  @details  This function subclasses from QuadraticResidualCost and
   *            provides a convenient constructor. It uses ManifoldDifferenceToPoint under the hood
   *            as the input residual for the parent.
   *            This struct also exposes a method to update the target point.
   */
  template<typename _Scalar>
  struct QuadraticDistanceCost : QuadraticResidualCost<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using FunctionType = ManifoldDifferenceToPoint<Scalar>;
    using M = ManifoldAbstractTpl<Scalar>;
    using Base = QuadraticResidualCost<Scalar>;
    using Base::m_residual;
    using Base::m_weights;

    QuadraticDistanceCost(const M& manifold,
                     const ConstVectorRef& target,
                     const ConstMatrixRef& weights)
      : Base(std::make_shared<FunctionType>(manifold, target), weights)
      {}

    QuadraticDistanceCost(const M& manifold, const ConstVectorRef& target)
      : QuadraticDistanceCost(manifold, target, MatrixXs::Identity(manifold.ndx(), manifold.ndx()))
      {}

    void updateTarget(const ConstVectorRef& x)
    {
      std::static_pointer_cast<FunctionType>(m_residual)->m_target = x;
    }
  };

} // namespace lienlp
