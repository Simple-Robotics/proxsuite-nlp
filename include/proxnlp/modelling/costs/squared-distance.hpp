#pragma once

#include "proxnlp/modelling/costs/quadratic-residual.hpp"
#include "proxnlp/modelling/residuals/state-residual.hpp"

namespace proxnlp {

/** @brief    Weighted quadratic distance \f$\frac{1}{2}\|x\ominus
 * \bar{x}\|^2_W\f$ on a space.
 *
 *  @details  This function subclasses from QuadraticResidualCost and
 *            provides a convenient constructor. It uses
 * ManifoldDifferenceToPoint under the hood as the input residual for the
 * parent. This struct also exposes a method to update the target point.
 */
template <typename _Scalar>
struct QuadraticDistanceCost : QuadraticResidualCost<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using FunctionType = ManifoldDifferenceToPoint<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Base = QuadraticResidualCost<Scalar>;
  using Base::residual_;
  using Base::weights_;

  QuadraticDistanceCost(const shared_ptr<Manifold> &space,
                        const ConstVectorRef &target,
                        const ConstMatrixRef &weights)
      : Base(std::make_shared<FunctionType>(space, target), weights) {}

  QuadraticDistanceCost(const shared_ptr<Manifold> &space,
                        const ConstVectorRef &target)
      : QuadraticDistanceCost(space, target,
                              MatrixXs::Identity(space->ndx(), space->ndx())) {}

  QuadraticDistanceCost(const shared_ptr<Manifold> &space)
      : QuadraticDistanceCost(space, space->neutral()) {}

  void updateTarget(const ConstVectorRef &x) {
    std::static_pointer_cast<FunctionType>(residual_)->target_ = x;
  }
};

} // namespace proxnlp

#ifdef PROXNLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxnlp/modelling/costs/squared-distance.txx"
#endif
