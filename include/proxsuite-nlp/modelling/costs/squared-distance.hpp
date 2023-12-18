#pragma once

#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"
#include "proxsuite-nlp/modelling/residuals/state-residual.hpp"

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
struct QuadraticDistanceCostTpl : QuadraticResidualCostTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  using FunctionType = ManifoldDifferenceToPoint<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Base = QuadraticResidualCostTpl<Scalar>;
  using Base::residual_;
  using Base::weights_;

  QuadraticDistanceCostTpl(const shared_ptr<Manifold> &space,
                           const ConstVectorRef &target,
                           const ConstMatrixRef &weights)
      : Base(std::make_shared<FunctionType>(space, target), weights) {}

  QuadraticDistanceCostTpl(const shared_ptr<Manifold> &space,
                           const ConstVectorRef &target)
      : QuadraticDistanceCostTpl(
            space, target, MatrixXs::Identity(space->ndx(), space->ndx())) {}

  QuadraticDistanceCostTpl(const shared_ptr<Manifold> &space)
      : QuadraticDistanceCostTpl(space, space->neutral()) {}

  ConstVectorRef getTarget() const {
    return std::static_pointer_cast<FunctionType>(residual_)->target_;
  }

  void updateTarget(const ConstVectorRef &x) {
    std::static_pointer_cast<FunctionType>(residual_)->target_ = x;
  }
};

} // namespace proxnlp

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/modelling/costs/squared-distance.txx"
#endif
