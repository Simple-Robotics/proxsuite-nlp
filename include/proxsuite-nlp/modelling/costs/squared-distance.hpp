#pragma once

#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"
#include "proxsuite-nlp/modelling/residuals/state-residual.hpp"

namespace proxsuite {
namespace nlp {

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
  using StateResidual = ManifoldDifferenceToPoint<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Base = QuadraticResidualCostTpl<Scalar>;
  using Base::residual_;
  using Base::weights_;

  QuadraticDistanceCostTpl(const polymorphic<Manifold> &space,
                           const ConstVectorRef &target,
                           const ConstMatrixRef &weights)
      : Base(std::make_shared<StateResidual>(space, target), weights) {}

  QuadraticDistanceCostTpl(const polymorphic<Manifold> &space,
                           const ConstVectorRef &target)
      : QuadraticDistanceCostTpl(
            space, target, MatrixXs::Identity(space->ndx(), space->ndx())) {}

  QuadraticDistanceCostTpl(const polymorphic<Manifold> &space)
      : QuadraticDistanceCostTpl(space, space->neutral()) {}

  ConstVectorRef getTarget() const {
    return static_cast<StateResidual *>(residual_.get())->target_;
  }

  void updateTarget(const ConstVectorRef &x) {
    static_cast<StateResidual *>(residual_.get())->target_ = x;
  }
};

} // namespace nlp
} // namespace proxsuite

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/modelling/costs/squared-distance.txx"
#endif
