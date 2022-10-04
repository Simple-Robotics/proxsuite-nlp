/**
 * @file    Implements a function which is the residual between two points on
 * the space, obtained by the space retraction op.
 */
#pragma once

#include "proxnlp/function-base.hpp"
#include "proxnlp/manifold-base.hpp"

namespace proxnlp {

/**
 * Constraint function to be equal to a given element of a space.
 * This is templated on the space.
 */
template <typename _Scalar>
struct ManifoldDifferenceToPoint : C2FunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);

  using Base = C2FunctionTpl<Scalar>;
  using Base::operator();
  using Base::computeJacobian;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  /// Target point on the space.
  typename Manifold::PointType target_;
  shared_ptr<Manifold> space_;

  ManifoldDifferenceToPoint(const shared_ptr<Manifold> &space,
                            const ConstVectorRef &target)
      : Base(space->nx(), space->ndx(), space->ndx()), target_(target),
        space_(space) {}

  ReturnType operator()(const ConstVectorRef &x) const {
    return space_->difference(target_, x);
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    space_->Jdifference(target_, x, Jout, 1);
  }
};

} // namespace proxnlp
