/**
 * @file    Implements a function which is the residual between two points on
 * the space, obtained by the space retraction op.
 */
#pragma once

#include "proxsuite-nlp/function-base.hpp"
#include "proxsuite-nlp/manifold-base.hpp"
#include "proxsuite-nlp/third-party/polymorphic_cxx14.hpp"

namespace proxsuite {
namespace nlp {
using xyz::polymorphic;

/**
 * Constraint function to be equal to a given element of a space.
 * This is templated on the space.
 */
template <typename _Scalar>
struct ManifoldDifferenceToPoint : C2FunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Base = C2FunctionTpl<Scalar>;
  using Base::operator();
  using Base::computeJacobian;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  /// Target point on the space.
  typename Manifold::PointType target_;
  polymorphic<Manifold> space_;

  ManifoldDifferenceToPoint(const polymorphic<Manifold> &space,
                            const ConstVectorRef &target)
      : Base(space->nx(), space->ndx(), space->ndx()), target_(target),
        space_(space) {
    if (!space->isNormalized(target_)) {
      PROXSUITE_NLP_RUNTIME_ERROR(
          "Target parameter is not a valid element of the manifold.");
    }
  }

  VectorXs operator()(const ConstVectorRef &x) const {
    return space_->difference(target_, x);
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    space_->Jdifference(target_, x, Jout, 1);
  }
};

} // namespace nlp
} // namespace proxsuite
