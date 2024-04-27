#pragma once

#include "proxsuite-nlp/constraint-base.hpp"

namespace proxsuite {
namespace nlp {

/**
 * @brief   Equality constraints \f$c(x) = 0\f$.
 *
 * @details This class implements the set associated with equality
 * constraints\f$ c(x) = 0 \f$, where \f$c : \calX \to \RR^p\f$ is a residual
 * function.
 */
template <typename _Scalar>
struct EqualityConstraintTpl : ConstraintSetBase<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Base = ConstraintSetBase<Scalar>;
  using ActiveType = typename Base::ActiveType;

  bool disableGaussNewton() const { return true; }

  inline void projection(const ConstVectorRef &, VectorRef zout) const {
    zout.setZero();
  }

  inline void normalConeProjection(const ConstVectorRef &z,
                                   VectorRef zout) const {
    zout = z;
  }

  inline void applyProjectionJacobian(const ConstVectorRef &,
                                      MatrixRef Jout) const {
    Jout.setZero();
  }

  inline void applyNormalConeProjectionJacobian(const ConstVectorRef &,
                                                MatrixRef) const {
    return; // do nothing
  }

  inline void computeActiveSet(const ConstVectorRef &,
                               Eigen::Ref<ActiveType> out) const {
    out.array() = true;
  }
};

template <typename Scalar>
using EqualityConstraint PROXSUITE_NLP_DEPRECATED_MESSAGE(
    "Use EqualityConstraintTpl<T> instead") = EqualityConstraintTpl<Scalar>;

} // namespace nlp
} // namespace proxsuite

#include "proxsuite-nlp/modelling/constraints.hpp"
