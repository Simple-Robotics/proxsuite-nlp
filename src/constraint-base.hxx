#pragma once

namespace proxnlp {
template <typename Scalar>
inline typename ConstraintSetBase<Scalar>::ReturnType
ConstraintSetBase<Scalar>::normalConeProjection(const ConstVectorRef &z) const {
  return z - projection(z);
}

template <typename Scalar>
void ConstraintSetBase<Scalar>::applyProjectionJacobian(const ConstVectorRef &z,
                                                        MatrixRef Jout) const {
  const int nr = (int)z.size();
  assert(nr == Jout.rows());
  ActiveType active_set(nr);
  computeActiveSet(z, active_set);
  for (int i = 0; i < nr; i++) {
    /// active constraints -> projector onto the constraint set is zero
    if (active_set(i)) {
      Jout.row(i).setZero();
    }
  }
}

template <typename Scalar>
void ConstraintSetBase<Scalar>::applyNormalConeProjectionJacobian(
    const ConstVectorRef &z, MatrixRef Jout) const {
  const int nr = (int)z.size();
  assert(nr == Jout.rows());
  ActiveType active_set(nr);
  computeActiveSet(z, active_set);
  for (int i = 0; i < nr; i++) {
    /// inactive constraint -> normal cone projection is zero
    if (not active_set(i)) {
      Jout.row(i).setZero();
    }
  }
}

} // namespace proxnlp
