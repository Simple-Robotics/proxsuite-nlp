#pragma once

#include "proxnlp/constraint-base.hpp"

namespace proxnlp {

/**
 * @brief   Composite \f$\ell_1\f$-penalty function \f$ \|c(x)\|_1 \f$.
 *
 * @details The composite \f$\ell_1\f$-penalty penalizes the norm
 *          \f$ \| r(x) \|_1\f$ of a residual function.
 *          This class implements the proximity operator (soft-thresholding)
 *          and an appropriate generalized Jacobian.
 */
template <typename _Scalar> struct L1Penalty : ConstraintSetBase<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);

  using Base = ConstraintSetBase<Scalar>;
  using ActiveType = typename Base::ActiveType;
  using FunctionType = typename Base::FunctionType;

  Scalar mu_;

  Scalar evaluate(const ConstVectorRef &zproj) const {
    return zproj.lpNorm<1>();
  }

  ReturnType projection(const ConstVectorRef &z) const {
    return z.array().sign() * (z.abs().array() - mu_).max(Scalar(0.));
  }

  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const {
    out = (z.abs().array() - mu_) <= Scalar(0.);
  }

  void setProxParameters(const Scalar mu) { mu_ = mu; };
};

} // namespace proxnlp
