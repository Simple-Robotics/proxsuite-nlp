/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
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

  Scalar mu_ = 0.;

  Scalar evaluate(const ConstVectorRef &zproj) const {
    return zproj.template lpNorm<1>();
  }

  decltype(auto) projection_impl(const ConstVectorRef &z) const {
    return z.array().sign() *
           (z.array().abs() - mu_).max(static_cast<Scalar>(0.));
  }

  void projection(const ConstVectorRef &z, VectorRef zout) const {
    zout = projection_impl(z);
  }

  void normalConeProjection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z.cwiseMin(mu_).cwiseMax(-mu_);
  }

  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const {
    out = z.array().abs() < mu_;
  }

  void setProxParameters(const Scalar mu) { mu_ = mu; };
};

} // namespace proxnlp
