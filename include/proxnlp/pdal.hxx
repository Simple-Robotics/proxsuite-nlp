/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/pdal.hpp"

namespace proxnlp {
template <typename Scalar>
PDALFunction<Scalar>::PDALFunction(shared_ptr<Problem> prob, const Scalar mu,
                                   const Scalar gamma)
    : problem_(prob), mu_(mu), gamma_(gamma) {}

template <typename Scalar>
Scalar
PDALFunction<Scalar>::evaluate(const ConstVectorRef &x, const VectorOfRef &lams,
                               const std::vector<VectorRef> &shift_cvals,
                               const std::vector<VectorRef> &proj_cvals) const {
  Scalar res = problem_->cost().call(x);
  const std::size_t nc = problem_->getNumConstraints();
  for (std::size_t i = 0; i < nc; i++) {
    const ConstraintObjectTpl<Scalar> &cstr = problem_->getConstraint(i);
    res += cstr.set_->evaluateMoreauEnvelope(shift_cvals[i], proj_cvals[i]);
    if (gamma_ > 0.) {
      res += 0.5 * gamma_ * mu_inv_ *
             (proj_cvals[i] - mu_ * lams[i]).squaredNorm();
    }
  }
  return res;
}

template <typename Scalar>
void PDALFunction<Scalar>::setPenalty(const Scalar &new_mu) noexcept {
  mu_ = new_mu;
  mu_inv_ = 1. / new_mu;
}

} // namespace proxnlp
