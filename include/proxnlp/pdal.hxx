/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/pdal.hpp"

namespace proxnlp {
template <typename Scalar>
PDALFunction<Scalar>::PDALFunction(shared_ptr<Problem> prob, const Scalar gamma)
    : problem_(prob), gamma_(gamma) {}

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
    Scalar mu = cstr.set_->mu();
    if (gamma_ > 0.) {
      res += 0.5 * gamma_ * cstr.set_->mu_inv() *
             (proj_cvals[i] - mu * lams[i]).squaredNorm();
    }
  }
  return res;
}

} // namespace proxnlp
