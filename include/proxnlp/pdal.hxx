/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/pdal.hpp"

namespace proxnlp {
template <typename Scalar>
ALMeritFunctionTpl<Scalar>::ALMeritFunctionTpl(shared_ptr<const Problem> prob,
                                               const Scalar gamma)
    : gamma_(gamma), problem_(prob) {}

template <typename Scalar>
Scalar ALMeritFunctionTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                            const std::vector<VectorRef> &lams,
                                            Workspace &workspace) const {
  Scalar res = problem_->cost().call(x);
  const std::size_t nc = problem_->getNumConstraints();
  for (std::size_t i = 0; i < nc; i++) {
    const ConstraintObject &cstr = problem_->getConstraint(i);
    const auto &scv = workspace.shift_cstr_values;
    const auto &pcv = workspace.shift_cstr_proj;
    res += cstr.set_->evaluateMoreauEnvelope(scv[i], pcv[i]);
    Scalar mu = cstr.set_->mu();
    if (gamma_ > 0.) {
      res += 0.5 * gamma_ * cstr.set_->mu_inv() *
             (pcv[i] - mu * lams[i]).squaredNorm();
    }
  }
  return res;
}

template <typename Scalar>
Scalar ALMeritFunctionTpl<Scalar>::derivative(Workspace &workspace) const {
  const auto &dx = workspace.prim_step;
  const auto &dlam = workspace.dual_step;
  return workspace.merit_gradient.dot(dx) -
         gamma_ * workspace.data_dual_prox_err.dot(dlam);
}

} // namespace proxnlp
