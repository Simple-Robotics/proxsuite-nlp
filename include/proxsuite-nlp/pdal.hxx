/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/pdal.hpp"

namespace proxsuite {
namespace nlp {
template <typename Scalar>
ALMeritFunctionTpl<Scalar>::ALMeritFunctionTpl(const Problem &prob,
                                               const Scalar &beta)
    : beta_(beta), problem_(prob) {}

template <typename Scalar>
Scalar ALMeritFunctionTpl<Scalar>::evaluate(const ConstVectorRef & /*x*/,
                                            const std::vector<VectorRef> &lams,
                                            Workspace &workspace) const {
  Scalar res = workspace.objective_value;
  // value c(x) + \mu\lambda_e
  const auto &pd_scv = workspace.shift_cstr_pdal;
  for (std::size_t i = 0; i < workspace.numblocks; i++) {
    const ConstraintObject &cstr = problem_.getConstraint(i);
    Scalar mu = cstr.set_->mu();
    VectorXs scv_tmp = pd_scv[i];
    res += 2.0 * cstr.set_->computeMoreauEnvelope(pd_scv[i], scv_tmp);
    res += mu * lams[i].squaredNorm() / 4.0;
  }
  return res;
}

template <typename Scalar>
void ALMeritFunctionTpl<Scalar>::computeGradient(
    const std::vector<VectorRef> &lams, Workspace &workspace) const {
  workspace.merit_gradient = workspace.objective_gradient;
  workspace.merit_gradient.noalias() +=
      workspace.data_jacobians.transpose() * workspace.data_lams_pdal;
  workspace.merit_dual_gradient.setZero();
  for (std::size_t i = 0; i < workspace.numblocks; i++) {
    const ConstraintObject &cstr = problem_.getConstraint(i);
    Scalar mu = cstr.set_->mu();
    problem_.getSegment(workspace.merit_dual_gradient, i).noalias() +=
        beta_ * mu * (lams[i] - workspace.lams_pdal[i]);
  }
}

} // namespace nlp
} // namespace proxsuite
