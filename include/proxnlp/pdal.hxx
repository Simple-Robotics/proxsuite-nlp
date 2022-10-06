#pragma once

#include "proxnlp/pdal.hpp"

namespace proxnlp {
template <typename Scalar>
PDALFunction<Scalar>::PDALFunction(shared_ptr<Problem> prob, const Scalar mu)
    : problem_(prob), mu_penal_(mu) {}

template <typename Scalar>
Scalar PDALFunction<Scalar>::evaluate(const ConstVectorRef &x,
                                      const VectorOfRef &lams,
                                      const VectorOfRef &lams_ext,
                                      std::vector<VectorRef> &tmp_cvals) const {
  Scalar res = problem_->cost().call(x);
  const std::size_t nc = problem_->getNumConstraints();
  for (std::size_t i = 0; i < nc; i++) {
    const ConstraintObject<Scalar> &cstr = problem_->getConstraint(i);
    tmp_cvals[i] = cstr.func()(x) + mu_penal_ * lams_ext[i];
    res +=
        computeMoreauEnvelope(*cstr.set_, tmp_cvals[i], tmp_cvals[i], mu_inv_);
    res += gamma_ * static_cast<Scalar>(0.5) * mu_inv_ *
           (tmp_cvals[i] - mu_penal_ * lams[i]).squaredNorm();
  }
  return res;
}

template <typename Scalar>
void PDALFunction<Scalar>::setPenalty(const Scalar &new_mu) noexcept {
  mu_penal_ = new_mu;
  mu_inv_ = 1. / new_mu;
};

} // namespace proxnlp
