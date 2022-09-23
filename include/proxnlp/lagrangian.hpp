#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/merit-function-base.hpp"

namespace proxnlp {

/**
 * The Lagrangian function of a problem instance.
 * This inherits from the merit function template with a single
 * extra argument.
 */
template <typename _Scalar>
struct LagrangianFunction
    : public MeritFunctionBaseTpl<_Scalar,
                                  typename math_types<_Scalar>::VectorOfRef> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;
  using Base = MeritFunctionBaseTpl<Scalar, VectorOfRef>;
  using Base::computeGradient;
  using Base::problem_;

  LagrangianFunction(shared_ptr<Problem> prob) : Base(prob) {}

  Scalar operator()(const ConstVectorRef &x, const VectorOfRef &lams) const {
    Scalar result_ = 0.;
    result_ = result_ + problem_->cost().call(x);
    const std::size_t num_c = problem_->getNumConstraints();
    for (std::size_t i = 0; i < num_c; i++) {
      const auto cstr = problem_->getConstraint(i);
      result_ = result_ + lams[i].dot(cstr.func()(x));
    }
    return result_;
  }

  void computeGradient(const ConstVectorRef &x, const VectorOfRef &lams,
                       VectorRef out) const {
    out = problem_->cost().computeGradient(x);
    const std::size_t num_c = problem_->getNumConstraints();
    for (std::size_t i = 0; i < num_c; i++) {
      auto cstr = problem_->getConstraint(i);
      out += (cstr.func().computeJacobian(x)).transpose() * lams[i];
    }
  }
};
} // namespace proxnlp
