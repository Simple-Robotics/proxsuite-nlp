// Basis for merit functions.
#pragma once

#include "proxnlp/problem-base.hpp"

#include "proxnlp/fwd.hpp"

namespace proxnlp {

template <typename _Scalar, typename... Args> struct MeritFunctionBaseTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;

  shared_ptr<Problem> problem_;

  MeritFunctionBaseTpl(shared_ptr<Problem> prob) : problem_(prob) {}

  /// Evaluate the merit function.
  virtual Scalar operator()(const ConstVectorRef &x,
                            const Args &...args) const = 0;
  /// Evaluate the merit function gradient.
  virtual void computeGradient(const ConstVectorRef &x, const Args &...args,
                               VectorRef out) const = 0;
};

/// Simply evaluate the objective function.
template <typename _Scalar>
struct EvalObjective : public MeritFunctionBaseTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;
  using Base = MeritFunctionBaseTpl<Scalar>;
  using Base::problem_;

  EvalObjective(shared_ptr<Problem> prob) : Base(prob) {}

  Scalar operator()(const ConstVectorRef &x) const {
    return problem_->cost().call(x);
  }

  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    problem_->cost().computeGradient(x, out);
  }
};

} // namespace proxnlp
