// Basis for merit functions.
#pragma once

#include "proxnlp/problem-base.hpp"

#include "proxnlp/fwd.hpp"


namespace proxnlp
{

  template<typename _Scalar, typename... Args>
  struct MeritFunctionBase
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Problem = ProblemTpl<Scalar>;

    shared_ptr<Problem> m_prob;

    MeritFunctionBase(shared_ptr<Problem> prob) : m_prob(prob) {}

    /// Evaluate the merit function.
    virtual Scalar operator()(const ConstVectorRef& x, const Args&... args) const = 0;
    /// Evaluate the merit function gradient.
    virtual void computeGradient(const ConstVectorRef& x, const Args&... args, VectorRef out) const = 0;

  };


  /// Simply evaluate the objective function.
  template<typename _Scalar>
  struct EvalObjective : public MeritFunctionBase<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Problem = ProblemTpl<Scalar>;
    using Base = MeritFunctionBase<Scalar>;
    using Base::m_prob;

    EvalObjective(shared_ptr<Problem> prob)
      : Base(prob) {}

    Scalar operator()(const ConstVectorRef& x) const
    {
      return m_prob->m_cost.call(x);
    }

    void computeGradient(const ConstVectorRef& x, VectorRef out) const
    {
      m_prob->m_cost.computeGradient(x, out);
    }
  };



}  // namespace proxnlp
