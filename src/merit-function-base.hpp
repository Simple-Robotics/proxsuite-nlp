// Basis for merit functions.
#pragma once

#include "lienlp/problem-base.hpp"

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"


namespace lienlp
{

  template<typename _Scalar, typename... Args>
  struct MeritFunctionBase
  {
    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;

    shared_ptr<Problem> m_prob;

    MeritFunctionBase(shared_ptr<Problem> prob) : m_prob(prob) {}

    /// Evaluate the merit function.
    virtual Scalar operator()(const ConstVectorRef& x, const Args&... args) const = 0;
    /// Evaluate the merit function gradient.
    virtual void computeGradient(const ConstVectorRef& x, const Args&... args, VectorRef out) const = 0;
    /// Compute the merit function Hessian matrix.
    virtual void computeHessian(const ConstVectorRef& x, const Args&... args, MatrixRef out) const = 0;

  };


  /// Simply evaluate the objective function.
  template<typename _Scalar>
  struct EvalObjective : public MeritFunctionBase<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;
    using Base = MeritFunctionBase<Scalar>;
    using Base::m_prob;
    using Base::computeGradient;

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

    void computeHessian(const ConstVectorRef& x , MatrixRef out) const
    {
      m_prob->m_cost.computeHessian(x, out);
    }
  };



}  // namespace lienlp
