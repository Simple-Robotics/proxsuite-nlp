// Basis for merit functions.
#pragma once

#include "lienlp/problem-base.hpp"

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"

#include <boost/shared_ptr.hpp>


namespace lienlp {
  using boost::shared_ptr;

  template<typename _Scalar, typename... Args>
  struct MeritFunctorBase
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

    shared_ptr<Prob_t> m_prob;

    MeritFunctorBase(shared_ptr<Prob_t> prob) : m_prob(prob) {}

    /// Evaluate the merit function.
    virtual Scalar operator()(const VectorXs& x, const Args&... args) const = 0;
    /// Evaluate the merit function gradient.
    virtual void gradient(const VectorXs& x, const Args&... args, VectorXs& out) const = 0;
    /// @copybrief gradient()
    VectorXs gradient(const VectorXs& x, const Args&... args) const
    {
      VectorXs out;
      gradient(x, args..., out);
      return out;
    }

    /// Compute the merit function Hessian matrix.
    virtual void hessian(const VectorXs& x, const Args&... args, MatrixXs& out) const = 0;
  };


  /// Simply evaluate the objective function.
  template<class _Scalar>
  struct EvalObjective : public MeritFunctorBase<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;
    using Parent = MeritFunctorBase<Scalar>;
    using Parent::m_prob;
    using Parent::gradient;

    EvalObjective(shared_ptr<Prob_t> prob)
      : Parent(prob) {}

    Scalar operator()(const VectorXs& x) const
    {
      return m_prob->m_cost(x);
    }

    void gradient(const VectorXs& x, VectorXs& out) const
    {
      out.noalias() = m_prob->m_cost.gradient(x);
    }

    void hessian(const VectorXs& x , MatrixXs& out) const
    {
      m_prob->m_cost.hessian(x, out);
    }
  };



}  // namespace lienlp
