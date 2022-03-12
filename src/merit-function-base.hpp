// Basis for merit functions.
#pragma once

#include "lienlp/problem-base.hpp"

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>


namespace lienlp {

  template<typename _Scalar, typename... Args>
  struct MeritFunctorBase
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

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
  };


  /// Simply evaluate the objective function.
  template<class _Scalar>
  struct EvalObjective : MeritFunctorBase<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;
    using MeritFunctorBase<_Scalar>::gradient;

    Prob_t* m_prob;

    EvalObjective(Prob_t* prob) : m_prob(prob) {}

    Scalar operator()(const VectorXs& x) const
    {
      return m_prob->m_cost(x);
    }

    void gradient(const VectorXs& x, VectorXs& out) const
    {
      VectorXs g = m_prob->m_cost.gradient(x);
      out.noalias() = g;
    }

  };



}  // namespace lienlp
