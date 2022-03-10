// Basis for merit functions.
#pragma once

#include "lienlp/problem-base.hpp"

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>


namespace lienlp {

  template<typename _Scalar, typename... Args>
  struct MeritFunctionTpl
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

    Prob_t* m_prob;

    virtual Scalar operator()(const VectorXs& x, const Args&... args) const = 0;
    virtual VectorXs gradient(const VectorXs& x, const Args&... args) const = 0;
  };


  /// Simply evaluate the objective function.
  template<class _Scalar>
  struct EvalObjective : MeritFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

    Prob_t* m_prob;

    EvalObjective(Prob_t* prob) : m_prob(prob) {}

    Scalar operator()(const VectorXs& x) const
    {
      return m_prob->m_cost(x);
    }

    VectorXs gradient(const VectorXs& x) const
    {
      return m_prob->m_cost.gradient(x);
    }

  };



}  // namespace lienlp
