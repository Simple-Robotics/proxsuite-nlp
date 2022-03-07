// Basis for merit functions.
#pragma once

#include "lienlp/manifold-base.hpp"
#include "lienlp/cost-function.hpp"
#include "lienlp/constraint.hpp"


namespace lienlp {


template<class M>
struct MeritFunction
{
  using Scalar = typename M::Scalar;
  using Point_t = typename M::Point_t;

  template<typename... Args>
  Scalar operator()(const Point_t& x, const Args&... args) const;
};


/// Simply evaluate the objective function.
template<class M>
struct SimpleEval : MeritFunction<M>
{
  using Cost_t = CostFunction<M>;
  Cost_t* m_func;

  using Scalar = typename M::Scalar;
  using Point_t = typename M::Point_t;

  SimpleEval(Cost_t* f) : m_func(f) {};

  Scalar operator()(const Point_t& x) const
  {
    return (*m_func)(x);
  }

};



}  // namespace lienlp
