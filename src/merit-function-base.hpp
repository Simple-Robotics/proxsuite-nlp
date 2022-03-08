// Basis for merit functions.
#pragma once

#include "lienlp/manifold-base.hpp"
#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"

#include "lienlp/fwd.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>


namespace lienlp {

  template<typename Scalar, typename... Args>
  struct MeritFunctionTpl
  {
    using VectorXs = typename math_types<Scalar>::VectorXs;
    virtual Scalar operator()(const VectorXs& x, const Args&... args) const = 0;
    virtual VectorXs gradient(const VectorXs& x, const Args&... args) const = 0;
  };


  /// Simply evaluate the objective function.
  template<class M>
  struct EvalObjective : MeritFunctionTpl<typename M::Scalar>
  {
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)

    using Cost_t = CostFunction<M>;
    Cost_t& m_func;

    EvalObjective(Cost_t& f) : m_func(f) {};

    Scalar operator()(const VectorXs& x) const
    {
      return m_func(x);
    }

    VectorXs gradient(const VectorXs& x) const
    {
      return m_func.gradient(x);
    }

  };



}  // namespace lienlp
