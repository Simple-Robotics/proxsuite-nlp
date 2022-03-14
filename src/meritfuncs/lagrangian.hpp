#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/merit-function-base.hpp"

namespace lienlp {


  /**
   * The Lagrangian function of a problem instance.
   * This inherits from the merit function template with a single
   * extra argument.
   */
  template<typename _Scalar>
  struct LagrangianFunction :
  public MeritFunctorBase<
    _Scalar, typename math_types<_Scalar>::VectorOfVectors
    >
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;
    using Parent = MeritFunctorBase<Scalar, VectorOfVectors>;
    using Parent::m_prob;
    using Parent::gradient;

    LagrangianFunction(shared_ptr<Prob_t> prob)
      : Parent(prob) {}

    Scalar operator()(const ConstVectorRef& x, const VectorOfVectors& lams) const
    {
      Scalar result_ = 0.;
      result_ = result_ + m_prob->m_cost(x);
      const auto num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getCstr(i);
        result_ = result_ + lams[i].dot((*cstr)(x));
      }
      return result_;
    }

    void gradient(const ConstVectorRef& x,
                  const VectorOfVectors& lams,
                  RefVector out) const
    {
      out.noalias() = m_prob->m_cost.gradient(x);
      const int num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getCstr(i);
        out.noalias() += (cstr->jacobian(x)).transpose() * lams[i];
      }
    }

    void hessian(const ConstVectorRef& x,
                 const VectorOfVectors& lams,
                 RefMatrix out) const
    {
      m_prob->m_cost.hessian(x, out);
      const int num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getCstr(i);
        out.noalias() += cstr->m_func.vhp(x, lams[i]);
      }
    }
  };
} // namespace lienlp

