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
    using Base = MeritFunctorBase<Scalar, VectorOfVectors>;
    using Base::m_prob;
    using Base::computeGradient;

    LagrangianFunction(shared_ptr<Prob_t> prob)
      : Base(prob) {}

    Scalar operator()(const ConstVectorRef& x, const VectorOfVectors& lams) const
    {
      Scalar result_ = 0.;
      result_ = result_ + m_prob->m_cost(x);
      const std::size_t num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        const auto cstr = m_prob->getCstr(i);
        result_ = result_ + lams[i].dot((*cstr)(x));
      }
      return result_;
    }

    void computeGradient(const ConstVectorRef& x,
                         const VectorOfVectors& lams,
                         RefVector out) const
    {
      out.noalias() = m_prob->m_cost.computeGradient(x);
      const std::size_t num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getCstr(i);
        out.noalias() += (cstr->computeJacobian(x)).transpose() * lams[i];
      }
    }

    void computeHessian(const ConstVectorRef& x,
                        const VectorOfVectors& lams,
                        RefMatrix out) const
    {
      m_prob->m_cost.computeHessian(x, out);
      const std::size_t num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getCstr(i);
        out.noalias() += cstr->m_func.vhp(x, lams[i]);
      }
    }
  };
} // namespace lienlp

