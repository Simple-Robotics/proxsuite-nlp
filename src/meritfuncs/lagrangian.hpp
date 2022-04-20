#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/merit-function-base.hpp"

namespace proxnlp
{


  /**
   * The Lagrangian function of a problem instance.
   * This inherits from the merit function template with a single
   * extra argument.
   */
  template<typename _Scalar>
  struct LagrangianFunction :
  public MeritFunctionBase<
    _Scalar, typename math_types<_Scalar>::VectorOfRef
    >
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;
    using Base = MeritFunctionBase<Scalar, VectorOfRef>;
    using Base::m_prob;
    using Base::computeGradient;

    LagrangianFunction(shared_ptr<Problem> prob)
      : Base(prob) {}

    Scalar operator()(const ConstVectorRef& x, const VectorOfRef& lams) const
    {
      Scalar result_ = 0.;
      result_ = result_ + m_prob->m_cost.call(x);
      const std::size_t num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        const auto cstr = m_prob->getConstraint(i);
        result_ = result_ + lams[i].dot(cstr->m_func(x));
      }
      return result_;
    }

    void computeGradient(const ConstVectorRef& x,
                         const VectorOfRef& lams,
                         VectorRef out) const
    {
      out = m_prob->m_cost.computeGradient(x);
      const std::size_t num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getConstraint(i);
        out += (cstr->m_func.computeJacobian(x)).transpose() * lams[i];
      }
    }

    void computeHessian(const ConstVectorRef& x,
                        const VectorOfRef& lams,
                        MatrixRef out) const
    {
      m_prob->m_cost.computeHessian(x, out);
      const auto num_c = m_prob->getNumConstraints();
      const int ndx = m_prob->m_cost.ndx();
      MatrixXs vhp_buffer(ndx, ndx);
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getConstraint(i);
        cstr->m_func.vectorHessianProduct(x, lams[i], vhp_buffer);
        out += vhp_buffer;
      }
    }
  };
} // namespace proxnlp

