#pragma once

#include "lienlp/cost-function.hpp"
#include "lienlp/residual-base.hpp"

namespace lienlp {

  /**
   * Cost function which is the weighted squares of some
   * residual function.
   */
  template<typename _Scalar>
  class QuadResidualCost : public CostFunctionBase<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    using CstrType = ResidualBase<Scalar>;  // base constraint func to use
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = CostFunctionBase<Scalar>;
    using Base::computeGradient;
    using Base::computeHessian;
    using Base::m_ndx;

    CstrType* m_residual;
    MatrixXs m_weights;

    QuadResidualCost(CstrType* residual,
                     const ConstVectorRef& weightMatrix)
    : Base(residual->ndx()), m_residual(residual), m_weights(weightMatrix)
    {}

    template<typename... ResidualArgs>
    QuadResidualCost(const ConstVectorRef& weightMatrix,
                     ResidualArgs&... args)
    : QuadResidualCost(new CstrType(args...),  weightMatrix)
    {}

    Scalar operator()(const ConstVectorRef& x) const
    {
      VectorXs err = m_residual->operator()(x);
      return Scalar(0.5) * err.dot(m_weights * err);
    }

    void computeGradient(const ConstVectorRef& x, RefVector out) const
    {
      MatrixXs Jres(m_residual->nr(), m_ndx);
      m_residual->computeJacobian(x, Jres);
      VectorXs err = m_residual->operator()(x);
      out.noalias() = Jres.transpose() * (m_weights * err);
    }

    void computeHessian(const ConstVectorRef& x, RefMatrix out) const
    {
      MatrixXs Jres(m_residual->nr(), m_ndx);
      m_residual->computeJacobian(x, Jres);
      out.noalias() = Jres.transpose() * (m_weights * Jres);
    }

  };

} // namespace lienlp
