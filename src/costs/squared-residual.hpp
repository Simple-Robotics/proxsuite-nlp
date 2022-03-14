#pragma once

#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"

namespace lienlp {

  /**
   * Cost function which is the weighted squares of some
   * residual function.
   */
  template<typename _Scalar>
  class QuadResidualCost : public CostFunction<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    using CstrType = ConstraintFuncTpl<Scalar>;  // base constraint func to use
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Parent = CostFunction<Scalar>;
    using Parent::gradient;
    using Parent::hessian;
    using Parent::m_ndx;

    CstrType* m_residual;
    MatrixXs m_weights;

    QuadResidualCost(CstrType* residual,
                     const ConstVectorRef& weightMatrix)
    : Parent(residual->ndx()), m_residual(residual), m_weights(weightMatrix)
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

    void gradient(const ConstVectorRef& x, RefVector out) const
    {
      MatrixXs Jres(m_residual->getDim(), m_ndx);
      m_residual->jacobian(x, Jres);
      VectorXs err = m_residual->operator()(x);
      out.noalias() = Jres.transpose() * (m_weights * err);
    }

    void hessian(const ConstVectorRef& x, RefMatrix out) const
    {
      MatrixXs Jres(m_residual->getDim(), m_ndx);
      m_residual->jacobian(x, Jres);
      out.noalias() = Jres.transpose() * (m_weights * Jres);
    }

  };

} // namespace lienlp
