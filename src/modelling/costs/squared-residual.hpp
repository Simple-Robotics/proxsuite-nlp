#pragma once

#include "lienlp/cost-function.hpp"
#include "lienlp/functor-base.hpp"

namespace lienlp
{

  /**
   * @brief Weighted quadratic residual \f$\frac{1}{2}\| r(x) \|_W \f$ of a residual \f$r(x)\f$.
   * 
   * Cost function which is the weighted squares \f$\frac12 \|r(x)\|_W \f$ of some
   * residual function \f$r :\calX\to \RR^p\f$.
   */
  template<typename _Scalar>
  struct QuadraticResidualCost : public CostFunctionBase<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    using FunctorType = DifferentiableFunctor<Scalar>;  // base constraint func to use
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = CostFunctionBase<Scalar>;
    using Base::computeGradient;
    using Base::computeHessian;
    using Base::m_ndx;

    shared_ptr<FunctorType> m_residual;
    MatrixXs m_weights;

    QuadraticResidualCost(const shared_ptr<FunctorType>& residual,
                          const ConstMatrixRef& weights)
      : Base(residual->nx(), residual->ndx()),
        m_residual(residual), m_weights(weights)
    {}

    template<typename... ResidualArgs>
    QuadraticResidualCost(const ConstVectorRef& weights,
                     ResidualArgs&... args)
    : QuadraticResidualCost(new FunctorType(args...),  weights)
    {}

    Scalar operator()(const ConstVectorRef& x) const
    {
      VectorXs err = (*m_residual)(x);
      return Scalar(0.5) * err.dot(m_weights * err);
    }

    void computeGradient(const ConstVectorRef& x, VectorRef out) const
    {
      MatrixXs Jres(m_residual->nr(), m_ndx);
      m_residual->computeJacobian(x, Jres);
      VectorXs err = (*m_residual)(x);
      out = Jres.transpose() * (m_weights * err);
    }

    void computeHessian(const ConstVectorRef& x, MatrixRef out) const
    {
      MatrixXs Jres(m_residual->nr(), m_ndx);
      m_residual->computeJacobian(x, Jres);
      VectorXs err = (*m_residual)(x);
      out = Jres.transpose() * (m_weights * Jres) + m_residual->vectorHessianProduct(x, err);
    }

  };

} // namespace lienlp
