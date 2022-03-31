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
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Base = CostFunctionBase<Scalar>;
    using Base::computeGradient;
    using Base::computeHessian;

    shared_ptr<FunctorType> m_residual;
    MatrixXs m_weights;
    VectorXs m_slope;
    Scalar m_constant;

    QuadraticResidualCost(const shared_ptr<FunctorType>& residual,
                          const ConstMatrixRef& weights,
                          const ConstVectorRef& slope,
                          const Scalar constant = Scalar(0.))
      : Base(residual->nx(), residual->ndx())
      , m_residual(residual)
      , m_weights(weights)
      , m_slope(slope)
      , m_constant(constant)
    {}

    QuadraticResidualCost(const shared_ptr<FunctorType>& residual,
                          const ConstMatrixRef& weights,
                          const Scalar constant = Scalar(0.))
      : QuadraticResidualCost(residual, weights, VectorXs::Zero(residual->nr()), constant)
    {}


    template<typename... ResidualArgs>
    QuadraticResidualCost(const ConstMatrixRef& weights,
                          const ConstVectorRef& slope,
                          const Scalar constant,
                          ResidualArgs&... args)
    : QuadraticResidualCost(new FunctorType(args...),  weights, slope, constant)
    {}

    Scalar call(const ConstVectorRef& x) const
    {
      VectorXs err = (*m_residual)(x);
      return Scalar(0.5) * err.dot(m_weights * err) + err.dot(m_slope) + m_constant;
    }

    void computeGradient(const ConstVectorRef& x, VectorRef out) const
    {
      MatrixXs Jres(m_residual->nr(), this->ndx());
      m_residual->computeJacobian(x, Jres);
      VectorXs err = (*m_residual)(x);
      out = Jres.transpose() * (m_weights * err + m_slope);
    }

    void computeHessian(const ConstVectorRef& x, MatrixRef out) const
    {
      MatrixXs Jres(m_residual->nr(), this->ndx());
      m_residual->computeJacobian(x, Jres);
      VectorXs err = (*m_residual)(x);
      out = Jres.transpose() * (m_weights * Jres) \
        + m_residual->vectorHessianProduct(x, m_weights * err + m_slope);
    }

  };

} // namespace lienlp
