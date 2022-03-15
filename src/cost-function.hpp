#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  template<typename _Scalar>
  class CostFunctionBase
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    const int m_ndx;

    CostFunctionBase(const int ndx) : m_ndx(ndx) {}

    virtual Scalar operator()(const ConstVectorRef& x) const = 0;
    virtual void computeGradient(const ConstVectorRef& x, RefVector out) const = 0;
    virtual void computeHessian(const ConstVectorRef& x, RefMatrix out) const = 0;

    VectorXs computeGradient(const ConstVectorRef& x) const
    {
      VectorXs out(m_ndx);
      computeGradient(x, out);
      return out;
    }

    MatrixXs computeHessian(const ConstVectorRef& x) const
    {
      MatrixXs out(m_ndx, m_ndx);
      computeHessian(x, out);
      return out;
    }

    virtual ~CostFunctionBase<Scalar>() = default;

  };

}  // namespace lienlp
  
