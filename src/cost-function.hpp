#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp
{

  template<typename _Scalar>
  class CostFunctionBase
  {
  protected:
    const int m_nx;
    const int m_ndx;
  public:
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    CostFunctionBase(const int nx, const int ndx) : m_nx(nx), m_ndx(ndx) {}

    virtual Scalar operator()(const ConstVectorRef& x) const = 0;
    virtual void computeGradient(const ConstVectorRef& x, VectorRef out) const = 0;
    virtual void computeHessian(const ConstVectorRef& x, MatrixRef out) const = 0;

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

    int ndx() const
    { return m_ndx; }

    virtual ~CostFunctionBase<Scalar>() = default;

  };

}  // namespace lienlp
  
