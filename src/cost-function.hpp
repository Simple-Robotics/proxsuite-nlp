#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  template<typename _Scalar>
  class CostFunction
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    const int m_ndx;

    CostFunction(const int& ndx) : m_ndx(ndx) {}

    virtual Scalar operator()(const ConstVectorRef& x) const = 0;
    virtual void gradient(const ConstVectorRef& x, RefVector out) const = 0;
    virtual void hessian(const ConstVectorRef& x, RefMatrix out) const = 0;

    VectorXs gradient(const ConstVectorRef& x) const
    {
      VectorXs out(m_ndx);
      gradient(x, out);
      return out;
    }

    MatrixXs hessian(const ConstVectorRef& x) const
    {
      MatrixXs out(m_ndx, m_ndx);
      hessian(x, out);
      return out;
    }

    virtual ~CostFunction<Scalar>() = default;

  };

}  // namespace lienlp
  
