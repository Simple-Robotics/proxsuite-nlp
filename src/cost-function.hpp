#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  template<class _Scalar>
  class CostFunction
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    virtual Scalar operator()(const VectorXs& x) const = 0;
    virtual VectorXs gradient(const VectorXs& x) const = 0;
    virtual void hessian(const VectorXs& x, MatrixXs& out) const = 0;

    MatrixXs hessian(const VectorXs& x) const
    {
      MatrixXs out;
      hessian(x, out);
      return out;
    }

    virtual ~CostFunction<Scalar>() = default;

  };

}  // namespace lienlp
  
