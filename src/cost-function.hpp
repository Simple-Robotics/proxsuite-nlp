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
    virtual MatrixXs hessian(const VectorXs& x) const = 0;

    virtual ~CostFunction<Scalar>() = default;

  };

}  // namespace lienlp
  
