#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  template<class M>
  class CostFunction
  {
  public:
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)

    virtual Scalar operator()(const VectorXs& x) const = 0;
    virtual VectorXs gradient(const VectorXs& x) const = 0;
    virtual MatrixXs hessian(const VectorXs& x) const = 0;

    virtual ~CostFunction<M>() = default;

  };

}  // namespace lienlp
  
