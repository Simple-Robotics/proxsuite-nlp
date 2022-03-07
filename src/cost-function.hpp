#pragma once

#include "lienlp/manifold-base.hpp"


namespace lienlp {

  template<class M>
  struct CostFunction
  {
    using Scalar = typename M::Scalar;
    using Point_t = typename M::Point_t;
    using Vec_t = typename M::TangentVec_t;
    using Hess_t = Eigen::Matrix<Scalar, M::NV, M::NV, M::Options>;

    Scalar operator()(const Eigen::MatrixBase<Point_t>& x) const;
    Vec_t jacobian(const Eigen::MatrixBase<Point_t>& x) const;
    Hess_t hessian(const Eigen::MatrixBase<Point_t>& x) const;

  };

}  // namespace lienlp
  
