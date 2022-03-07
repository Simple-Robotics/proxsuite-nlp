#pragma once

#include <Eigen/Core>

#include "lienlp/manifold-base.hpp"


namespace lienlp {

  template<class M, int NC=Eigen::Dynamic>
  struct ConstraintFunctor
  {
    using Scalar = typename M::Scalar;
    using Point_t = typename M::Point_t;
    using Vec_t = typename M::TangentVec_t;
    using C_t = Eigen::Matrix<Scalar, NC, M::NV, M::Options>;
    using Jac_t = Eigen::Matrix<Scalar, NC, M::NV, M::Options>;

    template<typename... Args>
    C_t operator()(const Eigen::MatrixBase<Point_t>& x, const Args&...) const;

  };

}
