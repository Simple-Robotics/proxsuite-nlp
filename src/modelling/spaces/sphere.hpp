#pragma once

#include "lienlp/manifold-base.hpp"

namespace lienlp
{

  template<int Dim, typename Scalar, int Options = 0>
  struct Sphere : ManifoldTpl<Sphere<Dim, Scalar, Options>>
  {};

  template<int Dim, typename scalar, int options>
  struct traits<Sphere<Dim, scalar, options>>
  {
    using Scalar = scalar;
    enum {
      NQ = Dim,
      NV = Dim,
      Options = options
    };
  };

  template<typename Scalar, int Options>
  struct Sphere<3, Scalar, Options> : ManifoldTpl<Sphere<3, Scalar, Options>>
  {
    /**
     * Geodesics on the 3D sphere are given by the great circles.
     */
    template<class Vec_t, class Tangent_t, class Out_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        const Eigen::MatrixBase<Out_t>& out) const
    {

    }

    /**
     * Geodesics on the 3D sphere are given by the great circles.
     * We first need to derive a parametrization of the circle.
     */
    template<class Vec1_t, class Vec2_t, class Tangent_t>
    void difference_impl(
      const Eigen::MatrixBase<Vec1_t>& x0,
      const Eigen::MatrixBase<Vec2_t>& x1,
      const Eigen::MatrixBase<Tangent_t>& out) const
    {

    }
  };

  template<typename Scalar>
  using Sphere3D = Sphere<3, Scalar>;

}
