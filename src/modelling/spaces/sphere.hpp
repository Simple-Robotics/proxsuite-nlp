#pragma once

#include "lienlp/manifold-base.hpp"

namespace lienlp
{

  template<int Dim, typename Scalar, int Options = 0>
  struct Sphere : ManifoldAbstract<Dim, Scalar, Options>
  {};

  template<typename _Scalar, int _Options=0>
  struct Sphere<3, _Scalar, _Options> : ManifoldAbstract<3, _Scalar, _Options>
  {
    using Scalar = _Scalar;
    enum {
      Options = _Options
    };
    using Base = ManifoldAbstract<Scalar, Options>;
    LIENLP_DEFINE_MANIFOLD_TYPES(Base)

    /**
     * Geodesics on the 3D sphere are given by the great circles.
     */
    void integrate_impl(const ConstVectorRef& x,
                        const ConstVectorRef& v,
                        VectorRef out) const
    {

    }

    /**
     * Geodesics on the 3D sphere are given by the great circles.
     * We first need to derive a parametrization of the circle.
     */
    void difference_impl(
      const ConstVectorRef& x0,
      const ConstVectorRef& x1,
      VectorRef out) const
    {

    }
  };

  template<typename Scalar, intt Options=0>
  using Sphere3D = Sphere<3, Scalar, Options>;

}
