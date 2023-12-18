#pragma once

#include "proxnlp/manifold-base.hpp"

namespace proxnlp {

template <int _Dim, typename _Scalar, int _Options = 0>
struct Sphere : ManifoldAbstractTpl<Scalar, Options> {
  using Scalar = _Scalar;
  enum { Dim = _Dim, Options = _Options };
};

template <typename _Scalar, int _Options>
struct Sphere<3, _Scalar, _Options>
    : ManifoldAbstractTpl<3, _Scalar, _Options> {
  using Scalar = _Scalar;
  enum { Options = _Options };
  using Base = ManifoldAbstractTpl<Scalar, Options>;
  PROXSUITE_NLP_DEFINE_MANIFOLD_TYPES(Base)

  /**
   * Geodesics on the 3D sphere are given by the great circles.
   */
  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const {}

  /**
   * Geodesics on the 3D sphere are given by the great circles.
   * We first need to derive a parametrization of the circle.
   */
  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const {}
};

} // namespace proxnlp
