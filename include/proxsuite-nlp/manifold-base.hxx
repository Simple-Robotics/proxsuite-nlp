/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/manifold-base.hpp"

namespace proxnlp {

/* Integrate */

template <typename Scalar, int Options>
void ManifoldAbstractTpl<Scalar, Options>::integrate(const ConstVectorRef &x,
                                                     const ConstVectorRef &v,
                                                     VectorRef out) const {
  integrate_impl(x, v, out);
}

template <typename Scalar, int Options>
void ManifoldAbstractTpl<Scalar, Options>::Jintegrate(const ConstVectorRef &x,
                                                      const ConstVectorRef &v,
                                                      MatrixRef Jout,
                                                      int arg) const {
  Jintegrate_impl(x, v, Jout, arg);
}

/* Difference */

template <typename Scalar, int Options>
void ManifoldAbstractTpl<Scalar, Options>::difference(const ConstVectorRef &x0,
                                                      const ConstVectorRef &x1,
                                                      VectorRef out) const {
  difference_impl(x0, x1, out);
}

template <typename Scalar, int Options>
void ManifoldAbstractTpl<Scalar, Options>::Jdifference(const ConstVectorRef &x0,
                                                       const ConstVectorRef &x1,
                                                       MatrixRef Jout,
                                                       int arg) const {
  Jdifference_impl(x0, x1, Jout, arg);
}

template <typename Scalar, int Options>
void ManifoldAbstractTpl<Scalar, Options>::interpolate(const ConstVectorRef &x0,
                                                       const ConstVectorRef &x1,
                                                       const Scalar &u,
                                                       VectorRef out) const {
  interpolate_impl(x0, x1, u, out);
}

} // namespace proxnlp
