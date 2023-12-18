#pragma once

#include "proxsuite-nlp/modelling/spaces/tangent-bundle.hpp"

namespace proxnlp {

template <class Base>
typename TangentBundleTpl<Base>::PointType
TangentBundleTpl<Base>::neutral() const {
  PointType out(nx());
  out.setZero();
  out.head(base_.nx()) = base_.neutral();
  return out;
}

template <class Base>
typename TangentBundleTpl<Base>::PointType
TangentBundleTpl<Base>::rand() const {
  PointType out(nx());
  out.head(base_.nx()) = base_.rand();
  using BTanVec_t = typename Base::TangentVectorType;
  out.tail(base_.ndx()) = BTanVec_t::Random(base_.ndx());
  return out;
}

/// Operators
template <class Base>
void TangentBundleTpl<Base>::integrate_impl(const ConstVectorRef &x,
                                            const ConstVectorRef &dx,
                                            VectorRef out) const {
  const int nv_ = base_.ndx();
  base_.integrate(getBasePoint(x), getBaseTangent(dx), out.head(base_.nx()));
  out.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
}

template <class Base>
void TangentBundleTpl<Base>::difference_impl(const ConstVectorRef &x0,
                                             const ConstVectorRef &x1,
                                             VectorRef out) const {
  const int nv_ = base_.ndx();
  out.resize(ndx());
  base_.difference(getBasePoint(x0), getBasePoint(x1), out.head(nv_));
  out.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
}

template <class Base>
void TangentBundleTpl<Base>::Jintegrate_impl(const ConstVectorRef &x,
                                             const ConstVectorRef &dx,
                                             MatrixRef J_, int arg) const {
  const int nv_ = base_.ndx();
  J_.resize(ndx(), ndx());
  J_.setZero();
  base_.Jintegrate(getBasePoint(x), getBaseTangent(dx), getBaseJacobian(J_),
                   arg);
  J_.bottomRightCorner(nv_, nv_).setIdentity();
}

template <class Base>
void TangentBundleTpl<Base>::Jdifference_impl(const ConstVectorRef &x0,
                                              const ConstVectorRef &x1,
                                              MatrixRef J_, int arg) const {
  const int nv_ = base_.ndx();
  J_.resize(ndx(), ndx());
  J_.setZero();
  base_.Jdifference(getBasePoint(x0), getBasePoint(x1), getBaseJacobian(J_),
                    arg);
  if (arg == 0) {
    J_.bottomRightCorner(nv_, nv_).diagonal().array() = Scalar(-1);
  } else if (arg == 1) {
    J_.bottomRightCorner(nv_, nv_).setIdentity();
  }
}

} // namespace proxnlp
