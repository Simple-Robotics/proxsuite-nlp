#pragma once

#include "proxnlp/modelling/spaces/tangent-bundle.hpp"

namespace proxnlp {

template <class Base>
typename TangentBundleTpl<Base>::PointType
TangentBundleTpl<Base>::neutral() const {
  PointType out(nx());
  out.setZero();
  out.head(m_base.nx()) = m_base.neutral();
  return out;
}

template <class Base>
typename TangentBundleTpl<Base>::PointType
TangentBundleTpl<Base>::rand() const {
  PointType out(nx());
  out.head(m_base.nx()) = m_base.rand();
  using BTanVec_t = typename Base::TangentVectorType;
  out.tail(m_base.ndx()) = BTanVec_t::Random(m_base.ndx());
  return out;
}

/// Operators
template <class Base>
void TangentBundleTpl<Base>::integrate_impl(const ConstVectorRef &x,
                                            const ConstVectorRef &dx,
                                            VectorRef out) const {
  const int nv_ = m_base.ndx();
  m_base.integrate(getBasePoint(x), getBaseTangent(dx), out.head(m_base.nx()));
  out.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
}

template <class Base>
void TangentBundleTpl<Base>::difference_impl(const ConstVectorRef &x0,
                                             const ConstVectorRef &x1,
                                             VectorRef out) const {
  const int nv_ = m_base.ndx();
  out.resize(ndx());
  m_base.difference(getBasePoint(x0), getBasePoint(x1), out.head(nv_));
  out.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
}

template <class Base>
void TangentBundleTpl<Base>::Jintegrate_impl(const ConstVectorRef &x,
                                             const ConstVectorRef &dx,
                                             MatrixRef J_, int arg) const {
  const int nv_ = m_base.ndx();
  J_.resize(ndx(), ndx());
  J_.setZero();
  m_base.Jintegrate(getBasePoint(x), getBaseTangent(dx), getBaseJacobian(J_),
                    arg);
  J_.bottomRightCorner(nv_, nv_).setIdentity();
}

template <class Base>
void TangentBundleTpl<Base>::Jdifference_impl(const ConstVectorRef &x0,
                                              const ConstVectorRef &x1,
                                              MatrixRef J_, int arg) const {
  const int nv_ = m_base.ndx();
  J_.resize(ndx(), ndx());
  J_.setZero();
  m_base.Jdifference(getBasePoint(x0), getBasePoint(x1), getBaseJacobian(J_),
                     arg);
  if (arg == 0) {
    J_.bottomRightCorner(nv_, nv_).diagonal().array() = Scalar(-1);
  } else if (arg == 1) {
    J_.bottomRightCorner(nv_, nv_).setIdentity();
  }
}

} // namespace proxnlp
