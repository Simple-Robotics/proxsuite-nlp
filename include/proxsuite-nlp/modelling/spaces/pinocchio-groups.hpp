#pragma once

#include "proxsuite-nlp/manifold-base.hpp"

#include <pinocchio/multibody/liegroup/liegroup-base.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace proxsuite {
namespace nlp {

namespace pin = pinocchio;

/** @brief  Wrap a Pinocchio Lie group into a ManifoldAbstractTpl object.
 *
 *
 */
template <typename _LieGroup>
struct PinocchioLieGroup
    : public ManifoldAbstractTpl<typename _LieGroup::Scalar,
                                 _LieGroup::Options> {
public:
  using LieGroup = _LieGroup;
  using Scalar = typename LieGroup::Scalar;
  enum { Options = LieGroup::Options };
  using Base = ManifoldAbstractTpl<Scalar, Options>;
  PROXSUITE_NLP_DEFINE_MANIFOLD_TYPES(Base)

  LieGroup lg_;
  PinocchioLieGroup() {}
  PinocchioLieGroup(const LieGroup &lg) : lg_(lg) {}

  template <typename... Args> PinocchioLieGroup(Args... args) : lg_(args...) {}

  inline int nx() const { return lg_.nq(); }
  inline int ndx() const { return lg_.nv(); }

  /// \name Implementations

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const {
    lg_.integrate(x, v, out);
  }

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef vout) const {
    lg_.difference(x0, x1, vout);
  }

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      lg_.dIntegrate_dq(x, v, Jout);
      break;
    case 1:
      lg_.dIntegrate_dv(x, v, Jout);
      break;
    }
  }

  void JintegrateTransport(const ConstVectorRef &x, const ConstVectorRef &v,
                           MatrixRef Jout, int arg) const {
    lg_.dIntegrateTransport(x, v, Jout, pin::ArgumentPosition(arg));
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      lg_.dDifference(x0, x1, Jout, pin::ARG0);
      break;
    case 1:
      lg_.dDifference(x0, x1, Jout, pin::ARG1);
      break;
    }
  }

  virtual void interpolate_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, const Scalar &u,
                                VectorRef out) const {
    lg_.interpolate(x0, x1, u, out);
  }

  PointType neutral() const { return lg_.neutral(); }

  PointType rand() const { return lg_.random(); }
  bool isNormalized(const ConstVectorRef &x) const {
    if (x.size() < nx())
      return false;
    return lg_.isNormalized(x);
  }
};

template <int D, typename Scalar>
using SETpl =
    PinocchioLieGroup<pinocchio::SpecialEuclideanOperationTpl<D, Scalar>>;

template <int D, typename Scalar>
using SOTpl =
    PinocchioLieGroup<pinocchio::SpecialOrthogonalOperationTpl<D, Scalar>>;

} // namespace nlp
} // namespace proxsuite
