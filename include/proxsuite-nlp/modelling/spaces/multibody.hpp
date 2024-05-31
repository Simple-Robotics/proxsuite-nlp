/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include "proxsuite-nlp/modelling/spaces/tangent-bundle.hpp"

namespace proxsuite {
namespace nlp {

/** @brief    Multibody configuration group \f$\mathcal{Q}\f$, defined using the
 * Pinocchio library.
 *
 *  @details  This uses a pinocchio::ModelTpl object to define the manifold.
 */
template <typename _Scalar, int _Options = 0>
struct MultibodyConfiguration : public ManifoldAbstractTpl<_Scalar, _Options> {
public:
  using Scalar = _Scalar;
  enum { Options = _Options };
  using Self = MultibodyConfiguration<Scalar, Options>;
  using ModelType = pinocchio::ModelTpl<Scalar, Options>;
  using Base = ManifoldAbstractTpl<Scalar, Options>;
  PROXSUITE_NLP_DEFINE_MANIFOLD_TYPES(Base)

  MultibodyConfiguration(const ModelType &model) : model_(model) {};

  const ModelType &getModel() const { return model_; }

  PointType neutral() const { return pinocchio::neutral(model_); }
  PointType rand() const { return pinocchio::randomConfiguration(model_); }
  bool isNormalized(const ConstVectorRef &x) const {
    return pinocchio::isNormalized(model_, x);
  }

  /// \name implementations
  /// \{

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef xout) const {
    pinocchio::integrate(model_, x, v, xout);
  }

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      pinocchio::dIntegrate(model_, x, v, Jout, pinocchio::ARG0);
      break;
    case 1:
      pinocchio::dIntegrate(model_, x, v, Jout, pinocchio::ARG1);
      break;
    }
  }

  void JintegrateTransport(const ConstVectorRef &x, const ConstVectorRef &v,
                           MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      pinocchio::dIntegrateTransport(model_, x, v, Jout, pinocchio::ARG0);
      break;
    case 1:
      pinocchio::dIntegrateTransport(model_, x, v, Jout, pinocchio::ARG1);
      break;
    default:
      break;
    }
  }

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef vout) const {
    pinocchio::difference(model_, x0, x1, vout);
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      pinocchio::dDifference(model_, x0, x1, Jout, pinocchio::ARG0);
      break;
    case 1:
      pinocchio::dDifference(model_, x0, x1, Jout, pinocchio::ARG1);
      break;
    }
  }

  void interpolate_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        const Scalar &u, VectorRef out) const {
    pinocchio::interpolate(model_, x0, x1, u, out);
  }

  inline int nx() const { return model_.nq; }
  inline int ndx() const { return model_.nv; }

  /// \}

protected:
  ModelType model_;
};

/** @brief      The tangent bundle of a multibody configuration group.
 *  @details    This is not a typedef, since we provide a constructor for the
 * class. Any point on the manifold is of the form \f$x = (q,v) \f$, where \f$q
 * \in \mathcal{Q} \f$ is a configuration and \f$v\f$ is a joint velocity
 * vector.
 */
template <typename Scalar, int Options = 0>
struct MultibodyPhaseSpace
    : TangentBundleTpl<MultibodyConfiguration<Scalar, Options>> {
  using ConfigSpace = MultibodyConfiguration<Scalar, Options>;
  using ModelType = typename ConfigSpace::ModelType;

  const ModelType &getModel() const { return this->base_.getModel(); }

  MultibodyPhaseSpace(const ModelType &model)
      : TangentBundleTpl<ConfigSpace>(ConfigSpace(model)) {}
};

} // namespace nlp
} // namespace proxsuite

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/modelling/spaces/multibody.txx"
#endif
