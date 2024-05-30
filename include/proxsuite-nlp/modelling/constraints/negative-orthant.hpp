/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/constraint-base.hpp"

namespace proxsuite {
namespace nlp {

/**
 * @brief   Negative orthant, for constraints \f$h(x)\leq 0\f$.
 *
 * Negative orthant, corresponding to constraints of the form
 * \f[
 *    h(x) \leq 0
 * \f]
 * where \f$h : \mathcal{X} \to \RR^p\f$ is a given residual.
 */
template <typename _Scalar>
struct NegativeOrthantTpl : ConstraintSetBase<_Scalar> {
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Base = ConstraintSetBase<Scalar>;
  using ActiveType = typename Base::ActiveType;

  void projection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z.cwiseMin(static_cast<Scalar>(0.));
  }

  void normalConeProjection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z.cwiseMax(static_cast<Scalar>(0.));
  }

  /// The elements of the active set are the components such that \f$z_i > 0\f$.
  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const {
    out.array() = (z.array() > static_cast<Scalar>(0.));
  }
};

template <typename Scalar>
using NegativeOrthant PROXSUITE_NLP_DEPRECATED_MESSAGE(
    "Use NegativeOrthantTpl<T> instead") = NegativeOrthantTpl<Scalar>;

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
extern template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    NegativeOrthantTpl<context::Scalar>;
#endif

} // namespace nlp
} // namespace proxsuite

#include "proxsuite-nlp/modelling/constraints.hpp"
