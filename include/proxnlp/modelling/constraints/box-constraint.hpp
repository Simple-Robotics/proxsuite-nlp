/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/constraint-base.hpp"

namespace proxnlp {

/**
 * @brief   Box constraint set \f$z \in [z_\min, z_\max]\f$.
 *
 */
template <typename Scalar> struct BoxConstraintTpl : ConstraintSetBase<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ConstraintSetBase<Scalar>;
  using ActiveType = typename Base::ActiveType;

  VectorXs lower_limit;
  VectorXs upper_limit;

  BoxConstraintTpl(const ConstVectorRef lower, const ConstVectorRef upper)
      : Base(), lower_limit(lower), upper_limit(upper) {}

  decltype(auto) projection_impl(const ConstVectorRef &z) const {
    return z.cwiseMin(upper_limit).cwiseMax(lower_limit);
  }

  void projection(const ConstVectorRef &z, VectorRef zout) const {
    zout = projection_impl(z);
  }

  void normalConeProjection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z - projection_impl(z);
  }

  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const {
    out.array() =
        (z.array() > upper_limit.array()) || (z.array() < lower_limit.array());
  }
};

} // namespace proxnlp

#include "proxnlp/modelling/constraints.hpp"
