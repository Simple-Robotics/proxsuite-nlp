/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

namespace proxnlp {
template <typename Scalar> struct BCLParamsTpl {

  /// Log-factor \f$\alpha_\eta\f$ for primal tolerance (failure)
  Scalar prim_alpha = 0.1;
  /// Log-factor \f$\beta_\eta\f$ for primal tolerance (success)
  Scalar prim_beta = 0.9;
  /// Log-factor \f$\alpha_\eta\f$ for dual tolerance (failure)
  Scalar dual_alpha = 1.;
  /// Log-factor \f$\beta_\eta\f$ for dual tolerance (success)
  Scalar dual_beta = 1.;
  /// Scale factor for the dual proximal penalty.
  Scalar mu_update_factor = 0.01;
  /// Scale factor for the primal proximal penalty.
  Scalar rho_update_factor = 1.0;
};

} // namespace proxnlp
