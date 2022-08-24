/// @file linesearch-base.hpp
/// @brief  Implements a basic Armijo back-tracking strategy.
#pragma once

#include "proxnlp/linesearch-base.hpp"

namespace proxnlp {

/// @brief  Basic backtracking Armijo line-search strategy.
template <typename Scalar> class ArmijoLinesearch : public Linesearch<Scalar> {
public:
  using Base = Linesearch<Scalar>;
  using FunctionSample = typename Base::FunctionSample;
  using Base::options;
  template <typename Fn>
  void run(Fn phi, const Scalar phi0, const Scalar dphi0, const Scalar ls_beta,
           const Scalar armijo_c1, const Scalar alpha_min, Scalar &alpha_try) {
    auto eval_sample = [&](const Scalar a) {
      return FunctionSample{a, phi(a), 0.};
    };
    alpha_try = 1.;
    FunctionSample try1 = eval_sample(alpha_try);
    Scalar &phi_trial = try1.phi;
    Scalar dM = 0.;

    if (std::abs(dphi0) < options().dphi_thresh) {
      return;
    }

    // try smaller and smaller step sizes
    // until Armijo condition is satisfied
    while (alpha_try > alpha_min) {
      try1 = eval_sample(alpha_try);
      dM = phi_trial - phi0;

      if (dM <= options().armijo_c1 * alpha_try * dphi0) {
        break;
      }

      alpha_try = std::max(alpha_try * ls_beta, alpha_min);
    }
  }
};

} // namespace proxnlp
