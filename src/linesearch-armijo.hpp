/// @file linesearch-base.hpp
/// @brief  Implements a basic Armijo back-tracking strategy.
#pragma once

#include "proxnlp/linesearch-base.hpp"


namespace proxnlp
{

  /// @brief  Basic backtracking Armijo line-search strategy.
  template<typename _Scalar>
  struct ArmijoLinesearch
  {
    using Scalar = _Scalar;

    /// Directional derivative threshold
    static constexpr Scalar dphi_tresh = 1e-13;

    template<typename Fn>
    static void run(Fn phi,
                    const Scalar phi0,
                    const Scalar dphi0,
                    const VerboseLevel,
                    const Scalar ls_beta,
                    const Scalar armijo_c1,
                    const Scalar alpha_min,
                    Scalar& alpha_try)
    {
      alpha_try = 1.;
      Scalar phi_trial = 0., dM = 0.;

      // try smaller and smaller step sizes
      // until Armijo condition is satisfied

      do
      {
        phi_trial = phi(alpha_try);
        dM = phi_trial - phi0;

        // exit if directional derivative is small
        if (std::abs(dphi0) < dphi_tresh) { return; }

        if (dM <= armijo_c1 * alpha_try * dphi0) { break; }

        alpha_try = std::max(alpha_try * ls_beta, alpha_min);

      } while (alpha_try > alpha_min);

    }

  };
  
} // namespace proxnlp

