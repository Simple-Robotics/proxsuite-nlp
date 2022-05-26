#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include <fmt/core.h>


namespace proxnlp
{
  /// @brief  Basic backtracking Armijo linesearch.
  template<typename _Scalar>
  struct ArmijoLinesearch
  {
    using Scalar = _Scalar;
    using Workspace = WorkspaceTpl<Scalar>;
    using Results = ResultsTpl<Scalar>;
    using Solver = SolverTpl<Scalar>;

    /// Directional derivative threshold
    static constexpr Scalar dphi_tresh = 1e-13;

    static void run(const Solver& solver, Workspace& workspace,
                    const Results& results, const Scalar phi0, const Scalar dphi0,
                    VerboseLevel verbosity,
                    Scalar& alpha_try)
    {
      alpha_try = 1.;
      const Scalar ls_beta = solver.ls_beta;
      Scalar phi_trial = 0., dM = 0.;
      const auto& manifold = solver.manifold;
      const Scalar armijo_c1 = solver.armijo_c1;

      Solver::tryStep(manifold, workspace, results, alpha_try);

      // exit if directional derivative is small
      if (std::abs(dphi0) < dphi_tresh) { return; }

      // try smaller and smaller step sizes
      // until Armijo condition is satisfied

      do
      {
        phi_trial = solver.merit_fun(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev);
        if (solver.rho > 0.) {
          phi_trial += solver.prox_penalty.call(workspace.xTrial);
        }
        dM = phi_trial - phi0;

        if (dM <= armijo_c1 * alpha_try * dphi0) { break; }

        alpha_try = std::max(alpha_try * ls_beta, solver.alpha_min);
        Solver::tryStep(manifold, workspace, results, alpha_try);

      } while (alpha_try > solver.alpha_min);

      if (verbosity >= 1)
      {
        fmt::print(" | alpha*={:5.3e}\n", alpha_try);
      }
    }

  };
} // namespace proxnlp

