/// @file linesearch-base.hpp
/// @brief  Base structs for linesearch algorithms, and implements a basic Armijo back-tracking
///         strategy.
#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include <fmt/core.h>


namespace proxnlp
{

  template<typename Scalar>
  Scalar compute_merit_for_step(const SolverTpl<Scalar>& solver, WorkspaceTpl<Scalar>& workspace, const ResultsTpl<Scalar>& results, const Scalar a0)
  {
    SolverTpl<Scalar>::tryStep(solver.manifold, workspace, results, a0);
    return solver.merit_fun(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev) + solver.prox_penalty.call(workspace.xTrial);
  }

  /// Recompute the line-search directional derivative at the current trial point.
  ///
  /// @param solver Solver instance; tied to the merit function and derivatives evaluation.
  /// @param workspace Problem workspace; the trial point is a data member.
  template<typename Scalar>
  Scalar recompute_merit_derivative_at_trial_point(const SolverTpl<Scalar>& solver, WorkspaceTpl<Scalar>& workspace)
  {
    solver.computeResidualsAndMultipliers(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev);
    solver.computeResidualDerivatives(workspace.xTrial, workspace, false);

    workspace.meritGradient = workspace.objectiveGradient + workspace.jacobians_data.transpose() * workspace.lamsPDAL_data;
    workspace.meritGradient.noalias() += workspace.prox_grad;
    Scalar d1_new = workspace.meritGradient.dot(workspace.prim_step) \
      - workspace.dual_prox_err_data.dot(workspace.dual_step);
    return d1_new;
  }

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

