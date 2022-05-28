#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include "proxnlp/linesearch-base.hpp"


namespace proxnlp
{
  /// @brief  Bracketing quadratic-interpolation linesearch.
  template<typename _Scalar>
  struct QuadInterpLinesearch
  {
    using Scalar = _Scalar;
    using Solver = SolverTpl<Scalar>;

    static void run(const Solver& solver, WorkspaceTpl<Scalar>& workspace,
                    const ResultsTpl<Scalar>& results, const Scalar phi0, const Scalar dphi0,
                    VerboseLevel verbosity,
                    Scalar& alpha_try)
    {
      Scalar a0 = 0.;
      Scalar a1 = 1.;

      const auto& manifold = solver.manifold;

      Scalar phi_a = compute_merit_for_step(solver, workspace, results, alpha_try);

      // minimum of the quadratic through 0, alpha, with dphi0
      Scalar amin_ = 

    }
  };
  
} // namespace proxnlp
