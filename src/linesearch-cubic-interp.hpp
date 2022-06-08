#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include "proxnlp/linesearch-base.hpp"

#include <Eigen/LU>


namespace proxnlp
{

  /// @brief  Bracketing quadratic-interpolation linesearch.
  template<typename _Scalar>
  struct CubicInterpLinesearch
  {
    using Scalar = _Scalar;
    using Solver = SolverTpl<Scalar>;
    using Workspace = WorkspaceTpl<Scalar>;
    using Results = ResultsTpl<Scalar>;
    struct ls_candidate
    {
      Scalar alpha;
      Scalar phi;
    };

    static void run(const Solver& solver,
                    Workspace& workspace,
                    const Results& results,
                    const Scalar phi0, const Scalar dphi0,
                    VerboseLevel verbosity,
                    Scalar& alpha_try)
    {
      auto eval = [&](auto alpha)
      {
        return ls_candidate {
          alpha, compute_merit_for_step(solver, workspace, results, alpha)
        };
      };

      Scalar aleft = 0.;

      ls_candidate cand0 = eval(1.);
      Scalar& a0 = cand0.alpha;

      Scalar armijo_c1 = solver.armijo_c1;

      auto check_cond = [&](auto cand)
      {
        return (cand.phi <= phi0 + armijo_c1 * cand.alpha * dphi0) || (dphi0 < 1e-13);
      };

      // check termination criterion
      if (check_cond(cand0))
      {
        alpha_try = cand0.alpha;
        fmt::print("  Accepted initial step.\n");
        return;
      }

      // minimize quadratic interpolant
      Scalar coeff2 = (cand0.phi - phi0 - cand0.alpha * dphi0) / std::pow(cand0.alpha, 2);
      ls_candidate cand1 = eval(- .5 * dphi0 / coeff2);
      Scalar& a1 = cand1.alpha;

      if (check_cond(cand1))
      {
        fmt::print("  Accepted quad interp sol. {}", cand1.alpha);
        alpha_try = a1;
        return;
      }

      // if not sufficient: use cubic interpolation
      Eigen::Matrix<Scalar, 2, 2> alphMat;
      Eigen::Vector2<Scalar> rhs;

      for (int i = 0; i < 10; i++)
      {

        alphMat(0, 0) = a0 * a0;
        alphMat(0, 1) = -a1 * a1;
        alphMat(1, 0) = -a0 * a0 * a0;
        alphMat(1, 1) = a1 * a1 * a1;

        Scalar den = a0 * a0 * a1 * a1 * (a1 - a0);
        rhs(0) = cand1.phi - phi0 - dphi0 * a1;
        rhs(1) = cand0.phi - phi0 - dphi0 * a0;
        alphMat.noalias() = alphMat / den;

        Eigen::Vector2<Scalar> newCoeffs = alphMat.fullPivLu().solve(rhs);

        Scalar c3 = newCoeffs(0);
        Scalar c2 = newCoeffs(1);

        Scalar anext = (-c2 + std::sqrt(c2 * c2 - 3 * c3 * dphi0));
        anext = anext / (3 * c3);

        // update interpolation points
        cand0 = cand1;
        cand1 = eval(anext);
        fmt::print("  trying step size {}\n", anext);

        if (check_cond(cand1))
        {
          alpha_try = cand1.alpha;
          break;
        }

      }

      alpha_try = std::max(alpha_try, solver.alpha_min);

    }

    static void bounded_interp(Scalar alow, Scalar ahigh)
    {

    }

  };
  
} // namespace proxnlp