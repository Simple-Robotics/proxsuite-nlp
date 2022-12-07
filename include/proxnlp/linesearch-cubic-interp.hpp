/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include "proxnlp/linesearch-base.hpp"

#include <Eigen/LU>

namespace proxnlp {

/// @brief  Backtracking cubic interpolation linesearch.
/// This linesearch searches steplengths through safeguarded interpolation.
/// See Nocedal & Wright Numerical Optimization, sec. 3.5.
template <typename Scalar>
class CubicInterpLinesearch : public Linesearch<Scalar> {
public:
  using Base = Linesearch<Scalar>;
  using FunctionSample = typename Base::FunctionSample;
  using Base::options;
  using Solver = SolverTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;

  template <typename Fn>
  void run(Fn phi, const Scalar phi0, const Scalar dphi0,
           const Scalar armijo_c1, const Scalar alpha_min, Scalar &alpha_try,
           Scalar red_low = 0.4, Scalar red_high = 0.9) {

    auto evaluate_candidate = [&](Scalar alpha) {
      return FunctionSample{alpha, phi(alpha), 0.};
    };

    Scalar alph0 = 1.;
    FunctionSample cand0 = evaluate_candidate(alph0);

    auto check_condition = [&](FunctionSample cand) {
      return cand.phi <= phi0 + armijo_c1 * cand.alpha * dphi0;
    };

    // check termination criterion
    if (check_condition(cand0) || (std::abs(dphi0) < options().dphi_thresh)) {
      alpha_try = cand0.alpha;
      return;
    }

    // step 2: construct & minimize quadratic interpolant
    Scalar al1_den = cand0.phi - phi0 - alph0 * dphi0;
    Scalar alph1 = -0.5 * dphi0 * alph0 * alph0 / al1_den;
    // safeguard
    if ((alph1 > red_high * alph0) || (alph1 < red_low * alph0)) {
      alph1 = alph0 / 2.0;
    }
    FunctionSample cand1 = evaluate_candidate(alph1);

    // check if good, if so exit
    if (check_condition(cand1)) {
      alpha_try = cand1.alpha;
      return;
    }

    using Matrix2s = Eigen::Matrix<Scalar, 2, 2>;
    using Vector2s = Eigen::Matrix<Scalar, 2, 1>;

    // buffers for cubic interpolation
    Matrix2s alph_mat;
    Vector2s alph_rhs;
    Vector2s coeffs_cubic_interpolant;

    for (int i = 0; i < 10; i++) {
      Scalar &a0 = cand0.alpha;
      Scalar &a1 = cand1.alpha;

      alph_mat(0, 0) = a0 * a0;
      alph_mat(0, 1) = -a1 * a1;
      alph_mat(1, 0) = -a0 * a0 * a0;
      alph_mat(1, 1) = a1 * a1 * a1;

      Scalar mat_denom = a0 * a0 * a1 * a1 * (a1 - a0);
      alph_mat /= mat_denom;

      alph_rhs(0) = cand1.phi - phi0 - dphi0 * a1;
      alph_rhs(1) = cand0.phi - phi0 - dphi0 * a0;
      coeffs_cubic_interpolant = alph_mat * alph_rhs;

      Scalar c3 = coeffs_cubic_interpolant(0);
      Scalar c2 = coeffs_cubic_interpolant(1);

      // minimizer of cubic interpolant -> solve dinterp/da = 0
      Scalar anext = (-c2 + std::sqrt(c2 * c2 - 3.0 * c3 * dphi0)) / (3.0 * c3);

      // safeguarding step
      if ((anext > red_high * a1) || (anext < red_low * a1)) {
        anext = a1 / 2.0;
      }

      // update interpolation points
      cand0 = cand1;
      cand1 = evaluate_candidate(anext);

      if (check_condition(cand1)) {
        alpha_try = cand1.alpha;
        break;
      }
    }

    alpha_try = std::max(alpha_try, alpha_min);
  }
};

} // namespace proxnlp
