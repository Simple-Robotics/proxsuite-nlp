/* Copyright (C) 2022 LAAS-CNRS, INRIA
 *
 */
#pragma once

#include "proxnlp/problem-base.hpp"

#include <fmt/ostream.h>

namespace proxnlp {

enum ConvergenceFlag { SUCCESS = 0, MAX_ITERS_REACHED = 1 };

/**
 * @brief   Results struct, holding the returned data from the solver.
 *
 * @details This struct holds the current (and output) primal-dual point,
 *          the optimal proximal parameters \f$(\rho, \mu)\f$.
 */
template <typename _Scalar> struct ResultsTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;
  using VecBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

  ConvergenceFlag converged = SUCCESS;

  Scalar merit;
  Scalar value;
  VectorXs x_opt;
  VectorXs lams_opt_data;
  VectorOfRef lams_opt;
  /// Current active set of the algorithm.
  std::vector<VecBool> active_set;
  Scalar dual_infeas = 0.;
  Scalar prim_infeas = 0.;
  /// Violations for each constraint
  VectorXs constraint_violations;

  /// Final solver parameters
  std::size_t num_iters = 0;
  Scalar mu;
  Scalar rho;

  ResultsTpl(const Problem &prob)
      : x_opt(prob.manifold_->neutral()),
        lams_opt_data(prob.getTotalConstraintDim()),
        constraint_violations(prob.getNumConstraints()), num_iters(0), mu(0.),
        rho(0.) {
    helpers::allocateMultipliersOrResiduals(prob, lams_opt_data, lams_opt);
    constraint_violations.setZero();
    active_set.reserve(prob.getNumConstraints());
    for (std::size_t i = 0; i < prob.getNumConstraints(); i++) {
      active_set.push_back(VecBool::Zero(prob.getConstraint(i).func().nr()));
    }
  }

  friend std::ostream &operator<<(std::ostream &oss,
                                  const ResultsTpl<Scalar> &self) {
    oss << "Results {" << fmt::format("\n  convergence:   {},", self.converged)
        << fmt::format("\n  merit:         {:.3e},", self.merit)
        << fmt::format("\n  value:         {:.3e},", self.value)
        << fmt::format("\n  num_iters:     {:d},", self.num_iters)
        << fmt::format("\n  mu:            {:.3e},", self.mu)
        << fmt::format("\n  rho:           {:.3e},", self.rho)
        << fmt::format("\n  dual_infeas:   {:.3e},", self.dual_infeas)
        << fmt::format("\n  prim_infeas:   {:.3e},", self.prim_infeas)
        << fmt::format("\n  cstr_values:   {}",
                       self.constraint_violations.transpose());
    for (std::size_t i = 0; i < self.active_set.size(); i++) {
      oss << fmt::format("\n  activeSet[{:d}]:  {}", i,
                         self.active_set[i].transpose());
    }
    oss << "\n}";
    return oss;
  }
};

} // namespace proxnlp
