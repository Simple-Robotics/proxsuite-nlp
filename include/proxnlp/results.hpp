/* Copyright (C) 2022 LAAS-CNRS, INRIA
 *
 */
#pragma once

#include "proxnlp/problem-base.hpp"

namespace proxnlp {

enum ConvergenceFlag { UNINIT = -1, SUCCESS = 0, MAX_ITERS_REACHED = 1 };

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

  ConvergenceFlag converged = ConvergenceFlag::UNINIT;

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

  ResultsTpl(const int nx, const Problem &prob)
      : x_opt(nx), lams_opt_data(prob.getTotalConstraintDim()),
        constraint_violations(prob.getNumConstraints()), num_iters(0), mu(0.),
        rho(0.) {
    x_opt.setZero();
    helpers::allocateMultipliersOrResiduals(prob, lams_opt_data, lams_opt);
    constraint_violations.setZero();
    active_set.reserve(prob.getNumConstraints());
    for (std::size_t i = 0; i < prob.getNumConstraints(); i++) {
      active_set.push_back(VecBool::Zero(prob.getConstraint(i).func().nr()));
    }
  }

  friend std::ostream &operator<<(std::ostream &s,
                                  const ResultsTpl<Scalar> &self) {
    s << "{\n"
      << "  convergence:   " << self.converged << ",\n"
      << "  merit:         " << self.merit << ",\n"
      << "  value:         " << self.value << ",\n"
      << "  num_iters:      " << self.num_iters << ",\n"
      << "  mu:            " << self.mu << ",\n"
      << "  rho:           " << self.rho << ",\n"
      << "  dual_infeas:   " << self.dual_infeas << ",\n"
      << "  primal_infeas: " << self.prim_infeas << ",\n"
      << "  cstr_values:   " << self.constraint_violations.transpose();
    for (std::size_t i = 0; i < self.active_set.size(); i++) {
      s << ",\n  activeSet[" << i << "]:  " << self.active_set[i].transpose();
    }
    s << "\n}";
    return s;
  }
};

} // namespace proxnlp
