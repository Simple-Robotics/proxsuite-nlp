/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
/// @brief     Implementations for the prox solver.
#pragma once

#include "proxnlp/solver-base.hpp"

#include "proxnlp/exceptions.hpp"

#include <fmt/ostream.h>
#include <fmt/color.h>

namespace proxnlp {
template <typename Scalar>
SolverTpl<Scalar>::SolverTpl(shared_ptr<Problem> prob, const Scalar tol,
                             const Scalar mu_init, const Scalar rho_init,
                             const VerboseLevel verbose, const Scalar mu_lower,
                             const Scalar prim_alpha, const Scalar prim_beta,
                             const Scalar dual_alpha, const Scalar dual_beta,
                             const LinesearchOptions ls_options)
    : problem_(prob), merit_fun(problem_, mu_init, 1.),
      prox_penalty(prob->manifold_, manifold().neutral(),
                   rho_init *
                       MatrixXs::Identity(manifold().ndx(), manifold().ndx())),
      verbose(verbose), rho_init_(rho_init), mu_init_(mu_init),
      mu_lower_(mu_lower), bcl_params{prim_alpha, prim_beta, dual_alpha,
                                      dual_beta},
      ls_options(ls_options), target_tol(tol) {}

template <typename Scalar>
ConvergenceFlag SolverTpl<Scalar>::solve(Workspace &workspace, Results &results,
                                         const ConstVectorRef &x0,
                                         const std::vector<VectorRef> &lams0) {
  VectorXs new_lam(problem_->getTotalConstraintDim());
  new_lam.setZero();
  int nr = 0;
  const std::size_t numc = problem_->getNumConstraints();
  if (numc != lams0.size()) {
    PROXNLP_RUNTIME_ERROR("Specified number of constraints is not the same "
                          "as the provided number of multipliers!");
  }
  for (std::size_t i = 0; i < numc; i++) {
    nr = problem_->getConstraintDim(i);
    new_lam.segment(problem_->getIndex(i), nr) = lams0[i];
  }
  return solve(workspace, results, x0, new_lam);
}

template <typename Scalar>
ConvergenceFlag SolverTpl<Scalar>::solve(Workspace &workspace, Results &results,
                                         const ConstVectorRef &x0) {
  VectorXs lams0(workspace.numdual);
  lams0.setZero();
  return solve(workspace, results, x0, lams0);
}

template <typename Scalar>
ConvergenceFlag SolverTpl<Scalar>::solve(Workspace &workspace, Results &results,
                                         const ConstVectorRef &x0,
                                         const ConstVectorRef &lams0) {
  if (verbose == 0)
    logger.active = false;

  setPenalty(mu_init_);
  setProxParameter(rho_init_);

  // init variables
  results.x_opt = x0;
  workspace.x_prev = x0;
  results.data_lams_opt = lams0;
  workspace.data_lams_prev = lams0;

  updateToleranceFailure();

  results.converged = ConvergenceFlag::UNINIT;

  std::size_t &i = results.num_iters;
  std::size_t &al_iter = results.al_iters;
  i = 0;
  al_iter = 0;
  while ((i < max_iters) && (al_iter < max_al_iters)) {
    results.mu = mu_;
    results.rho = rho_;
    if (i == 0) {
      logger.start();
    }
    innerLoop(workspace, results);

    // accept new primal iterate
    workspace.x_prev = results.x_opt;
    prox_penalty.updateTarget(workspace.x_prev);

    if (results.prim_infeas < prim_tol_) {
      acceptMultipliers(results, workspace);
      updateToleranceSuccess();
    } else {
      updatePenalty();
      updateToleranceFailure();
    }
    if (std::max(results.prim_infeas, results.dual_infeas) < target_tol) {
      results.converged = ConvergenceFlag::SUCCESS;
      break;
    }
    setProxParameter(rho_ * bcl_params.rho_update_factor);

    al_iter++;
  }

  if (results.converged == SUCCESS)
    fmt::print(fmt::fg(fmt::color::dodger_blue),
               "Solver successfully converged");

  switch (results.converged) {
  case MAX_ITERS_REACHED:
    fmt::print(fmt::fg(fmt::color::orange_red),
               "Max number of iterations reached.");
    break;
  default:
    break;
  }
  fmt::print("\n");

  invokeCallbacks(workspace, results);

  return results.converged;
}

template <typename Scalar>
typename SolverTpl<Scalar>::InertiaFlag
SolverTpl<Scalar>::checkInertia(const Eigen::VectorXi &signature) const {
  const int ndx = manifold().ndx();
  const int numc = problem_->getTotalConstraintDim();
  const long n = signature.size();
  int numpos = 0;
  int numneg = 0;
  int numzer = 0;
  for (long i = 0; i < n; i++) {
    switch (signature(i)) {
    case 1:
      numpos++;
      break;
    case 0:
      numzer++;
      break;
    case -1:
      numneg++;
      break;
    default:
      PROXNLP_RUNTIME_ERROR(
          "Matrix signature should only have Os, 1s, and -1s.");
    }
  }
  InertiaFlag flag = INERTIA_OK;
  bool pos_ok = numpos == ndx;
  bool neg_ok = numneg == numc;
  bool zer_ok = numzer == 0;
  if (!(pos_ok && neg_ok && zer_ok)) {
    if (!zer_ok)
      flag = INERTIA_HAS_ZEROS;
    else
      flag = INERTIA_BAD;
  } else {
    flag = INERTIA_OK;
  }
  return flag;
}

template <typename Scalar>
void SolverTpl<Scalar>::acceptMultipliers(Results &results,
                                          Workspace &workspace) const {
  switch (mul_up_mode) {
  case MultiplierUpdateMode::NEWTON:
    workspace.data_lams_prev = results.data_lams_opt;
    break;
  case MultiplierUpdateMode::PRIMAL:
    workspace.data_lams_prev = workspace.data_lams_plus;
    break;
  case MultiplierUpdateMode::PRIMAL_DUAL:
    workspace.data_lams_prev = workspace.data_lams_pdal;
    break;
  default:
    break;
  }
  results.data_lams_opt = workspace.data_lams_prev;
}

template <typename Scalar>
void SolverTpl<Scalar>::computeMultipliers(
    const ConstVectorRef &inner_lams_data, Workspace &workspace) const {
  PROXNLP_EIGEN_ALLOW_MALLOC(false);
  workspace.data_shift_cstr_values =
      workspace.data_cstr_values + mu_ * workspace.data_lams_prev;
  // project multiplier estimate
  for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
    const ConstraintSet &cstr_set = *problem_->getConstraint(i).set_;
    // apply proximal op to shifted constraint
    cstr_set.normalConeProjection(workspace.shift_cstr_values[i],
                                  workspace.shift_cstr_proj[i]);
  }
  workspace.data_lams_plus = mu_inv_ * workspace.data_shift_cstr_proj;
  workspace.data_dual_prox_err =
      mu_ * (workspace.data_lams_plus - inner_lams_data);
  workspace.data_lams_pdal = workspace.data_lams_plus +
                             merit_fun.gamma_ * workspace.data_dual_prox_err;
  PROXNLP_EIGEN_ALLOW_MALLOC(true);
}

template <typename Scalar>
void SolverTpl<Scalar>::computeConstraintDerivatives(const ConstVectorRef &x,
                                                     Workspace &workspace,
                                                     bool second_order) const {
  problem_->computeDerivatives(x, workspace);
  if (second_order) {
    problem_->cost().computeHessian(x, workspace.objective_hessian);
  }
  workspace.data_jacobians_proj = workspace.data_jacobians;
  for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
    const ConstraintObject<Scalar> &cstr = problem_->getConstraint(i);
    cstr.set_->applyNormalConeProjectionJacobian(
        workspace.shift_cstr_values[i], workspace.cstr_jacobians_proj[i]);

    bool use_vhp = !cstr.set_->disableGaussNewton() ||
                   (hess_approx == HessianApprox::EXACT);
    if (second_order && use_vhp) {
      cstr.func().vectorHessianProduct(x, workspace.lams_pdal[i],
                                       workspace.cstr_vector_hessian_prod[i]);
    }
  }
}

template <typename Scalar>
void SolverTpl<Scalar>::computePrimalResiduals(Workspace &workspace,
                                               Results &results) const {
  PROXNLP_EIGEN_ALLOW_MALLOC(false);
  workspace.data_shift_cstr_values =
      workspace.data_cstr_values + mu_ * results.data_lams_opt;

  for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
    const ConstraintSet &cstr_set = *problem_->getConstraint(i).set_;
    auto displ_cstr = workspace.shift_cstr_values[i];
    // apply proximal operator
    cstr_set.projection(displ_cstr, displ_cstr);

    auto cstr_prox_err = workspace.cstr_values[i] - displ_cstr;
    results.constraint_violations(long(i)) = math::infty_norm(cstr_prox_err);
  }
  results.prim_infeas = math::infty_norm(results.constraint_violations);
  PROXNLP_EIGEN_ALLOW_MALLOC(true);
}

template <typename Scalar> void SolverTpl<Scalar>::updatePenalty() {
  if (mu_ == mu_lower_) {
    setPenalty(mu_init_);
  } else {
    setPenalty(std::max(mu_ * bcl_params.mu_update_factor, mu_lower_));
  }
}

template <typename Scalar>
void SolverTpl<Scalar>::innerLoop(Workspace &workspace, Results &results) {
  const int ndx = manifold().ndx();
  const long ntot = workspace.kkt_rhs.size();
  const long ndual = ntot - ndx;
  const std::size_t num_c = problem_->getNumConstraints();

  Scalar delta_last = 0.;
  Scalar delta = delta_last;
  Scalar old_delta = 0.;
  Scalar phi_new = 0.;

  // lambda for evaluating the merit function
  auto phi_eval = [&](const Scalar alpha) {
    tryStep(workspace, results, alpha);
    problem_->evaluate(workspace.x_trial, workspace);
    computeMultipliers(workspace.data_lams_trial, workspace);
    return merit_fun.evaluate(workspace.x_trial, workspace.lams_trial,
                              workspace.shift_cstr_values,
                              workspace.shift_cstr_proj) +
           prox_penalty.call(workspace.x_trial);
  };

  while (true) {

    problem_->evaluate(results.x_opt, workspace);
    computeMultipliers(results.data_lams_opt, workspace);
    computeConstraintDerivatives(results.x_opt, workspace, true);

    for (std::size_t i = 0; i < num_c; i++) {
      const ConstraintSet &cstr_set = *problem_->getConstraint(i).set_;
      cstr_set.computeActiveSet(workspace.shift_cstr_values[i],
                                results.active_set[i]);
    }

    results.value = workspace.objective_value;
    results.merit = merit_fun.evaluate(results.x_opt, results.lams_opt,
                                       workspace.shift_cstr_values,
                                       workspace.shift_cstr_proj);

    if (rho_ > 0.) {
      results.merit += prox_penalty.call(results.x_opt);
      prox_penalty.computeGradient(results.x_opt, workspace.prox_grad);
      prox_penalty.computeHessian(results.x_opt, workspace.prox_hess);
    }

    PROXNLP_EIGEN_ALLOW_MALLOC(false);
    //// fill in KKT RHS
    workspace.kkt_rhs.setZero();

    // add jacobian-vector products to gradients
    workspace.kkt_rhs.head(ndx) = workspace.objective_gradient;
    workspace.kkt_rhs.head(ndx).noalias() +=
        workspace.data_jacobians_proj.transpose() * results.data_lams_opt;

    workspace.kkt_rhs.tail(ndual) = workspace.data_dual_prox_err;

    workspace.merit_gradient = workspace.objective_gradient;
    workspace.merit_gradient.noalias() +=
        workspace.data_jacobians.transpose() * workspace.data_lams_plus;

    workspace.dual_residual = workspace.kkt_rhs.head(ndx);

    // add proximal penalty terms
    if (rho_ > 0.) {
      workspace.kkt_rhs.head(ndx) += workspace.prox_grad;
      workspace.merit_gradient += workspace.prox_grad;
    }
    PROXNLP_EIGEN_ALLOW_MALLOC(true);

    PROXNLP_RAISE_IF_NAN_NAME(workspace.kkt_rhs, "kkt_rhs");
    PROXNLP_RAISE_IF_NAN_NAME(workspace.kkt_matrix, "kkt_matrix");

    computePrimalResiduals(workspace, results);
    results.dual_infeas = math::infty_norm(workspace.dual_residual);
    Scalar inner_crit = math::infty_norm(workspace.kkt_rhs);
    Scalar outer_crit = std::max(results.prim_infeas, results.dual_infeas);

    bool inner_cond = inner_crit <= inner_tol_;
    bool outer_cond = outer_crit <= target_tol; // allows early stopping
    if (inner_cond || outer_cond) {
      return;
    }

    // If not optimal: compute the step

    // fill in KKT matrix

    PROXNLP_EIGEN_ALLOW_MALLOC(false);
    workspace.kkt_matrix.setZero();
    workspace.kkt_matrix.topLeftCorner(ndx, ndx) = workspace.objective_hessian;
    workspace.kkt_matrix.topRightCorner(ndx, ndual) =
        workspace.data_jacobians_proj.transpose();
    workspace.kkt_matrix.bottomLeftCorner(ndual, ndx) =
        workspace.data_jacobians_proj;
    workspace.kkt_matrix.bottomRightCorner(ndual, ndual)
        .diagonal()
        .setConstant(-mu_);
    if (rho_ > 0.) {
      workspace.kkt_matrix.topLeftCorner(ndx, ndx) += workspace.prox_hess;
    }
    for (std::size_t i = 0; i < num_c; i++) {
      const ConstraintSet &cstr_set = *problem_->getConstraint(i).set_;
      bool use_vhp = !cstr_set.disableGaussNewton() ||
                     (hess_approx == HessianApprox::EXACT);
      if (use_vhp) {
        workspace.kkt_matrix.topLeftCorner(ndx, ndx) +=
            workspace.cstr_vector_hessian_prod[i];
      }
    }

    // choose regularisation level

    delta = DELTA_INIT;
    InertiaFlag is_inertia_correct = INERTIA_BAD;

    while (!(is_inertia_correct == INERTIA_OK) && delta <= DELTA_MAX) {
      if (delta > 0.) {
        workspace.kkt_matrix.diagonal().head(ndx).array() += delta;
      }
      workspace.ldlt_.compute(workspace.kkt_matrix);
      auto vecD(workspace.ldlt_.vectorD());
      workspace.signature.array() = vecD.array().sign().template cast<int>();
      workspace.kkt_matrix.diagonal().head(ndx).array() -= delta;
      is_inertia_correct = checkInertia(workspace.signature);
      old_delta = delta;

      if (is_inertia_correct == INERTIA_OK) {
        delta_last = delta;
        break;
      } else if (delta == 0.) {
        // check if previous was zero
        if (delta_last == 0.) {
          delta = DELTA_NONZERO_INIT; // try a set nonzero value
        } else {
          delta = std::max(DELTA_MIN, del_dec_k * delta_last);
        }
      } else {
        // check previous; decide increase factor
        if (delta_last == 0.) {
          delta *= del_inc_big;
        } else {
          delta *= del_inc_k;
        }
      }
    }

    workspace.pd_step = -workspace.kkt_rhs;
    workspace.ldlt_.solveInPlace(workspace.pd_step);

    PROXNLP_RAISE_IF_NAN_NAME(workspace.pd_step, "pd_step");

    const std::size_t MAX_REFINEMENT_STEPS = 5;
    for (std::size_t n = 0; n < MAX_REFINEMENT_STEPS; n++) {
      workspace.kkt_err = workspace.kkt_rhs;
      workspace.kkt_err.noalias() += workspace.kkt_matrix * workspace.pd_step;
      workspace.kkt_err *= -1.;
      Scalar resdl_norm = math::infty_norm(workspace.kkt_err);
      if (resdl_norm < 1e-13)
        break;
      workspace.ldlt_.solveInPlace(workspace.kkt_err);
      workspace.pd_step += workspace.kkt_err;
    }
    PROXNLP_EIGEN_ALLOW_MALLOC(true);

    //// Take the step

    workspace.dmerit_dir = workspace.merit_gradient.dot(workspace.prim_step) -
                           merit_fun.gamma_ * workspace.data_dual_prox_err.dot(
                                                  workspace.dual_step);

    Scalar phi0 = results.merit;
    Scalar dphi0 = workspace.dmerit_dir;
    switch (ls_strat) {
    case LinesearchStrategy::ARMIJO: {
      phi_new = ArmijoLinesearch<Scalar>(ls_options)
                    .run(phi_eval, results.merit, dphi0, workspace.alpha_opt);
      break;
    }
    default:
      PROXNLP_RUNTIME_ERROR("Unrecognized linesearch alternative.\n");
      break;
    }

    tryStep(workspace, results, workspace.alpha_opt);

    PROXNLP_RAISE_IF_NAN_NAME(workspace.alpha_opt, "alpha_opt");
    PROXNLP_RAISE_IF_NAN_NAME(workspace.x_trial, "x_trial");
    PROXNLP_RAISE_IF_NAN_NAME(workspace.data_lams_trial, "lams_trial");
    results.x_opt = workspace.x_trial;
    results.data_lams_opt = workspace.data_lams_trial;
    results.merit = phi_new;
    PROXNLP_RAISE_IF_NAN_NAME(results.merit, "merit");

    invokeCallbacks(workspace, results);

    LogRecord record{results.num_iters + 1,
                     workspace.alpha_opt,
                     inner_crit,
                     results.prim_infeas,
                     results.dual_infeas,
                     delta,
                     dphi0,
                     results.merit,
                     phi_new - phi0,
                     results.al_iters};

    logger.log(record);

    results.num_iters++;
    if (results.num_iters >= max_iters) {
      results.converged = ConvergenceFlag::MAX_ITERS_REACHED;
      break;
    }
  }

  return;
}

template <typename Scalar>
void SolverTpl<Scalar>::setPenalty(const Scalar &new_mu) noexcept {
  mu_ = new_mu;
  mu_inv_ = 1. / mu_;
  merit_fun.setPenalty(mu_);
  for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
    const ConstraintObject<Scalar> &cstr = problem_->getConstraint(i);
    cstr.set_->setProxParameters(mu_);
  }
}

template <typename Scalar>
void SolverTpl<Scalar>::setProxParameter(const Scalar &new_rho) noexcept {
  rho_ = new_rho;
  prox_penalty.weights_.setZero();
  prox_penalty.weights_.diagonal().setConstant(rho_);
}

template <typename Scalar>
void SolverTpl<Scalar>::updateToleranceFailure() noexcept {
  prim_tol_ = prim_tol0 * std::pow(mu_, bcl_params.prim_alpha);
  inner_tol_ = inner_tol0 * std::pow(mu_, bcl_params.dual_alpha);
  tolerancePostUpdate();
}

template <typename Scalar>
void SolverTpl<Scalar>::updateToleranceSuccess() noexcept {
  prim_tol_ = prim_tol_ * std::pow(mu_ / mu_upper_, bcl_params.prim_beta);
  inner_tol_ = inner_tol_ * std::pow(mu_ / mu_upper_, bcl_params.dual_beta);
  tolerancePostUpdate();
}

template <typename Scalar>
void SolverTpl<Scalar>::tolerancePostUpdate() noexcept {
  inner_tol_ = std::max(inner_tol_, inner_tol_min);
  prim_tol_ = std::max(prim_tol_, target_tol);
}

template <typename Scalar>
void SolverTpl<Scalar>::tryStep(Workspace &workspace, const Results &results,
                                Scalar alpha) {
  PROXNLP_EIGEN_ALLOW_MALLOC(false);
  workspace.tmp_dx_scaled = alpha * workspace.prim_step;
  manifold().integrate(results.x_opt, workspace.tmp_dx_scaled,
                       workspace.x_trial);
  workspace.data_lams_trial =
      results.data_lams_opt + alpha * workspace.dual_step;
  PROXNLP_EIGEN_ALLOW_MALLOC(true);
}
} // namespace proxnlp
