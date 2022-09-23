/// @file solver-base.hxx
/// Implementations for the prox solver.
#pragma once

#include "proxnlp/solver-base.hpp"

#include "proxnlp/exceptions.hpp"

#include <fmt/ostream.h>
#include <fmt/color.h>

namespace proxnlp {
template <typename Scalar>
SolverTpl<Scalar>::SolverTpl(const Manifold &manifold,
                             const shared_ptr<Problem> &prob, const Scalar tol,
                             const Scalar mu_init, const Scalar rho_init,
                             const VerboseLevel verbose, const Scalar mu_lower,
                             const Scalar prim_alpha, const Scalar prim_beta,
                             const Scalar dual_alpha, const Scalar dual_beta,
                             const LSOptions ls_options)
    : manifold(manifold), problem_(prob), merit_fun(problem_, mu_init),
      prox_penalty(manifold, manifold.neutral(),
                   rho_init *
                       MatrixXs::Identity(manifold.ndx(), manifold.ndx())),
      verbose(verbose), rho_init_(rho_init), mu_init_(mu_init),
      mu_lower_(mu_lower), target_tol(tol), prim_alpha_(prim_alpha),
      prim_beta(prim_beta), dual_alpha(dual_alpha), dual_beta(dual_beta),
      ls_options(ls_options) {}

template <typename Scalar>
ConvergenceFlag SolverTpl<Scalar>::solve(Workspace &workspace, Results &results,
                                         const ConstVectorRef &x0,
                                         const std::vector<VectorRef> &lams0) {
  VectorXs new_lam(problem_->getTotalConstraintDim());
  new_lam.setZero();
  int nr = 0;
  const std::size_t numc = problem_->getNumConstraints();
  if (numc != lams0.size()) {
    proxnlp_runtime_error("Specified number of constraints is not the same "
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

  // init variables
  results.xOpt = x0;
  workspace.xPrev = x0;
  results.lams_opt_data = lams0;
  workspace.data_lams_prev = lams0;

  updateToleranceFailure();

  results.numIters = 0;

  auto outer_col = fmt::color::white;

  std::size_t i = 0;
  while (results.numIters < MAX_ITERS) {
    results.mu = mu_;
    results.rho = rho_;
    fmt::print(fmt::emphasis::bold | fmt::fg(outer_col),
               "[AL iter {:>2d}] omega={:.3g}, eta={:.3g}, mu={:g}\n", i,
               inner_tol, prim_tol, mu_);
    if (results.numIters == 0) {
      logger.start();
    }
    solveInner(workspace, results);

    // accept new primal iterate
    workspace.xPrev = results.xOpt;
    prox_penalty.updateTarget(workspace.xPrev);

    if (results.primalInfeas < prim_tol) {
      outer_col = fmt::color::lime_green;
      acceptMultipliers(workspace);
      if ((results.primalInfeas < target_tol) &&
          (results.dualInfeas < target_tol)) {
        // terminate algorithm
        results.converged = ConvergenceFlag::SUCCESS;
        break;
      }
      updateToleranceSuccess();
    } else {
      outer_col = fmt::color::orange_red;
      updatePenalty();
      updateToleranceFailure();
    }
    // safeguard tolerances
    inner_tol = std::max(inner_tol, inner_tol_min);

    i++;
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
  const int ndx = manifold.ndx();
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
      proxnlp_runtime_error(
          "Matrix signature should only have Os, 1s, and -1s.");
    }
  }
  InertiaFlag flag = OK;
  bool print_info = verbose >= 2;
  // if (print_info)
  //   fmt::print(" | Inertia: {:d}+, {:d}, {:d}-", numpos, numzer, numneg);
  bool pos_ok = numpos == ndx;
  bool neg_ok = numneg == numc;
  bool zer_ok = numzer == 0;
  if (!(pos_ok && neg_ok && zer_ok)) {
    if (!zer_ok)
      flag = ZEROS;
    else
      flag = BAD;
  } else {
    flag = OK;
  }
  return flag;
}

template <typename Scalar>
void SolverTpl<Scalar>::computeConstraintsAndMultipliers(
    const ConstVectorRef &x, const ConstVectorRef &lams_data,
    Workspace &workspace) const {
  problem_->evaluate(x, workspace);
  workspace.data_lams_plus_pre =
      workspace.data_lams_prev + mu_inv_ * workspace.data_cstr_values;
  // project multiplier estimate
  for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
    const ConstraintSetBase<Scalar> &cstr_set =
        *problem_->getConstraint(i).m_set;
    cstr_set.normalConeProjection(workspace.lamsPlusPre[i],
                                  workspace.lamsPlus[i]);
  }
  workspace.dual_prox_err_data = mu_ * (workspace.data_lams_plus - lams_data);
  workspace.data_lams_pdal = 2 * workspace.data_lams_plus - lams_data;
}

/// Compute problem derivatives
template <typename Scalar>
void SolverTpl<Scalar>::computeConstraintDerivatives(const ConstVectorRef &x,
                                                     Workspace &workspace,
                                                     bool second_order) const {
  problem_->computeDerivatives(x, workspace);
  if (second_order) {
    problem_->cost().computeHessian(x, workspace.objectiveHessian);
  }
  for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
    const ConstraintObject<Scalar> &cstr = problem_->getConstraint(i);
    cstr.m_set->applyNormalConeProjectionJacobian(workspace.lamsPlusPre[i],
                                                  workspace.cstrJacobians[i]);

    bool use_vhp = (use_gauss_newton && !(cstr.m_set->disableGaussNewton())) ||
                   !(use_gauss_newton);
    if (second_order && use_vhp) {
      cstr.func().vectorHessianProduct(x, workspace.lamsPDAL[i],
                                       workspace.cstrVectorHessianProd[i]);
    }
  }
  if (rho_ > 0.) {
    prox_penalty.computeGradient(x, workspace.prox_grad);
    if (second_order)
      prox_penalty.computeHessian(x, workspace.prox_hess);
  }
}

template <typename Scalar> void SolverTpl<Scalar>::updatePenalty() {
  if (mu_ == mu_lower_) {
    setPenalty(mu_init_);
  } else {
    setPenalty(std::max(mu_ * mu_factor_, mu_lower_));
  }
  for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
    const ConstraintObject<Scalar> &cstr = problem_->getConstraint(i);
    cstr.m_set->setProxParameters(mu_);
  }
}

template <typename Scalar>
void SolverTpl<Scalar>::solveInner(Workspace &workspace, Results &results) {
  const int ndx = manifold.ndx();
  const long ntot = workspace.kktRhs.size();
  const long ndual = ntot - ndx;
  const std::size_t num_c = problem_->getNumConstraints();

  results.lams_opt_data = workspace.data_lams_prev;

  Scalar delta_last = 0.;
  Scalar delta = delta_last;
  Scalar old_delta = 0.;
  Scalar conditioning = 0;

  merit_fun.setPenalty(mu_);

  // lambda for evaluating the merit function
  auto phi_eval = [&](const Scalar alpha) {
    tryStep(manifold, workspace, results, alpha);
    return merit_fun.evaluate(workspace.xTrial, workspace.lamsTrial,
                              workspace.lamsPrev, workspace.lamsPlusPre) +
           prox_penalty.call(workspace.xTrial);
  };

  std::size_t k;
  for (k = 0; k < MAX_ITERS; k++) {

    //// precompute temp data

    results.value = problem_->cost().call(results.xOpt);

    computeConstraintsAndMultipliers(results.xOpt, results.lams_opt_data,
                                     workspace);
    computeConstraintDerivatives(results.xOpt, workspace, true);

    results.merit =
        merit_fun.evaluate(results.xOpt, results.lamsOpt, workspace.lamsPrev,
                           workspace.lamsPlusPre);
    if (rho_ > 0.)
      results.merit += prox_penalty.call(results.xOpt);

    //// fill in LHS/RHS
    //// TODO create an Eigen::Map to map submatrices to the active sets of each
    /// constraint

    workspace.kktRhs.setZero();
    workspace.kktMatrix.setZero();

    workspace.kktMatrix.topLeftCorner(ndx, ndx) = workspace.objectiveHessian;
    workspace.kktMatrix.topRightCorner(ndx, ndual) =
        workspace.jacobians_data.transpose();
    workspace.kktMatrix.bottomLeftCorner(ndual, ndx) = workspace.jacobians_data;
    workspace.kktMatrix.bottomRightCorner(ndual, ndual)
        .diagonal()
        .setConstant(-mu_);

    // add jacobian-vector products to gradients
    workspace.kktRhs.head(ndx) =
        workspace.objectiveGradient +
        workspace.jacobians_data.transpose() * results.lams_opt_data;
    workspace.kktRhs.tail(ndual) = workspace.dual_prox_err_data;
    workspace.meritGradient =
        workspace.objectiveGradient +
        workspace.jacobians_data.transpose() * workspace.data_lams_pdal;

    // add proximal penalty terms
    if (rho_ > 0.) {
      workspace.kktRhs.head(ndx) += workspace.prox_grad;
      workspace.kktMatrix.topLeftCorner(ndx, ndx) += workspace.prox_hess;
      workspace.meritGradient += workspace.prox_grad;
    }

    for (std::size_t i = 0; i < num_c; i++) {
      const ConstraintSetBase<Scalar> &cstr_set =
          *problem_->getConstraint(i).m_set;
      cstr_set.computeActiveSet(workspace.cstrValues[i], results.activeSet[i]);

      bool use_vhp = (use_gauss_newton && !cstr_set.disableGaussNewton()) ||
                     !use_gauss_newton;
      if (use_vhp) {
        workspace.kktMatrix.topLeftCorner(ndx, ndx) +=
            workspace.cstrVectorHessianProd[i];
      }
    }

    // Compute dual residual and infeasibility
    workspace.dualResidual = workspace.kktRhs.head(ndx);
    if (rho_ > 0.)
      workspace.dualResidual -= workspace.prox_grad;

    results.dualInfeas = math::infty_norm(workspace.dualResidual);
    for (std::size_t i = 0; i < problem_->getNumConstraints(); i++) {
      const ConstraintSetBase<Scalar> &cstr_set =
          *problem_->getConstraint(i).m_set;
      cstr_set.normalConeProjection(workspace.cstrValues[i],
                                    workspace.cstrValuesProj[i]);
      results.constraint_violations_(long(i)) =
          math::infty_norm(workspace.cstrValuesProj[i]);
    }
    results.primalInfeas = math::infty_norm(results.constraint_violations_);
    Scalar inner_crit = math::infty_norm(workspace.kktRhs);

    // fmt::print(
    //     " | crit={:>5.2e}, d={:>5.3g}, p={:>5.3g} (inner stop {:>5.2e})\n",
    //     inner_crit, results.dualInfeas, results.primalInfeas, inner_tol);

    bool outer_cond = (results.primalInfeas <= target_tol &&
                       results.dualInfeas <= target_tol);
    if ((inner_crit <= inner_tol) || outer_cond) {
      return;
    }

    /* Compute the step */
    delta = DELTA_INIT;
    InertiaFlag is_inertia_correct = BAD;
    while (!(is_inertia_correct == OK) && delta <= DELTA_MAX) {
      if (delta > 0.) {
        workspace.kktMatrix.diagonal().head(ndx).array() += delta;
      }
      workspace.ldlt_.compute(workspace.kktMatrix);
      conditioning = 1. / workspace.ldlt_.rcond();
      workspace.signature.array() =
          workspace.ldlt_.vectorD().array().sign().template cast<int>();
      workspace.kktMatrix.diagonal().head(ndx).array() -= delta;
      is_inertia_correct = checkInertia(workspace.signature);
      // if (verbose >= 2)
      //   fmt::print(" (reg={:>.3g})\n", delta);
      old_delta = delta;

      if (is_inertia_correct == OK) {
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

    workspace.pd_step = -workspace.kktRhs;
    workspace.ldlt_.solveInPlace(workspace.pd_step);

    //// Take the step

    workspace.dmerit_dir =
        workspace.meritGradient.dot(workspace.prim_step) -
        workspace.dual_prox_err_data.dot(workspace.dual_step);

    if (verbose >= 1) {
      // auto resdl = workspace.kktMatrix * workspace.pd_step +
      // workspace.kktRhs; fmt::print(
      //     " | KKT res={:>.2e} | dir={:>4.3g} | cond={:>4.3g} | reg={:>.3g}",
      //     math::infty_norm(resdl), workspace.dmerit_dir, conditioning,
      //     delta);
    }

    Scalar phi0 = results.merit;
    Scalar phi_new = 0.;
    switch (ls_strat) {
    case ARMIJO: {
      phi_new = ArmijoLinesearch<Scalar>(ls_options)
                    .run(phi_eval, results.merit, workspace.dmerit_dir,
                         workspace.alpha_opt);
      break;
    }
    default:
      proxnlp_runtime_error("Unrecognized linesearch alternative.\n");
      break;
    }
    // fmt::print(" | alph_opt={:4.3e}\n", workspace.alpha_opt);

    results.xOpt = workspace.xTrial;
    results.lams_opt_data = workspace.lamsTrial_data;
    results.merit = phi_new;

    invokeCallbacks(workspace, results);

    LogRecord record{results.numIters + 1, workspace.alpha_opt, inner_crit,
                     results.primalInfeas, results.dualInfeas,  delta,
                     workspace.dmerit_dir, results.merit,       phi_new - phi0};

    logger.log(record);

    results.numIters++;
    if (results.numIters >= MAX_ITERS) {
      results.converged = ConvergenceFlag::MAX_ITERS_REACHED;
      break;
    }
  }

  if (results.numIters >= MAX_ITERS)
    results.converged = ConvergenceFlag::MAX_ITERS_REACHED;

  return;
}

template <typename Scalar>
void SolverTpl<Scalar>::setPenalty(const Scalar &new_mu) noexcept {
  mu_ = new_mu;
  mu_inv_ = 1. / mu_;
  merit_fun.setPenalty(mu_);
}

template <typename Scalar>
void SolverTpl<Scalar>::setProxParameter(const Scalar &new_rho) {
  rho_ = new_rho;
  prox_penalty.m_weights.setZero();
  prox_penalty.m_weights.diagonal().setConstant(rho_);
}

template <typename Scalar>
void SolverTpl<Scalar>::updateToleranceFailure() noexcept {
  prim_tol = prim_tol0 * std::pow(mu_, prim_alpha_);
  inner_tol = inner_tol0 * std::pow(mu_, dual_alpha);
}

template <typename Scalar>
void SolverTpl<Scalar>::updateToleranceSuccess() noexcept {
  prim_tol = prim_tol * std::pow(mu_, prim_beta);
  inner_tol = inner_tol * std::pow(mu_, dual_beta);
}

template <typename Scalar>
void SolverTpl<Scalar>::tryStep(const Manifold &manifold, Workspace &workspace,
                                const Results &results, Scalar alpha) {
  manifold.integrate(results.xOpt, alpha * workspace.prim_step,
                     workspace.xTrial);
  workspace.lamsTrial_data =
      results.lams_opt_data + alpha * workspace.dual_step;
}
} // namespace proxnlp
