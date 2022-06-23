/**
 * @file solver-base.hpp
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/problem-base.hpp"
#include "proxnlp/pdal.hpp"
#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"
#include "proxnlp/helpers-base.hpp"

#include "proxnlp/modelling/costs/squared-distance.hpp"

#include "proxnlp/linesearch-base.hpp"

#include <cassert>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ostream.h>

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

  template<typename _Scalar>
  class SolverTpl
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Problem = ProblemTpl<Scalar>;

    using Workspace = WorkspaceTpl<Scalar>;
    using Results = ResultsTpl<Scalar>;
  
    using Manifold = ManifoldAbstractTpl<Scalar>;

    /// Manifold on which to optimize.
    const Manifold& manifold;
    shared_ptr<Problem> problem;
    /// Merit function.
    PDALFunction<Scalar> merit_fun;
    /// Proximal regularization penalty.
    QuadraticDistanceCost<Scalar> prox_penalty;

    //// Other settings

    VerboseLevel verbose = QUIET;           // Level of verbosity of the solver.
    bool use_gauss_newton = false;          // Use a Gauss-Newton approximation for the Lagrangian Hessian.
    bool record_linesearch_process = false;

    LinesearchStrategy ls_strat = ARMIJO;

    //// Algo params which evolve

    const Scalar inner_tol0 = 1.;
    const Scalar prim_tol0 = 1.;
    Scalar inner_tol = inner_tol0;
    Scalar prim_tol = prim_tol0;
    Scalar rho_init;                        // Initial primal proximal penalty parameter.
    Scalar rho_ = rho_init;                  // Primal proximal penalty parameter.
    Scalar mu_eq_init;                      // Initial penalty parameter.
    Scalar mu_eq_ = mu_eq_init;             // Penalty parameter.
    Scalar mu_eq_inv_ = 1. / mu_eq_;        // Inverse penalty parameter.
    Scalar mu_factor_;                      // Penalty update multiplicative factor.
    Scalar rho_factor_ = mu_factor_;        // Primal penalty update factor.

    const Scalar inner_tol_min = 1e-9;      // Lower safeguard for the subproblem tolerance.
    Scalar mu_lower_ = 1e-9;                // Lower safeguard for the penalty parameter.

    //// Algo hyperparams

    Scalar target_tol;                      // Target tolerance for the problem.
    const Scalar prim_alpha;                // BCL failure scaling (primal)
    const Scalar prim_beta;                 // BCL success scaling (primal)
    const Scalar dual_alpha;                // BCL failure scaling (dual)
    const Scalar dual_beta;                 // BCL success scaling (dual)

    const Scalar alpha_min;                 // Linesearch minimum step size.
    const Scalar armijo_c1;                 // Armijo rule c1 parameter.
    Scalar ls_beta;                         // Linesearch step size decrease factor.
    
    const Scalar del_inc_k = 8.;
    const Scalar del_inc_big = 100.;
    const Scalar del_dec_k = 1./3.;

    const Scalar DELTA_MIN = 1e-14;         // Minimum nonzero regularization strength.
    const Scalar DELTA_MAX = 1e6;           // Maximum regularization strength.
    const Scalar DELTA_NONZERO_INIT = 1e-4;
    const Scalar DELTA_INIT = 0.;

    /// Callbacks
    using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>; 
    std::vector<CallbackPtr> callbacks_;

    SolverTpl(const Manifold& manifold,
              shared_ptr<Problem>& prob,
              const Scalar tol=1e-6,
              const Scalar mu_eq_init=1e-2,
              const Scalar rho_init=0.,
              const VerboseLevel verbose=QUIET,
              const Scalar mu_factor=0.1,
              const Scalar mu_lower=1e-9,
              const Scalar prim_alpha=0.1,
              const Scalar prim_beta=0.9,
              const Scalar dual_alpha=1.,
              const Scalar dual_beta=1.,
              const Scalar alpha_min=1e-7,
              const Scalar armijo_c1=1e-4,
              const Scalar ls_beta=0.5
              )
      : manifold(manifold)
      , problem(prob)
      , merit_fun(problem, mu_eq_init)
      , prox_penalty(manifold, manifold.neutral(), rho_init * MatrixXs::Identity(manifold.ndx(), manifold.ndx()))
      , verbose(verbose)
      , rho_init(rho_init)
      , mu_eq_init(mu_eq_init)
      , mu_factor_(mu_factor)
      , mu_lower_(mu_lower)
      , target_tol(tol)
      , prim_alpha(prim_alpha)
      , prim_beta(prim_beta)
      , dual_alpha(dual_alpha)
      , dual_beta(dual_beta)
      , alpha_min(alpha_min)
      , armijo_c1(armijo_c1)
      , ls_beta(ls_beta)
    {}

    enum InertiaFlag
    {
      OK = 0,
      BAD = 1,
      ZEROS = 2
    };

    /// @copybrief solve().
    ///
    /// @param lams0 Initial Lagrange multipliers given separately for each constraint.
    ConvergenceFlag solve(Workspace& workspace,
                          Results& results,
                          const ConstVectorRef& x0,
                          const std::vector<VectorRef>& lams0);

    /// @brief Solve the problem.
    ConvergenceFlag solve(Workspace& workspace,
                          Results& results,
                          const ConstVectorRef& x0,
                          const ConstVectorRef& lams0)
    {
      // init variables
      results.xOpt = x0;
      workspace.xPrev = x0;
      results.lamsOpt_data = lams0;
      workspace.lamsPrev_data = lams0;

      updateToleranceFailure();

      results.numIters = 0;

      std::size_t i = 0;
      while (results.numIters < MAX_ITERS)
      {
        results.mu = mu_eq_;
        results.rho = rho_;
        fmt::print(fmt::fg(fmt::color::yellow),
                   "[Outer iter {:>2d}] omega={:.3g}, eta={:.3g}, mu={:g}\n",
                   i, inner_tol, prim_tol, mu_eq_);
        solveInner(workspace, results);

        // accept new primal iterate
        workspace.xPrev = results.xOpt;
        prox_penalty.updateTarget(workspace.xPrev);

        if (results.primalInfeas < prim_tol)
        {
          fmt::print(fmt::fg(fmt::color::lime_green), "> Accept multipliers\n");
          acceptMultipliers(workspace);
          if ((results.primalInfeas < target_tol) && (results.dualInfeas < target_tol))
          {
            // terminate algorithm
            results.converged = ConvergenceFlag::SUCCESS;
            break;
          }
          updateToleranceSuccess();
        } else {
          fmt::print(fmt::fg(fmt::color::orange_red), "> Reject multipliers\n");
          updatePenalty();
          updateToleranceFailure();
        }
        // safeguard tolerances
        inner_tol = std::max(inner_tol, inner_tol_min);

        i++;
      }

      if (results.converged == SUCCESS)
        fmt::print("Solver successfully converged\n"
                   "  numIters : {:d}\n"
                   "  residuals: p={:.3g}, d={:.3g}\n",
                   results.numIters, results.primalInfeas, results.dualInfeas);

      switch(results.converged)
      {
      case MAX_ITERS_REACHED: fmt::print(fmt::fg(fmt::color::orange_red), "Max number of iterations reached.\n");
                              break;
      default: break;
      }
      fmt::print("\n");

      invokeCallbacks(workspace, results);

      return results.converged;
    }

    /// Set solver convergence threshold.
    void setTolerance(const Scalar tol) { target_tol = tol; }
    /// Set solver maximum allowed number of iterations.
    void setMaxIters(const std::size_t val) { MAX_ITERS = val; }
    /// Get solver maximum number of iterations.
    std::size_t getMaxIters() const { return MAX_ITERS; }

    /// Update penalty parameter using the provided factor (with a safeguard SolverTpl::mu_lower).
    inline void updatePenalty()
    {
      if (mu_eq_ == mu_lower_)
      {
        setPenalty(mu_eq_init);
      } else {
        setPenalty(std::max(mu_eq_ * mu_factor_, mu_lower_));
      }
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        const typename Problem::ConstraintPtr& cstr = problem->getConstraint(i);
        cstr->m_set->updateProxParameters(mu_eq_);
      }
    }

    /// @brief Set penalty parameter, its inverse and update the merit function.
    /// @param new_mu The new penalty parameter.
    void setPenalty(const Scalar& new_mu)
    {
      mu_eq_ = new_mu;
      mu_eq_inv_ = 1. / mu_eq_;
      merit_fun.setPenalty(mu_eq_);
    }

    /// Set proximal penalty parameter.
    void setProxParameter(const Scalar& new_rho)
    {
      rho_ = new_rho;
      prox_penalty.m_weights.setZero();
      prox_penalty.m_weights.diagonal().setConstant(rho_);
    }

    /// @brief    Add a callback to the solver instance.
    inline void registerCallback(const CallbackPtr& cb)
    {
      callbacks_.push_back(cb);
    }

    /// @brief    Remove all callbacks from the instance.
    inline void clearCallbacks()
    {
      callbacks_.clear();
    }

    std::size_t MAX_ITERS = 100;

    void solveInner(Workspace& workspace, Results& results)
    {
      const int ndx = manifold.ndx();
      const long ntot = workspace.kktRhs.size();
      const long ndual = ntot - ndx;
      const std::size_t num_c = problem->getNumConstraints();

      results.lamsOpt_data = workspace.lamsPrev_data;

      Scalar delta_last = 0.;
      Scalar delta = delta_last;
      Scalar old_delta = 0.;
      Scalar conditioning_ = 0;

      VectorXs resdl(workspace.ndx + workspace.numdual);
      resdl.setZero();

      merit_fun.setPenalty(mu_eq_);

      // lambda for evaluating the merit function
      auto phiEval = [&](Scalar alpha) {
        return compute_merit_for_step(*this, workspace, results, alpha);
      };

      std::size_t k;
      for (k = 0; k < MAX_ITERS; k++)
      {

        //// precompute temp data

        results.value = problem->m_cost.call(results.xOpt);

        computeResidualsAndMultipliers(results.xOpt, results.lamsOpt_data, workspace);
        computeResidualDerivatives(results.xOpt, workspace, true);

        results.merit = merit_fun(results.xOpt, results.lamsOpt, workspace.lamsPrev);
        if (rho_ > 0.)
          results.merit += prox_penalty.call(results.xOpt);

        if (verbose >= 0)
        {
          fmt::print("[iter {:>3d}] objective: {:g} merit: {:g}\n", results.numIters, results.value, results.merit);
        }

        //// fill in LHS/RHS
        //// TODO create an Eigen::Map to map submatrices to the active sets of each constraint

        workspace.kktRhs.setZero();
        workspace.kktMatrix.setZero();

        workspace.kktMatrix.topLeftCorner(ndx, ndx)      = workspace.objectiveHessian;
        workspace.kktMatrix.topRightCorner(ndx, ndual)   = workspace.jacobians_data.transpose();
        workspace.kktMatrix.bottomLeftCorner(ndual, ndx) = workspace.jacobians_data;
        workspace.kktMatrix.bottomRightCorner(ndual, ndual).diagonal().setConstant(-mu_eq_);

        // add jacobian-vector products to gradients
        workspace.kktRhs.head(ndx)   = workspace.objectiveGradient + workspace.jacobians_data.transpose() * results.lamsOpt_data;
        workspace.kktRhs.tail(ndual) = workspace.dual_prox_err_data;
        workspace.meritGradient = workspace.objectiveGradient + workspace.jacobians_data.transpose() * workspace.lamsPDAL_data;

        // add proximal penalty terms
        if (rho_ > 0.)
        {
          workspace.kktRhs.head(ndx).noalias() += workspace.prox_grad;
          workspace.kktMatrix.topLeftCorner(ndx, ndx).noalias() += workspace.prox_hess;
          workspace.meritGradient.noalias() += workspace.prox_grad;
        }

        for (std::size_t i = 0; i < num_c; i++)
        {
          const typename Problem::ConstraintPtr& cstr = problem->getConstraint(i);
          cstr->m_set->computeActiveSet(workspace.cstrValues[i], results.activeSet[i]);

          bool use_vhp = (use_gauss_newton && !cstr->m_set->disableGaussNewton()) || !use_gauss_newton; 
          if (use_vhp)
          {
            workspace.kktMatrix.topLeftCorner(ndx, ndx).noalias() += workspace.cstrVectorHessianProd[i];
          }
        }

        // Compute dual residual and infeasibility
        workspace.dualResidual = workspace.kktRhs.head(ndx);
        if (rho_ > 0.)
          workspace.dualResidual.noalias() -= workspace.prox_grad;

        results.dualInfeas = math::infty_norm(workspace.dualResidual);
        for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
        {
          const typename Problem::ConstraintPtr& cstr = problem->getConstraint(i);
          auto set = cstr->m_set;
          results.constraint_violations_((long)i) = math::infty_norm(
            set->normalConeProjection(workspace.cstrValues[i]));
        }
        results.primalInfeas = math::infty_norm(results.constraint_violations_);
        // Compute inner stopping criterion
        Scalar inner_crit = math::infty_norm(workspace.kktRhs);

        fmt::print(" | crit={:>5.2e}, d={:>5.3g}, p={:>5.3g} (inner stop {:>5.2e})\n",
                   inner_crit, results.dualInfeas, results.primalInfeas, inner_tol);

        bool outer_cond = (results.primalInfeas <= target_tol && results.dualInfeas <= target_tol);
        if ((inner_crit <= inner_tol) || outer_cond)
        {
          return;
        }

        /* Compute the step */

        // factorization
        // regularization strength : always try 0
        delta = DELTA_INIT;
        InertiaFlag is_inertia_correct = BAD;
        while (!(is_inertia_correct == OK) && delta <= DELTA_MAX)
        {
          if (delta > 0.)
          {
            workspace.kktMatrix.diagonal().head(ndx).array() += delta;
          }
          workspace.ldlt_.compute(workspace.kktMatrix);
          conditioning_ = 1. / workspace.ldlt_.rcond();
          workspace.signature.array() = workspace.ldlt_.vectorD().array().sign().template cast<int>();
          workspace.kktMatrix.diagonal().head(ndx).array() -= delta;
          is_inertia_correct = checkInertia(workspace.signature);
          if (verbose >= 2)
            fmt::print(" (reg={:>.3g})\n", delta);
          old_delta = delta;

          if (is_inertia_correct == OK)
          {
            delta_last = delta;
            break;
          }
          else if (delta == 0.) {

            // check if previous was zero:
            // either use some nonzero value
            // or try some fraction of previous
            if (delta_last == 0.)
            {
              delta = DELTA_NONZERO_INIT; // try a set nonzero value
            } else {
              delta = std::max(DELTA_MIN, del_dec_k * delta_last);
            }

          } else  {

            // check previous; decide increase factor
            if (delta_last == 0.)
            {
              delta *= del_inc_big;
            } else {
              delta *= del_inc_k;
            }
          }
        }

        workspace.pd_step = -workspace.kktRhs;
        workspace.ldlt_.solveInPlace(workspace.pd_step);
        resdl = workspace.kktMatrix * workspace.pd_step + workspace.kktRhs;

        assert(workspace.ldlt_.info() == Eigen::ComputationInfo::Success);

        //// Take the step

        workspace.dmerit_dir = workspace.meritGradient.dot(workspace.prim_step) - workspace.dual_prox_err_data.dot(workspace.dual_step);

        if (verbose >= 1)
        {
          fmt::print(" | KKT res={:>.2e} | dir={:>4.3g} | cond={:>4.3g} | reg={:>.3g}",
                     math::infty_norm(resdl), workspace.dmerit_dir, conditioning_, delta);
        }

        Scalar& alpha_opt = workspace.alpha_opt;

        switch (ls_strat)
        {
        case ARMIJO: {
          ArmijoLinesearch<Scalar>::run(phiEval, results.merit, workspace.dmerit_dir, verbose, ls_beta, armijo_c1, alpha_min, alpha_opt);
          break;
        }
        case CUBIC_INTERP: {
          CubicInterpLinesearch<Scalar>::run(phiEval, results.merit, workspace.dmerit_dir, verbose, armijo_c1, alpha_min, alpha_opt);
          break;
        }
        default: break;
        }
        fmt::print(" | alph_opt={:4.3e}\n", alpha_opt);

        results.xOpt = workspace.xTrial;
        results.lamsOpt_data = workspace.lamsTrial_data;

        invokeCallbacks(workspace, results);

        results.numIters++;
        if (results.numIters >= MAX_ITERS)
        {
          results.converged = ConvergenceFlag::MAX_ITERS_REACHED;
          break;
        }
      }

      if (results.numIters >= MAX_ITERS)
        results.converged = ConvergenceFlag::MAX_ITERS_REACHED;

      return;
    }

    /**
     * Update primal-dual subproblem tolerances upon
     * failure (insufficient primal feasibility)
     * 
     * Also call this upon initialization of the solver.
     */
    void updateToleranceFailure()
    {
      prim_tol = prim_tol0 * std::pow(mu_eq_, prim_alpha);
      inner_tol = inner_tol0 * std::pow(mu_eq_, dual_alpha);
    }

    /**
     * Update primal-dual subproblem tolerances upon
     * successful outer-loop iterate (good primal feasibility)
     */
    void updateToleranceSuccess()
    {
      prim_tol = prim_tol * std::pow(mu_eq_, prim_beta);
      inner_tol = inner_tol * std::pow(mu_eq_, dual_beta);
    }

    /// @brief  Accept Lagrange multiplier estimates.
    void acceptMultipliers(Workspace& workspace) const
    {
      workspace.lamsPrev_data = workspace.lamsPDAL_data;
    }

    /** 
     * Evaluate the problem data, as well as the proximal/projection operators,
     * and the first-order & primal-dual multiplier estimates.
     *
     * @param workspace Problem workspace.
     */
    void computeResidualsAndMultipliers(const ConstVectorRef& x, const ConstVectorRef& lams_data, Workspace& workspace) const;

    /**
     * Evaluate the derivatives (cost gradient, Hessian, constraint Jacobians, vector-Hessian products)
     * of the problem data.
     * 
     * @param workspace Problem workspace.
     * @param second_order Whether to compute the second-order information; set to false for e.g. linesearch.
     */
    void computeResidualDerivatives(const ConstVectorRef& x, Workspace& workspace, bool second_order) const;

    /**
     * Take a trial step.
     * 
     * @param workspace Workspace
     * @param results   Contains the previous primal-dual point
     * @param alpha     Step size
     */
    static void tryStep(const Manifold& manifold, Workspace& workspace, const Results& results, Scalar alpha)
    {
      manifold.integrate(results.xOpt, alpha * workspace.prim_step, workspace.xTrial);
      workspace.lamsTrial_data = results.lamsOpt_data + alpha * workspace.dual_step;
    }

    void invokeCallbacks(Workspace& workspace, Results& results)
    {
      for (auto cb : callbacks_)
      {
        cb->call(workspace, results);
      }
    }

  protected:
    /// Check the matrix has the desired inertia.
    /// @param    signature The computed inertia as a vector of ints valued -1, 0, or 1.
    /// @param    delta     Scale factor for the identity matrix to add
    InertiaFlag checkInertia(const Eigen::VectorXi& signature) const;

  };

} // namespace proxnlp

#include "proxnlp/solver-base.hxx"
