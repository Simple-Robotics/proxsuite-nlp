/** Copyright (c) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cassert>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ostream.h>

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"
#include "lienlp/meritfuncs/pdal.hpp"
#include "lienlp/workspace.hpp"
#include "lienlp/results.hpp"

#include "lienlp/modelling/costs/squared-distance.hpp"


namespace lienlp {

  template<typename M>
  class Solver
  {
  public:
    using Scalar = typename M::Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;
    using Merit_t = PDALFunction<Scalar>;

    using Workspace = SWorkspace<Scalar>;
    using Results = SResults<Scalar>;

    shared_ptr<Prob_t> problem;
    Merit_t merit_fun;
    /// Manifold on which to optimize.
    M& manifold;
    /// Proximal regularization penalty.
    QuadDistanceCost<M> prox_penalty;

    //// Other settings

    bool verbose = true;
    bool use_gauss_newton = false;    /// Use a Gauss-Newton approximation for the Lagrangian Hessian.

    //// Algo params which evolve

    const Scalar inner_tol0 = 1.;
    const Scalar prim_tol0 = 1.;
    Scalar inner_tol = inner_tol0;
    Scalar prim_tol = prim_tol0;
    Scalar rho;                 /// Primal proximal penalty parameter.
    Scalar mu_eq_init;          /// Initial penalty parameter.
    Scalar mu_eq = mu_eq_init;  /// Penalty parameter.
    Scalar mu_eq_inv = 1. / mu_eq;
    Scalar mu_factor;           /// Penalty update multiplicative factor.
    Scalar rho_factor = mu_factor;

    const Scalar inner_tol_min = 1e-9;  /// Lower safeguard for the subproblem tolerance.
    Scalar mu_lower_ = 1e-9;      /// Lower safeguard for the penalty parameter.

    //// Algo hyperparams

    Scalar target_tol;        /// Target tolerance for the problem.
    const Scalar prim_alpha;  /// BCL failure scaling (primal)
    const Scalar prim_beta;   /// BCL success scaling (primal)
    const Scalar dual_alpha;  /// BCL failure scaling (dual)
    const Scalar dual_beta;   /// BCL success scaling (dual)

    const Scalar alpha_min = 1e-7;
    const Scalar ls_c1 = 1e-4;

    Solver(M& man,
           shared_ptr<Prob_t>& prob,
           const Scalar tol=1e-6,
           const Scalar mu_eq_init=1e-2,
           const Scalar rho=0.,
           const Scalar mu_factor=0.1,
           const Scalar mu_lower_=1e-9,
           const Scalar prim_alpha=0.1,
           const Scalar prim_beta=0.9,
           const Scalar dual_alpha=1.,
           const Scalar dual_beta=1.)
      : manifold(man),
        problem(prob),
        prox_penalty(man),
        merit_fun(prob),
        target_tol(tol),
        mu_eq_init(mu_eq_init),
        rho(rho),
        mu_factor(mu_factor),
        mu_lower_(mu_lower_),
        prim_alpha(prim_alpha),
        prim_beta(prim_beta),
        dual_alpha(dual_alpha),
        dual_beta(dual_beta)
    {
      merit_fun.setPenalty(mu_eq);
    }

    ConvergedFlag
    solve(Workspace& workspace,
          Results& results,
          const VectorXs& x0,
          const VectorOfVectors& lams0)
    {
      // init variables
      results.xOpt = x0;
      results.lamsOpt = lams0;

      updateToleranceFailure();

      results.numIters = 0;

      std::size_t i = 0;
      while (results.numIters < MAX_ITERS)
      {
        results.mu = mu_eq;
        results.rho = rho;
        if (verbose)
        {
          fmt::print(fmt::fg(fmt::color::yellow), "[Iter {:d}] omega={:.3g}, eta={:.3g}, mu={:g} (1/mu={:g})\n",
                     i, inner_tol, prim_tol, mu_eq, mu_eq_inv);
        }
        solveInner(workspace, results);

        // accept new primal iterate
        workspace.xPrev = results.xOpt;
        prox_penalty.updateTarget(workspace.xPrev);

        if (workspace.primalInfeas < prim_tol)
        {
          // accept dual iterate
          fmt::print(fmt::fg(fmt::color::lime_green), "  Accept multipliers\n");
          acceptMultipliers(workspace);
          if ((workspace.primalInfeas < target_tol) && (workspace.dualInfeas < target_tol))
          {
            // terminate algorithm
            results.converged = ConvergedFlag::SUCCESS;
            break;
          }
          updateToleranceSuccess();
        } else {
          fmt::print(fmt::fg(fmt::color::orange_red), "  Reject multipliers\n");
          updatePenalty();
          updateToleranceFailure();
        }
        // safeguard tolerances
        inner_tol = std::max(inner_tol, inner_tol_min);

        i++;
      }

      return results.converged;
    }

    /// Set solver convergence threshold.
    void setTolerance(const Scalar tol) { target_tol = tol; }
    /// Set solver maximum allowed number of iterations.
    void setMaxIters(const std::size_t val) { MAX_ITERS = val; }

    /// Update penalty parameter using the provided factor (with a safeguard Solver::mu_lower_).
    inline void updatePenalty()
    {
      if (mu_eq == mu_lower_)
      {
        setPenalty(mu_eq_init);
      } else {
        setPenalty(std::max(mu_eq * mu_factor, mu_lower_));
      }
    }

    /// Set penalty parameter, its inverse and propagate to merit function.
    void setPenalty(const Scalar new_mu)
    {
      mu_eq = new_mu;
      mu_eq_inv = 1. / mu_eq;
      merit_fun.setPenalty(mu_eq);
    }

  protected:
    std::size_t MAX_ITERS = 100;

    void solveInner(Workspace& workspace, Results& results)
    {
      const auto ndx = manifold.ndx();
      VectorXs& x = results.xOpt; // shorthand
      const std::size_t num_c = problem->getNumConstraints();

      std::size_t k;
      for (k = 0; k < MAX_ITERS; k++)
      {

        //// precompute temp data

        results.value = problem->m_cost(x);
        problem->m_cost.computeGradient(x, workspace.objectiveGradient);
        problem->m_cost.computeHessian(x, workspace.objectiveHessian);

        if (verbose)
        {
          fmt::print("[{}] Iterate {:d}\n", __func__, results.numIters);
          fmt::print(" | objective: {:g}\n", results.value);
        }

        computeResidualsAndMultipliers(x, workspace, results.lamsOpt);
        computeResidualDerivatives(x, workspace);

        //// fill in LHS/RHS
        //// TODO create an Eigen::Map to map submatrices to the active sets of each constraint

        auto idx_prim = Eigen::seq(0, ndx - 1);
        workspace.kktRhs.setZero();
        workspace.kktMatrix.setZero();

        workspace.meritGradient = workspace.objectiveGradient;

        workspace.kktRhs(idx_prim) = workspace.objectiveGradient;
        workspace.kktMatrix(idx_prim, idx_prim) = workspace.objectiveHessian;

        if (rho > 0.)
        {
          workspace.kktRhs(idx_prim).noalias() += rho * prox_penalty.computeGradient(x);
          workspace.kktMatrix(idx_prim, idx_prim).noalias() += rho * prox_penalty.computeHessian(x);
        }

        int nc = 0;   // constraint size
        int cursor = ndx;  // starts after ndx (primal grad size)
        for (std::size_t i = 0; i < num_c; i++)
        {
          Eigen::Ref<MatrixXs> J_ = workspace.cstrJacobians[i];

          workspace.kktRhs(idx_prim).noalias() += J_.transpose() * results.lamsOpt[i];
          if (not use_gauss_newton)
          {
            workspace.kktMatrix(idx_prim, idx_prim) += workspace.cstrVectorHessProd[i];
          }

          workspace.meritGradient.noalias() += J_.transpose() * workspace.lamsPDAL[i];

          // fill in the dual part of the KKT
          auto cstr = problem->getCstr(i);
          nc = cstr->nr();
          cstr->computeActiveSet(workspace.primalResiduals[i], results.activeSet[i]);
          auto block_slice = Eigen::seq(cursor, cursor + nc - 1);
          workspace.kktRhs(block_slice) = workspace.auxProxDualErr[i];
          // jacobian block and transpose
          workspace.kktMatrix(block_slice, idx_prim) = J_;
          workspace.kktMatrix(idx_prim, block_slice) = J_.transpose();
          // reg block
          workspace.kktMatrix(block_slice, block_slice).setIdentity();
          workspace.kktMatrix(block_slice, block_slice).array() *= -mu_eq;

          cursor += nc;
        }

        // now check if we can stop
        workspace.dualResidual = workspace.kktRhs(idx_prim);
        if (rho > 0.)
        {
          workspace.dualResidual -= rho * prox_penalty.computeGradient(x);
        }
        workspace.dualInfeas = infNorm(workspace.dualResidual);
        Scalar inner_crit = infNorm(workspace.kktRhs);

        if (verbose)
        {
          // fmt::print(" | KKT RHS: {} << RHS\n",  workspace.kktRhs.transpose());
          // fmt::print(" | KKT LHS:\n{} << LHS\n", workspace.kktMatrix);
          fmt::print(" | inner stop {:.2g} / dualInfeas: {:.2g} / primInfeas = {:.2g}\n",
                     inner_crit, workspace.dualInfeas, workspace.primalInfeas);
        }

        if ((inner_crit <= inner_tol) && (k > 0))
        {
          return;
        }

        // factorization
        workspace.ldlt_.compute(workspace.kktMatrix);
        workspace.pdStep = -workspace.kktRhs;
        workspace.ldlt_.solveInPlace(workspace.pdStep);
        const Scalar conditioning_ = 1. / workspace.ldlt_.rcond();
        workspace.signature.array() = workspace.ldlt_.vectorD().array().sign().template cast<int>();

        if (verbose)
        {
          fmt::print(" | conditioning:  {}\n", conditioning_);
          fmt::print(" | KKT signature: {}\n", workspace.signature.transpose());
        }

        assert(workspace.ldlt_.info() == Eigen::ComputationInfo::Success);

        //// Take the step

        Scalar merit0 = merit_fun(results.xOpt, results.lamsOpt, workspace.lamsPrev);
        if (rho > 0.)
        {
          merit0 += rho * prox_penalty(x);
          workspace.meritGradient.noalias() += rho * prox_penalty.computeGradient(x);
        }
        Scalar dir_x = workspace.meritGradient.dot(workspace.pdStep(idx_prim));
        Scalar dir_dual = 0;
        cursor = ndx;
        for (std::size_t i = 0; i < num_c; i++)
        {
          nc = problem->getCstr(i)->nr();
          auto block_slice = Eigen::seq(cursor, cursor + nc - 1);

          dir_dual += (-workspace.auxProxDualErr[i]).dot(workspace.pdStep(block_slice));
          cursor += nc;
        }

        Scalar dir_deriv = dir_x + dir_dual;

        doLinesearch(workspace, results, merit0, dir_deriv);
        results.xOpt = workspace.xTrial;
        results.lamsOpt = workspace.lamsTrial;

        results.numIters++;
        if (results.numIters >= MAX_ITERS)
        {
          results.converged = ConvergedFlag::TOO_MANY_ITERS;
          break;
        }
      }

      if (results.numIters >= MAX_ITERS)
        results.converged = ConvergedFlag::TOO_MANY_ITERS;

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
      prim_tol = prim_tol0 * std::pow(mu_eq, prim_alpha);
      inner_tol = inner_tol0 * std::pow(mu_eq, dual_alpha);
    }

    /**
     * Update primal-dual subproblem tolerances upon
     * successful outer-loop iterate (good primal feasibility)
     */
    void updateToleranceSuccess()
    {
      prim_tol = prim_tol * std::pow(mu_eq, prim_beta);
      inner_tol = inner_tol * std::pow(mu_eq, dual_beta);
    }

    /// @brief  Accept Lagrange multiplier estimates.
    void acceptMultipliers(Workspace& workspace) const
    {
      const auto nc = problem->getNumConstraints();
      for (std::size_t i = 0; i < nc; i++)
      {
        // copy the (cached) estimates from the algo
        workspace.lamsPrev[i] = workspace.lamsPDAL[i];
      }
    }

    /** 
     * Evaluate the primal residual vectors, and compute
     * the first-order and primal-dual Lagrange multiplier estimates.
     */
    void computeResidualsAndMultipliers(
      const ConstVectorRef& x,
      Workspace& workspace,
      VectorOfVectors& lams) const
    {
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        auto cstr = problem->getCstr(i);
        workspace.primalResiduals[i] = cstr->m_func(x);

        // multiplier
        workspace.lamsPlusPre[i] = workspace.lamsPrev[i] + mu_eq_inv * workspace.primalResiduals[i];
        workspace.lamsPlus[i] = cstr->dualProjection(workspace.lamsPlusPre[i]);
        workspace.auxProxDualErr[i] = mu_eq * (workspace.lamsPlus[i] - lams[i]);
        workspace.lamsPDAL[i] = 2 * workspace.lamsPlus[i] - lams[i];
      } 

      // update primal infeas measure
      workspace.primalInfeas = 0.;
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        auto cstr = problem->getCstr(i);
        workspace.primalInfeas = std::max(
          workspace.primalInfeas,
          infNorm(cstr->dualProjection(workspace.primalResiduals[i])));
      }
    }

    /**
     * Evaluate the derivatives (Jacobian, and vector-Hessian products) of the
     * constraint residuals.
     */
    void computeResidualDerivatives(
      const ConstVectorRef& x,
      Workspace& workspace) const
    {
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        auto cstr = problem->getCstr(i);

        Eigen::Ref<MatrixXs> J_ = workspace.cstrJacobians[i];
        cstr->m_func.computeJacobian(x, J_);
        MatrixXs jacProj = cstr->JdualProjection(workspace.lamsPlusPre[i]);
        workspace.cstrJacobians[i] = jacProj * J_;
        cstr->m_func.vhp(x, workspace.lamsPDAL[i], workspace.cstrVectorHessProd[i]);
      }
    } 

    /**
     * Perform the inexact backtracking line-search procedure.
     * 
     * @param workspace Workspace.
     * @param results   Result struct.
     * @param merit0    Value of the merit function at the previous point.
     * @param d1        Directional derivative of the merit function in the search direction.
     */
    Scalar doLinesearch(Workspace& workspace, Results& results, Scalar merit0, Scalar d1) const
    {
      Scalar alpha_try = 1.;

      const Scalar ls_beta = 0.5;
      fmt::print(fmt::fg(fmt::color::yellow), "  [{}] current M = {:.5g} | d1 = {:.3g}\n", __func__, merit0, d1);

      Scalar merit_trial = 0., dM = 0.;
      while (alpha_try >= alpha_min)
      {
        tryStep(workspace, results, alpha_try);
        merit_trial = merit_fun(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev);
        merit_trial += rho * prox_penalty(workspace.xTrial);
        dM = merit_trial - merit0;
        fmt::print(fmt::fg(fmt::color::yellow), "  [{}] alpha {:.2e}, M = {:.5g}, dM = {:.5g}\n", __func__, alpha_try, merit_trial, dM);

        bool armijo_cond = dM <= ls_c1 * alpha_try * d1;
        if (armijo_cond)
        {
          break;
        }
        alpha_try *= ls_beta;
      }

      if (alpha_try < alpha_min)
      {
        alpha_try = alpha_min;
        tryStep(workspace, results, alpha_try);
      }

      return alpha_try;
    }

    /**
     * Take a trial step.
     * 
     * @param workspace Workspace
     * @param results   Contains the previous primal-dual point
     * @param alpha     Step size
     */
    void tryStep(Workspace &workspace, Results& results, Scalar alpha) const
    {
      const int ndx = manifold.ndx();
      const auto idx_prim = Eigen::seq(0, ndx - 1);
      manifold.integrate(results.xOpt, alpha * workspace.pdStep(idx_prim), workspace.xTrial);

      int cursor = ndx;
      int nc = 0;
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        nc = problem->getCstr(i)->nr();

        auto block_slice = Eigen::seq(cursor, cursor + nc - 1);
        workspace.lamsTrial[i].noalias() = results.lamsOpt[i] + alpha * workspace.pdStep(block_slice);

        cursor += nc;
      }
    }

  };

} // namespace lienlp

