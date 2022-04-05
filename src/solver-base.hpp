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
#include "lienlp/helpers-base.hpp"

#include "lienlp/modelling/costs/squared-distance.hpp"


namespace lienlp
{

  template<typename _Scalar>
  class SolverTpl
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;
    using Merit_t = PDALFunction<Scalar>;

    using Workspace = SWorkspace<Scalar>;
    using Results = SResults<Scalar>;
  
    using M = ManifoldAbstractTpl<Scalar>;

    /// Manifold on which to optimize.
    const M& manifold;
    shared_ptr<Problem> problem;
    /// Merit function.
    Merit_t merit_fun;
    /// Proximal regularization penalty.
    QuadraticDistanceCost<Scalar> prox_penalty;

    //// Other settings

    bool verbose = true;
    bool use_gauss_newton = false;    /// Use a Gauss-Newton approximation for the Lagrangian Hessian.

    //// Algo params which evolve

    const Scalar inner_tol0 = 1.;
    const Scalar prim_tol0 = 1.;
    Scalar inner_tol = inner_tol0;
    Scalar prim_tol = prim_tol0;
    Scalar rho_init;            /// Initial primal proximal penalty parameter.
    Scalar rho = rho_init;      /// Primal proximal penalty parameter.
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

    const Scalar alpha_min;
    const Scalar armijo_c1;

    /// Callbacks
    using CallbackPtr = shared_ptr<helpers::callback<Scalar>>; 
    std::vector<CallbackPtr> callbacks_;

    SolverTpl(const M& man,
           shared_ptr<Problem>& prob,
           const Scalar tol=1e-6,
           const Scalar mu_eq_init=1e-2,
           const Scalar rho_init=0.,
           const bool verbose=true,
           const Scalar mu_factor=0.1,
           const Scalar mu_lower_=1e-9,
           const Scalar prim_alpha=0.1,
           const Scalar prim_beta=0.9,
           const Scalar dual_alpha=1.,
           const Scalar dual_beta=1.,
           const Scalar alpha_min=1e-7,
           const Scalar armijo_c1=1e-4
           )
      : manifold(man),
        problem(prob),
        merit_fun(problem),
        prox_penalty(manifold, manifold.neutral(),
                     rho_init * MatrixXs::Identity(manifold.ndx(), manifold.ndx())),
        verbose(verbose),
        rho_init(rho_init),
        mu_eq_init(mu_eq_init),
        mu_factor(mu_factor),
        mu_lower_(mu_lower_),
        target_tol(tol),
        prim_alpha(prim_alpha),
        prim_beta(prim_beta),
        dual_alpha(dual_alpha),
        dual_beta(dual_beta),
        alpha_min(alpha_min),
        armijo_c1(armijo_c1)
    {
      merit_fun.setPenalty(mu_eq);
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

    ConvergenceFlag solve(Workspace& workspace,
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
        fmt::print(fmt::fg(fmt::color::yellow),
                   "[Outer iter {:>2d}] omega={:.3g}, eta={:.3g}, mu={:g}\n",
                   i, inner_tol, prim_tol, mu_eq);
        solveInner(workspace, results);

        // accept new primal iterate
        workspace.xPrev = results.xOpt;
        prox_penalty.updateTarget(workspace.xPrev);

        if (workspace.primalInfeas < prim_tol)
        {
          // accept dual iterate
          fmt::print(fmt::fg(fmt::color::lime_green), "> Accept multipliers\n");
          acceptMultipliers(workspace);
          if ((workspace.primalInfeas < target_tol) && (workspace.dualInfeas < target_tol))
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
                   "  number of iterations: {:d}\n"
                   "  residuals: p={:.3g}, d={:.3g}\n",
                   results.numIters, workspace.primalInfeas, workspace.dualInfeas);
      return results.converged;
    }

    /// Set solver convergence threshold.
    void setTolerance(const Scalar tol) { target_tol = tol; }
    /// Set solver maximum allowed number of iterations.
    void setMaxIters(const std::size_t val) { MAX_ITERS = val; }

    /// Update penalty parameter using the provided factor (with a safeguard SolverTpl::mu_lower_).
    inline void updatePenalty()
    {
      if (mu_eq == mu_lower_)
      {
        setPenalty(mu_eq_init);
      } else {
        setPenalty(std::max(mu_eq * mu_factor, mu_lower_));
      }
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        auto cstr = problem->getConstraint(i);
        cstr->updateProxParameters(mu_eq);
      }
    }

    /// Set penalty parameter, its inverse and propagate to merit function.
    void setPenalty(const Scalar new_mu)
    {
      mu_eq = new_mu;
      mu_eq_inv = 1. / mu_eq;
      merit_fun.setPenalty(mu_eq);
    }

    /// Set proximal penalty parameter.
    void setProxParam(const Scalar new_rho)
    {
      rho = new_rho;
      prox_penalty.m_weights.setIdentity();
      prox_penalty.m_weights *= rho;
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

        results.value = problem->m_cost.call(x);
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

        workspace.kktRhs.setZero();
        workspace.kktMatrix.setZero();

        workspace.meritGradient = workspace.objectiveGradient;

        workspace.kktRhs.head(ndx) = workspace.objectiveGradient;
        workspace.kktMatrix.topLeftCorner(ndx, ndx) = workspace.objectiveHessian;

        if (rho > 0.)
        {
          workspace.kktRhs.head(ndx).noalias() += rho * prox_penalty.computeGradient(x);
          workspace.kktMatrix.topLeftCorner(ndx, ndx).noalias() += rho * prox_penalty.computeHessian(x);
        }

        int nc = 0;   // constraint size
        int cursor = ndx;  // starts after ndx (primal grad size)
        for (std::size_t i = 0; i < num_c; i++)
        {
          Eigen::Ref<MatrixXs> J_ = workspace.cstrJacobians[i];

          workspace.kktRhs.head(ndx).noalias() += J_.transpose() * results.lamsOpt[i];
          if (not use_gauss_newton)
          {
            workspace.kktMatrix.topLeftCorner(ndx, ndx) += workspace.cstrVectorHessProd[i];
          }

          workspace.meritGradient.noalias() += J_.transpose() * workspace.lamsPDAL[i];

          // fill in the dual part of the KKT
          auto cstr = problem->getConstraint(i);
          nc = cstr->nr();
          cstr->computeActiveSet(workspace.primalResiduals[i], results.activeSet[i]);
          workspace.kktRhs.segment(cursor, nc) = workspace.subproblemDualErr[i];
          // jacobian block and transpose
          workspace.kktMatrix.block(cursor, 0, nc, ndx) = J_;
          workspace.kktMatrix.block(0, cursor, ndx, nc) = J_.transpose();
          // reg block
          workspace.kktMatrix.block(cursor, cursor, nc, nc).setIdentity();
          workspace.kktMatrix.block(cursor, cursor, nc, nc).array() *= -mu_eq;

          cursor += nc;
        }

        // now check if we can stop
        workspace.dualResidual = workspace.kktRhs.head(ndx);
        if (rho > 0.)
        {
          workspace.dualResidual -= rho * prox_penalty.computeGradient(x);
        }
        workspace.dualInfeas = infNorm(workspace.dualResidual);
        Scalar inner_crit = infNorm(workspace.kktRhs);

        fmt::print("[iter {:>3d}] inner stop {:.4g}, d={:.3g}, p={:.3g}\n",
                  results.numIters, inner_tol, workspace.dualInfeas, workspace.primalInfeas);

        if (inner_crit <= inner_tol)
        {
          return;
        }

        for (auto cb : callbacks_)
        {
          cb->call(workspace, results);
        }

        // factorization
        workspace.ldlt_.compute(workspace.kktMatrix);
        workspace.pdStep = -workspace.kktRhs;
        workspace.ldlt_.solveInPlace(workspace.pdStep);
        const Scalar conditioning_ = 1. / workspace.ldlt_.rcond();
        workspace.signature.array() = workspace.ldlt_.vectorD().array().sign().template cast<int>();

        if (verbose)
        {
          fmt::print(" | conditioning:  {:.3g}\n", conditioning_);
          fmt::print(" | KKT signature: {}\n", workspace.signature.transpose());
        }

        assert(workspace.ldlt_.info() == Eigen::ComputationInfo::Success);

        //// Take the step

        Scalar merit0 = merit_fun(results.xOpt, results.lamsOpt, workspace.lamsPrev);
        if (rho > 0.)
        {
          merit0 += rho * prox_penalty.call(x);
          workspace.meritGradient.noalias() += rho * prox_penalty.computeGradient(x);
        }
        Scalar dir_x = workspace.meritGradient.dot(workspace.pdStep.head(ndx));
        Scalar dir_dual = 0;
        cursor = ndx;
        for (std::size_t i = 0; i < num_c; i++)
        {
          nc = problem->getConstraint(i)->nr();

          dir_dual += (-workspace.subproblemDualErr[i]).dot(workspace.pdStep.segment(cursor, nc));
          cursor += nc;
        }

        Scalar dir_deriv = dir_x + dir_dual;

        doLinesearch(workspace, results, merit0, dir_deriv);
        results.xOpt = workspace.xTrial;
        results.lamsOpt = workspace.lamsTrial;

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
        auto cstr = problem->getConstraint(i);
        workspace.primalResiduals[i] = cstr->m_func(x);

        // multiplier
        workspace.lamsPlusPre[i] = workspace.lamsPrev[i] + mu_eq_inv * workspace.primalResiduals[i];
        workspace.lamsPlus[i] = cstr->normalConeProjection(workspace.lamsPlusPre[i]);
        workspace.subproblemDualErr[i] = mu_eq * (workspace.lamsPlus[i] - lams[i]);
        workspace.lamsPDAL[i] = 2 * workspace.lamsPlus[i] - lams[i];
      } 

      // update primal infeas measure
      workspace.primalInfeas = 0.;
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        auto cstr = problem->getConstraint(i);
        workspace.primalInfeas = std::max(
          workspace.primalInfeas,
          infNorm(cstr->normalConeProjection(workspace.primalResiduals[i])));
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
        auto cstr = problem->getConstraint(i);

        MatrixXs& J_ = workspace.cstrJacobians[i];
        cstr->m_func.computeJacobian(x, J_);
        cstr->applyNormalConeProjectionJacobian(workspace.lamsPlusPre[i], J_);
        cstr->m_func.vectorHessianProduct(x, workspace.lamsPDAL[i], workspace.cstrVectorHessProd[i]);
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
      if (verbose)
        fmt::print(" | [{}] current M = {:.5g} | d1 = {:.3g}\n", __func__, merit0, d1);

      Scalar merit_trial = 0., dM = 0.;
      while (alpha_try >= alpha_min)
      {
        tryStep(workspace, results, alpha_try);
        merit_trial = merit_fun(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev);
        merit_trial += rho * prox_penalty.call(workspace.xTrial);
        dM = merit_trial - merit0;
        if (verbose)
          fmt::print(" | [{}] alpha {:.2e}, M = {:.5g}, dM = {:.5g}\n", __func__, alpha_try, merit_trial, dM);

        bool armijo_cond = dM <= armijo_c1 * alpha_try * d1;
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
      manifold.integrate(results.xOpt, alpha * workspace.pdStep.head(ndx), workspace.xTrial);

      int cursor = ndx;
      int nc = 0;
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        nc = problem->getConstraint(i)->nr();

        workspace.lamsTrial[i].noalias() = results.lamsOpt[i] + alpha * workspace.pdStep.segment(cursor, nc);

        cursor += nc;
      }
    }

  };

} // namespace lienlp

