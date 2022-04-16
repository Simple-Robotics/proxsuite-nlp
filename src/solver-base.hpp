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

    bool verbose = false;
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
    const Scalar ls_beta;
    
    const Scalar DELTA_MIN = 1e-14;   /// Minimum nonzero regularization strength.
    const Scalar DELTA_MAX = 1e3;    /// Maximum regularization strength.

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
              const Scalar armijo_c1=1e-4,
              const Scalar ls_beta=0.5
              )
      : manifold(man)
      , problem(prob)
      , merit_fun(problem)
      , prox_penalty(manifold, manifold.neutral(), rho_init * MatrixXs::Identity(manifold.ndx(), manifold.ndx()))
      , verbose(verbose)
      , rho_init(rho_init)
      , mu_eq_init(mu_eq_init)
      , mu_factor(mu_factor)
      , mu_lower_(mu_lower_)
      , target_tol(tol)
      , prim_alpha(prim_alpha)
      , prim_beta(prim_beta)
      , dual_alpha(dual_alpha)
      , dual_beta(dual_beta)
      , alpha_min(alpha_min)
      , armijo_c1(armijo_c1)
      , ls_beta(ls_beta)
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
                          const std::vector<VectorRef>& lams0)
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
    std::size_t getMaxIters() const { return MAX_ITERS; }

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

      Scalar old_delta = 0.;
      Scalar del_up_k = 3.;
      Scalar del_down_k = 0.5;
      Scalar conditioning_ = 0;

      Scalar merit0 = results.merit;

      std::size_t k;
      for (k = 0; k < MAX_ITERS; k++)
      {

        //// precompute temp data

        results.value = problem->m_cost.call(x);
        problem->m_cost.computeGradient(x, workspace.objectiveGradient);
        problem->m_cost.computeHessian(x, workspace.objectiveHessian);

        if (verbose)
        {
          fmt::print("[iter {:>3d}] objective: {:g}\n", results.numIters, results.value);
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
          auto cstr = problem->getConstraint(i);

          MatrixRef& J_ = workspace.cstrJacobians[i];

          bool use_vhp = (use_gauss_newton && not cstr->disableGaussNewton()) || not use_gauss_newton; 
          if (use_vhp)
          {
            workspace.kktMatrix.topLeftCorner(ndx, ndx).noalias() += workspace.cstrVectorHessProd[i];
          }

          workspace.meritGradient.noalias() += J_.transpose() * workspace.lamsPDAL[i];

          // fill in the dual part of the KKT
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

        // Compute dual residual and infeasibility
        workspace.dualResidual = workspace.kktRhs.head(ndx);
        if (rho > 0.)
        {
          workspace.dualResidual -= rho * prox_penalty.computeGradient(x);
        }
        results.dualInfeas = math::infNorm(workspace.dualResidual);
        results.primalInfeas = 0.;
        for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
        {
          auto cstr = problem->getConstraint(i);
          results.primalInfeas = std::max(results.primalInfeas,
                                          math::infNorm(cstr->normalConeProjection(workspace.primalResiduals[i])));
        }
        // Compute inner stopping criterion
        Scalar inner_crit = math::infNorm(workspace.kktRhs);

        fmt::print(" | inner stop {:.4g}, d={:.3g}, p={:.3g}\n",
                  inner_tol, results.dualInfeas, results.primalInfeas);

        bool outer_cond = (results.primalInfeas <= target_tol && results.dualInfeas <= target_tol);
        if ((inner_crit <= inner_tol) || outer_cond)
        {
          return;
        }

        invokeCallbacks(workspace, results);

        /* Compute the step */

        // factorization
        // regularization strength
        Scalar delta = 0.;
        InertiaFlag is_inertia_correct = BAD;
        while (not(is_inertia_correct == OK) && delta <= DELTA_MAX)
        {
          correctInertia(workspace, delta, old_delta);
          workspace.ldlt_.compute(workspace.kktMatrix);
          conditioning_ = 1. / workspace.ldlt_.rcond();
          workspace.signature.array() = workspace.ldlt_.vectorD().array().sign().template cast<int>();
          is_inertia_correct = checkInertia(workspace.signature);
          if (delta == 0.) {
            delta = DELTA_MIN;
          } else {
            delta *= del_up_k;
          }
          old_delta = delta;
        }

        workspace.pdStep = -workspace.kktRhs;
        workspace.ldlt_.solveInPlace(workspace.pdStep);

        if (verbose)
        {
          fmt::print(" | conditioning:  {:.3g}\n", conditioning_);
        }

        assert(workspace.ldlt_.info() == Eigen::ComputationInfo::Success);

        //// Take the step

        merit0 = merit_fun(results.xOpt, results.lamsOpt, workspace.lamsPrev);
        merit0 += rho * prox_penalty.call(x);
        workspace.meritGradient.noalias() += rho * prox_penalty.computeGradient(x);
        results.merit = merit0;

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
        results.lamsOpt_data = workspace.lamsTrial_data;

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

    /// @brief    Correct the primal Hessian block of the KKT matrix to get the correct inertia.
    inline void correctInertia(Workspace& workspace, Scalar delta, Scalar old_delta) const
    {
      if (verbose)
        fmt::print(" | xreg : {:.3g}", delta);
      const int ndx = manifold.ndx();
      workspace.kktMatrix.diagonal().head(ndx).array() -= old_delta;
      workspace.kktMatrix.diagonal().head(ndx).array() += delta;
    }

    enum InertiaFlag
    {
      OK = 0,
      BAD = 1,
      ZEROS = 2
    };

    /// Check the matrix has the desired inertia.
    /// @param    kktMatrix The KKT matrix.
    /// @param    signature The computed inertia as a vector of ints valued -1, 0, or 1.
    const InertiaFlag checkInertia(const Eigen::VectorXi& signature) const
    {
      const int ndx = manifold.ndx();
      const int numc = problem->getTotalConstraintDim();
      const long n = signature.size();
      int numpos = 0;
      int numneg = 0;
      int numzer = 0;
      for (long i = 0; i < n; i++)
      {
        if (signature[i] > 0) numpos++;
        else if (signature[i] < 0) numneg++;
        else numzer++;
      }
      fmt::print(" | Inertia ({:d}+, {:d}, {:d}-)", numpos, numzer, numneg);
      if (numpos < ndx)
      {
        fmt::print(" is wrong: num+ < ndx!\n");
        return BAD;
      } else if (numneg < numc) {
        fmt::print(" is wrong: num- < num_cstr!\n");
        return BAD;
      } else if (numzer > 0) {
        fmt::print(" is wrong: there are null eigenvalues!\n");
        return ZEROS;
      }
      fmt::print(" is OK\n");
      return OK;
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
      workspace.lamsPrev_data = workspace.lamsPDAL_d;
    }

    /** 
     * Evaluate the primal residual vectors, and compute
     * the first-order and primal-dual Lagrange multiplier estimates.
     */
    void computeResidualsAndMultipliers(
      const ConstVectorRef& x,
      Workspace& workspace,
      VectorOfRef& lams) const
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

        cstr->m_func.computeJacobian(x, workspace.cstrJacobians[i]);
        cstr->applyNormalConeProjectionJacobian(workspace.lamsPlusPre[i], workspace.cstrJacobians[i]);
        cstr->m_func.vectorHessianProduct(x, workspace.lamsPDAL[i], workspace.cstrVectorHessProd[i]);
      }
    } 

    /**
     * Perform the inexact backtracking line-search procedure.
     * 
     * @param workspace Workspace.
     * @param results   Results struct.
     * @param merit0    Value of the merit function at the previous point.
     * @param d1        Directional derivative of the merit function in the search direction.
     */
    Scalar doLinesearch(Workspace& workspace, Results& results, const Scalar merit0, const Scalar d1) const
    {
      Scalar alpha_try = 1.;

      if (verbose)
        fmt::print(" | current M = {:.5g} | d1 = {:.3g}\n", merit0, d1);

      Scalar merit_trial = 0., dM = 0.;
      while (alpha_try >= alpha_min)
      {
        tryStep(workspace, results, alpha_try);
        if (std::abs(d1) < 1e-13)
        {
          return alpha_try;
        }
        merit_trial = merit_fun(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev);
        merit_trial += rho * prox_penalty.call(workspace.xTrial);
        dM = merit_trial - merit0;
        if (verbose)
          fmt::print(" | alpha {:5.2e}, M = {:5.5g}, dM = {:5.5g}\n", alpha_try, merit_trial, dM);

        if (dM <= armijo_c1 * alpha_try * d1)
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
    void tryStep(Workspace& workspace, Results& results, Scalar alpha) const
    {
      const int ndx = manifold.ndx();
      const long ntot = workspace.pdStep.rows();
      manifold.integrate(results.xOpt, alpha * workspace.pdStep.head(ndx), workspace.xTrial);
      workspace.lamsTrial_data = results.lamsOpt_data + alpha * workspace.pdStep.tail(ntot - ndx);
    }

    void invokeCallbacks(Workspace& workspace, Results& results)
    {
      for (auto cb : callbacks_)
      {
        cb->call(workspace, results);
      }
    }

  };

} // namespace lienlp

