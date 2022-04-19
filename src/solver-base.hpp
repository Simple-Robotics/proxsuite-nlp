/** Copyright (c) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"
#include "lienlp/meritfuncs/pdal.hpp"
#include "lienlp/workspace.hpp"
#include "lienlp/results.hpp"
#include "lienlp/helpers-base.hpp"

#include "lienlp/modelling/costs/squared-distance.hpp"

#ifndef NDEBUG
  #include <Eigen/Eigenvalues>
#endif

#include <cassert>
#include <stdexcept>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ostream.h>

namespace lienlp
{

  /// Verbosity level.
  enum VerboseLevel
  {
    QUIET=0,
    VERBOSE=1,
    VERY=2
  };

  template<typename _Scalar>
  class SolverTpl
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;

    using Workspace = WorkspaceTpl<Scalar>;
    using Results = ResultsTpl<Scalar>;
  
    using M = ManifoldAbstractTpl<Scalar>;

    /// Manifold on which to optimize.
    const M& manifold;
    shared_ptr<Problem> problem;
    /// Merit function.
    PDALFunction<Scalar> merit_fun;
    /// Proximal regularization penalty.
    QuadraticDistanceCost<Scalar> prox_penalty;

    //// Other settings

    bool verbose = QUIET;
    bool use_gauss_newton = false;          // Use a Gauss-Newton approximation for the Lagrangian Hessian.

    //// Algo params which evolve

    const Scalar inner_tol0 = 1.;
    const Scalar prim_tol0 = 1.;
    Scalar inner_tol = inner_tol0;
    Scalar prim_tol = prim_tol0;
    Scalar rho_init;                        // Initial primal proximal penalty parameter.
    Scalar rho = rho_init;                  // Primal proximal penalty parameter.
    Scalar mu_eq_init;                      // Initial penalty parameter.
    Scalar mu_eq = mu_eq_init;              // Penalty parameter.
    Scalar mu_eq_inv = 1. / mu_eq;          // Inverse penalty parameter.
    Scalar mu_factor;                       // Penalty update multiplicative factor.
    Scalar rho_factor = mu_factor;

    const Scalar inner_tol_min = 1e-9;      // Lower safeguard for the subproblem tolerance.
    Scalar mu_lower = 1e-9;                 // Lower safeguard for the penalty parameter.

    //// Algo hyperparams

    Scalar target_tol;                      // Target tolerance for the problem.
    const Scalar prim_alpha;                // BCL failure scaling (primal)
    const Scalar prim_beta;                 // BCL success scaling (primal)
    const Scalar dual_alpha;                // BCL failure scaling (dual)
    const Scalar dual_beta;                 // BCL success scaling (dual)

    const Scalar alpha_min;                 // Linesearch minimum step size.
    const Scalar armijo_c1;                 // Armijo rule c1 parameter.
    const Scalar ls_beta;                   // Linesearch step size decrease factor.
    
    const Scalar del_inc_k = 8.;
    const Scalar del_inc_big = 100.;
    const Scalar del_dec_k = 1./3.;

    const Scalar DELTA_MIN = 1e-14;         // Minimum nonzero regularization strength.
    const Scalar DELTA_MAX = 1e5;           // Maximum regularization strength.
    const Scalar DELTA_NONZERO_INIT = 1e-4;
    const Scalar DELTA_INIT = DELTA_MIN;

    /// Callbacks
    using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>; 
    std::vector<CallbackPtr> callbacks_;

    SolverTpl(const M& manifold,
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
      , mu_factor(mu_factor)
      , mu_lower(mu_lower)
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

    ConvergenceFlag solve(Workspace& workspace,
                          Results& results,
                          const ConstVectorRef& x0,
                          const std::vector<VectorRef>& lams0)
    {
      VectorXs new_lam(problem->getTotalConstraintDim());
      new_lam.setZero();
      int cursor = 0;
      int nr = 0;
      const std::size_t numc = problem->getNumConstraints();
      if (numc != lams0.size())
      {
        throw std::runtime_error("Specified number of constraints is not the same "
                                 "as the provided number of multipliers!");
      }
      for (std::size_t i = 0; i < numc; i++)
      {
        nr = problem->getConstraintDims()[i];
        new_lam.segment(cursor, nr) = lams0[i];
        cursor += nr;
      }
      return solve(workspace, results, x0, new_lam);
    }

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
    /// Get solver maximum number of iterations.
    std::size_t getMaxIters() const { return MAX_ITERS; }

    /// Update penalty parameter using the provided factor (with a safeguard SolverTpl::mu_lower).
    inline void updatePenalty()
    {
      if (mu_eq == mu_lower)
      {
        setPenalty(mu_eq_init);
      } else {
        setPenalty(std::max(mu_eq * mu_factor, mu_lower));
      }
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        typename Problem::ConstraintPtr cstr = problem->getConstraint(i);
        cstr->updateProxParameters(mu_eq);
      }
    }

    /// Set penalty parameter, its inverse and propagate to merit function.
    void setPenalty(const Scalar& new_mu)
    {
      mu_eq = new_mu;
      mu_eq_inv = 1. / mu_eq;
      merit_fun.setPenalty(mu_eq);
    }

    /// Set proximal penalty parameter.
    void setProxParam(const Scalar& new_rho)
    {
      rho = new_rho;
      prox_penalty.m_weights.setZero();
      prox_penalty.m_weights.diagonal().setConstant(rho);
    }

    /// @brief    Add a callback to the solver instance.
    inline void registerCallback(CallbackPtr cb)
    {
      callbacks_.push_back(cb);
    }

    /// @brief    Remove all callbacks from the instance.
    inline void clearCallbacks()
    {
      callbacks_.clear();
    }

  protected:
    std::size_t MAX_ITERS = 100;

    void solveInner(Workspace& workspace, Results& results)
    {
      const int ndx = manifold.ndx();
      const long ntot = workspace.kktRhs.size();
      const std::size_t num_c = problem->getNumConstraints();

      results.lamsOpt_data = workspace.lamsPrev_data;

      Scalar delta_last = 0.;
      Scalar delta = delta_last;
      Scalar old_delta = 0.;
      Scalar conditioning_ = 0;

      Scalar& merit0 = results.merit;
      merit_fun.setPenalty(mu_eq);

      std::size_t k;
      for (k = 0; k < MAX_ITERS; k++)
      {

        //// precompute temp data

        results.value = problem->m_cost.call(results.xOpt);
        problem->m_cost.computeGradient(results.xOpt, workspace.objectiveGradient);
        problem->m_cost.computeHessian(results.xOpt, workspace.objectiveHessian);

        computeResidualsAndMultipliers(results.xOpt, results.lamsOpt_data, workspace);
        computeResidualDerivatives(results.xOpt, workspace);

        merit0 = merit_fun(results.xOpt, results.lamsOpt, workspace.lamsPrev);
        if (rho > 0.)
          merit0 += prox_penalty.call(results.xOpt);

        if (verbose >= 0)
        {
          fmt::print("[iter {:>3d}] objective: {:g} merit: {:g}\n", results.numIters, results.value, merit0);
        }

        //// fill in LHS/RHS
        //// TODO create an Eigen::Map to map submatrices to the active sets of each constraint

        workspace.kktRhs.setZero();
        workspace.kktMatrix.setZero();

        workspace.meritGradient = workspace.objectiveGradient;
        workspace.kktRhs.head(ndx) = workspace.objectiveGradient;
        workspace.kktRhs.tail(ntot - ndx) = workspace.subproblemDualErr_data;

        workspace.kktMatrix.topLeftCorner(ndx, ndx)           = workspace.objectiveHessian;
        workspace.kktMatrix.topRightCorner(ndx, ntot - ndx)   = workspace.jacobians_data.transpose();
        workspace.kktMatrix.bottomLeftCorner(ntot - ndx, ndx) = workspace.jacobians_data;
        workspace.kktMatrix.bottomRightCorner(ntot - ndx, ntot - ndx).diagonal().setConstant(-mu_eq);

        // add jacobian-vector products to gradients
        workspace.meritGradient.noalias()    += workspace.jacobians_data.transpose() * workspace.lamsPDAL_data;
        workspace.kktRhs.head(ndx).noalias() += workspace.jacobians_data.transpose() * results.lamsOpt_data;

        // add proximal penalty terms
        VectorXs prox_grad = prox_penalty.computeGradient(results.xOpt);
        if (rho > 0.)
        {
          workspace.meritGradient.noalias() += prox_grad;
          workspace.kktRhs.head(ndx).noalias() += prox_grad;
          workspace.kktMatrix.topLeftCorner(ndx, ndx).noalias() += prox_penalty.computeHessian(results.xOpt);
        }

        for (std::size_t i = 0; i < num_c; i++)
        {
          const typename Problem::ConstraintPtr cstr = problem->getConstraint(i);
          cstr->computeActiveSet(workspace.primalResiduals[i], results.activeSet[i]);

          bool use_vhp = (use_gauss_newton && not cstr->disableGaussNewton()) || not use_gauss_newton; 
          if (use_vhp)
          {
            workspace.kktMatrix.topLeftCorner(ndx, ndx).noalias() += workspace.cstrVectorHessProd[i];
          }

        }

        // Compute dual residual and infeasibility
        workspace.dualResidual = workspace.kktRhs.head(ndx);
        if (rho > 0.)
          workspace.dualResidual.noalias() -= prox_grad;

        results.dualInfeas = math::infty_norm(workspace.dualResidual);
        results.primalInfeas = 0.;
        for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
        {
          typename Problem::ConstraintPtr cstr = problem->getConstraint(i);
          workspace.primalResiduals[i].noalias() = cstr->normalConeProjection(workspace.primalResiduals[i]);
        }
        results.primalInfeas = math::infty_norm(workspace.primalResiduals_data);
        // Compute inner stopping criterion
        Scalar inner_crit = math::infty_norm(workspace.kktRhs);

        fmt::print("| crit = {:>5.2e}, d={:>5.3g}, p={:>5.3g} (inner stop {:>5.2e})\n",
                  inner_crit, results.dualInfeas, results.primalInfeas, inner_tol);

        bool outer_cond = (results.primalInfeas <= target_tol && results.dualInfeas <= target_tol);
        if ((inner_crit <= inner_tol) || outer_cond)
        {
          return;
        }

        invokeCallbacks(workspace, results);

        /* Compute the step */

#ifndef NDEBUG
        Eigen::SelfAdjointEigenSolver<MatrixXs> eigsolve(workspace.kktRhs.size());
#endif
        // factorization
        // regularization strength : always try 0
        delta = DELTA_INIT;
        InertiaFlag is_inertia_correct = BAD;
        while (not(is_inertia_correct == OK) && delta <= DELTA_MAX)
        {
          correctInertia(workspace, delta, old_delta);
          workspace.ldlt_.compute(workspace.kktMatrix);
          conditioning_ = 1. / workspace.ldlt_.rcond();
          workspace.signature.array() = workspace.ldlt_.vectorD().array().sign().template cast<int>();
#ifndef NDEBUG
          eigsolve.compute(workspace.kktMatrix, Eigen::EigenvaluesOnly);
          if (verbose >= 2)
          {
            fmt::print("| KKT Eigenvalues:\n{}\n", eigsolve.eigenvalues().transpose());
          }
#endif
          is_inertia_correct = checkInertia(workspace.signature);

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
          old_delta = delta;
        }

        workspace.pdStep = -workspace.kktRhs;
        workspace.ldlt_.solveInPlace(workspace.pdStep);

        if (verbose >= 1)
        {
          VectorXs resdl = workspace.kktMatrix * workspace.pdStep + workspace.kktRhs;
          fmt::print("| KKT residual: {:>5.2e} | conditioning {:>4.3g} | xreg = {:>4.3g}\n",
                     math::infty_norm(resdl), conditioning_, delta);
        }

        assert(workspace.ldlt_.info() == Eigen::ComputationInfo::Success);

        //// Take the step

        Scalar total_directional_derivative = \
                  workspace.meritGradient.dot(workspace.pdStep.head(ndx)) \
                  - workspace.subproblemDualErr_data.dot(workspace.pdStep.tail(ntot - ndx));
        workspace.d1 = total_directional_derivative;

        doLinesearch(workspace, results, merit0, total_directional_derivative);
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
      const int ndx = manifold.ndx();
      workspace.kktMatrix.diagonal().head(ndx).array() -= old_delta;
      workspace.kktMatrix.diagonal().head(ndx).array() += delta;
    }

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
        switch (signature(i))
        {
        case 1 : numpos++;
                 break;
        case 0 : numzer++;
                 break;
        case -1: numneg++;
                 break;
        default: throw std::runtime_error("Matrix signature should only have Os, 1s, and -1s.");
        }
      }
      InertiaFlag flag = OK;
      bool print_info = verbose >= 2;
      if (print_info) fmt::print(" | Inertia ({:d}+, {:d}, {:d}-)", numpos, numzer, numneg);
      if (numpos < ndx)
      {
        if (print_info) fmt::print(" is wrong: num+ < ndx!\n");
        flag = BAD;
      } else if (numneg < numc) {
        if (print_info) fmt::print(" is wrong: num- < num_cstr!\n");
        flag = BAD;
      } else if (numzer > 0) {
        if (print_info) fmt::print(" is wrong: there are null eigenvalues!\n");
        flag = ZEROS;
      } else {
        if (print_info) fmt::print(" is OK\n");
      }
      return flag;
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
      workspace.lamsPrev_data = workspace.lamsPDAL_data;
    }

    /** 
     * Evaluate the primal residual vectors, and compute
     * the first-order and primal-dual Lagrange multiplier estimates.
     */
    void computeResidualsAndMultipliers(
      const ConstVectorRef& x,
      const ConstVectorRef& lams_data,
      Workspace& workspace) const
    {
      for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
      {
        typename Problem::ConstraintPtr cstr = problem->getConstraint(i);
        workspace.primalResiduals[i] = cstr->m_func(x);

        // multiplier
        workspace.lamsPlusPre[i] = workspace.lamsPrev[i] + mu_eq_inv * workspace.primalResiduals[i];
        workspace.lamsPlus[i] = cstr->normalConeProjection(workspace.lamsPlusPre[i]);
      }
      workspace.subproblemDualErr_data = mu_eq * (workspace.lamsPlus_data - lams_data);
      workspace.lamsPDAL_data = 2 * workspace.lamsPlus_data - lams_data;
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
        const typename Problem::ConstraintPtr cstr = problem->getConstraint(i);

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
    Scalar doLinesearch(Workspace& workspace, const Results& results, const Scalar merit0, const Scalar d1) const
    {
      Scalar alpha_try = 1.;

      if (verbose >= 2)
        fmt::print("| M = {:>5.3e} | d1 = {:>5.3e}\n", merit0, d1);

#ifndef NDEBUG
      std::vector<Scalar>& alphas_ = workspace.ls_alphas;
      std::vector<Scalar>& values_ = workspace.ls_values;
      alphas_.clear();
      values_.clear();
      VectorXs alphplot = VectorXs::LinSpaced(400, 0., 1.);
      for (long i = 0; i < alphplot.size(); i++)
      {
        tryStep(workspace, results, alphplot(i));
        alphas_.push_back(alphplot(i));
        values_.push_back(
          merit_fun(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev)
          + prox_penalty.call(workspace.xTrial)
        );
      }
#endif

      Scalar merit_trial = 0., dM = 0.;
      while (alpha_try > alpha_min)
      {
        tryStep(workspace, results, alpha_try);
        merit_trial = merit_fun(workspace.xTrial, workspace.lamsTrial, workspace.lamsPrev);
        if (rho > 0.) {
          merit_trial += prox_penalty.call(workspace.xTrial);
        }
#ifndef NDEBUG
        alphas_.push_back(alpha_try);
        values_.push_back(merit_trial);
#endif
        if (std::abs(d1) < 1e-13)
        {
          return alpha_try;
        }
        dM = merit_trial - merit0;
        if (verbose >= 2)
          fmt::print("| M = {:>5.3e} | dM = {:>5.3e} | alpha = {:>5.3e}\n", merit_trial, dM, alpha_try);

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

      if (verbose == 1)
      {
        fmt::print("| alpha_opt = {:>5.3e}\n", alpha_try);
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
    void tryStep(Workspace& workspace, const Results& results, Scalar alpha) const
    {
      const int ndx = manifold.ndx();
      const long ntot = workspace.kktRhs.rows();
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

