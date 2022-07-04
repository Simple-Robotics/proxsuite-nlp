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
    Scalar rho_init_;                       // Initial primal proximal penalty parameter.
    Scalar rho_ = rho_init_;                // Primal proximal penalty parameter.
    Scalar mu_init_;                        // Initial penalty parameter.
    Scalar mu_ = mu_init_;                  // Penalty parameter.
    Scalar mu_inv_ = 1. / mu_;              // Inverse penalty parameter.
    Scalar mu_factor_ = 0.1;                // Penalty update multiplicative factor.
    Scalar rho_factor_ = mu_factor_;        // Primal penalty update factor.

    const Scalar inner_tol_min = 1e-9;      // Lower safeguard for the subproblem tolerance.
    Scalar mu_lower_ = 1e-9;                // Lower safeguard for the penalty parameter.

    //// Algo hyperparams

    Scalar target_tol;                      // Target tolerance for the problem.
    const Scalar prim_alpha_;                // BCL failure scaling (primal)
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

    std::size_t MAX_ITERS = 100;

    /// Callbacks
    using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>; 
    std::vector<CallbackPtr> callbacks_;

    SolverTpl(const Manifold& manifold,
              const shared_ptr<Problem>& prob,
              const Scalar tol=1e-6,
              const Scalar mu_eq_init=1e-2,
              const Scalar rho_init=0.,
              const VerboseLevel verbose=QUIET,
              const Scalar mu_lower=1e-9,
              const Scalar prim_alpha=0.1,
              const Scalar prim_beta=0.9,
              const Scalar dual_alpha=1.,
              const Scalar dual_beta=1.,
              const Scalar alpha_min=1e-7,
              const Scalar armijo_c1=1e-4,
              const Scalar ls_beta=0.5);

    enum InertiaFlag
    {
      OK = 0,
      BAD = 1,
      ZEROS = 2
    };

    /**
     * @brief Solve the problem.
     * 
     * @param workspace
     * @param results
     * @param x0    Initial guess.
     * @param lams0 Initial Lagrange multipliers given separately for each constraint.
     * 
     */
    ConvergenceFlag solve(Workspace& workspace,
                          Results& results,
                          const ConstVectorRef& x0,
                          const std::vector<VectorRef>& lams0);

    /**
     * @copybrief solve().
     * 
     * @param workspace
     * @param results
     * @param x0    Initial guess.
     * @param lams0 Initial Lagrange multipliers given separately for each constraint.
     * 
     */
    ConvergenceFlag solve(Workspace& workspace,
                          Results& results,
                          const ConstVectorRef& x0,
                          const ConstVectorRef& lams0);

    /// Set solver convergence threshold.
    void setTolerance(const Scalar tol) { target_tol = tol; }

    void solveInner(Workspace& workspace, Results& results);

    /// Update penalty parameter using the provided factor (with a safeguard SolverTpl::mu_lower).
    inline void updatePenalty();

    /// @brief Set penalty parameter, its inverse and update the merit function.
    /// @param new_mu The new penalty parameter.
    void setPenalty(const Scalar& new_mu);
    
    /// Set proximal penalty parameter.
    void setProxParameter(const Scalar& new_rho);

    /// @brief    Add a callback to the solver instance.
    inline void registerCallback(const CallbackPtr& cb) { callbacks_.push_back(cb); }

    /// @brief    Remove all callbacks from the instance.
    inline void clearCallbacks() { callbacks_.clear(); }

    /**
     * @brief Update primal-dual subproblem tolerances upon failure (insufficient primal feasibility)
     * 
     * This is called upon initialization of the solver.
     */
    void updateToleranceFailure();

    /**
     * @brief Update primal-dual subproblem tolerances upon  successful outer-loop iterate (good primal feasibility)
     */
    void updateToleranceSuccess();

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

    /** 
     * @brief Check the matrix has the desired inertia.
     * @param signature The computed inertia as a vector of ints valued -1, 0, or 1.
     * @param delta     Scale factor for the identity matrix to add
     */
    InertiaFlag checkInertia(const Eigen::VectorXi& signature) const;

  };

} // namespace proxnlp

#include "proxnlp/solver-base.hxx"
