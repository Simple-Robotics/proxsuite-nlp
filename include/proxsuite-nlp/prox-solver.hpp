/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/fwd.hpp"
#include "proxsuite-nlp/problem-base.hpp"
#include "proxsuite-nlp/pdal.hpp"
#include "proxsuite-nlp/workspace.hpp"
#include "proxsuite-nlp/results.hpp"
#include "proxsuite-nlp/helpers-base.hpp"
#include "proxsuite-nlp/logger.hpp"
#include "proxsuite-nlp/bcl-params.hpp"

#include <boost/mpl/bool.hpp>

#include "proxsuite-nlp/modelling/costs/squared-distance.hpp"

#include "proxsuite-nlp/linesearch-base.hpp"

namespace proxsuite {
namespace nlp {

enum class MultiplierUpdateMode { NEWTON, PRIMAL, PRIMAL_DUAL };

enum class HessianApprox {
  /// Exact Hessian construction from provided function Hessians
  EXACT,
  /// Gauss-Newton (or rather SCQP) approximation
  GAUSS_NEWTON,
};

enum InertiaFlag { INERTIA_OK = 0, INERTIA_BAD = 1, INERTIA_HAS_ZEROS = 2 };

enum KktSystem { KKT_CLASSIC, KKT_PRIMAL_DUAL };

/// Semi-smooth Newton-based solver for nonlinear optimization using a
/// primal-dual method of multipliers. This solver works by approximately
/// solving the proximal subproblems in the method of multipliers.
template <typename _Scalar> class ProxNLPSolverTpl {
public:
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using LinesearchOptions = typename Linesearch<Scalar>::Options;
  using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>;
  using ConstraintSet = ConstraintSetBase<Scalar>;
  using ConstraintObject = ConstraintObjectTpl<Scalar>;

  /// Manifold on which to optimize.
  shared_ptr<Problem> problem_;
  /// Merit function.
  ALMeritFunctionTpl<Scalar> merit_fun;
  /// Proximal regularization penalty.
  QuadraticDistanceCostTpl<Scalar> prox_penalty;

  /// Level of verbosity of the solver.
  VerboseLevel verbose = QUIET;
  /// Use a Gauss-Newton approximation for the Lagrangian Hessian.
  HessianApprox hess_approx = HessianApprox::GAUSS_NEWTON;
  /// Linesearch strategy.
  LinesearchStrategy ls_strat = LinesearchStrategy::ARMIJO;
  MultiplierUpdateMode mul_update_mode = MultiplierUpdateMode::NEWTON;

  /// linear algebra opts
  std::size_t max_refinement_steps_ = 5;
  Scalar kkt_tolerance_ = 1e-13;
  LDLTChoice ldlt_choice_;
  KktSystem kkt_system_ = KKT_CLASSIC;

  //// Algorithm proximal parameters

  Scalar inner_tol0 = 1.;
  Scalar prim_tol0 = 1.;
  Scalar inner_tol_ = inner_tol0;
  Scalar prim_tol_ = prim_tol0;
  Scalar rho_init_; //< Initial primal proximal penalty parameter.
  Scalar mu_init_;  //< Initial penalty parameter.
private:
  Scalar rho_ = rho_init_;   //< Primal proximal penalty parameter.
  Scalar mu_ = mu_init_;     //< Penalty parameter.
  Scalar mu_inv_ = 1. / mu_; //< Inverse penalty parameter.
public:
  Scalar inner_tol_min = 1e-9; //< Lower safeguard for the subproblem tolerance.
  Scalar mu_upper_ = 1.;       //< Upper safeguard for the penalty parameter.
  Scalar mu_lower_ = 1e-9;     //< Lower safeguard for the penalty parameter.
  Scalar pdal_beta_ = 0.5;     //< primal-dual weight for the dual variables.

  /// BCL strategy parameters.
  BCLParamsTpl<Scalar> bcl_params;

  /// Linesearch options.
  LinesearchOptions ls_options;

  /// Target tolerance for the problem.
  Scalar target_tol;

  /// Logger.
  BaseLogger logger{};

  //// Parameters for the inertia-correcting strategy.

  const Scalar del_inc_k = 8.;
  const Scalar del_inc_big = 100.;
  const Scalar del_dec_k = 1. / 3.;

  const Scalar DELTA_MIN = 1e-14; // Minimum nonzero regularization strength.
  const Scalar DELTA_MAX = 1e6;   // Maximum regularization strength.
  const Scalar DELTA_NONZERO_INIT = 1e-4;
  Scalar DELTA_INIT = 0.;

  /// Solver maximum number of iterations.
  std::size_t max_iters = 100;
  std::size_t max_al_iters = 1000;

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  unique_ptr<Workspace> workspace_;
  unique_ptr<Results> results_;

  ProxNLPSolverTpl(shared_ptr<Problem> prob, const Scalar tol = 1e-6,
                   const Scalar mu_eq_init = 1e-2, const Scalar rho_init = 0.,
                   const VerboseLevel verbose = QUIET,
                   const Scalar mu_lower = 1e-9, const Scalar prim_alpha = 0.1,
                   const Scalar prim_beta = 0.9, const Scalar dual_alpha = 1.,
                   const Scalar dual_beta = 1.,
                   LDLTChoice ldlt_blocked = LDLTChoice::BUNCHKAUFMAN,
                   const LinesearchOptions ls_options = LinesearchOptions());

  const Manifold &manifold() const { return *problem_->manifold_; }

  void setup() {
    workspace_ = std::make_unique<Workspace>(*problem_, ldlt_choice_);
    results_ = std::make_unique<Results>(*problem_);
  }

  /**
   * @brief Solve the problem.
   *
   * @param x0    Initial guess.
   * @param lams0 Initial Lagrange multipliers given separately for each
   * constraint.
   *
   */
  ConvergenceFlag solve(const ConstVectorRef &x0,
                        const std::vector<VectorRef> &lams0);

  PROXSUITE_NLP_DEPRECATED
  const Workspace &getWorkspace() const { return *workspace_; }
  PROXSUITE_NLP_DEPRECATED
  const Results &getResults() const { return *results_; }

  /**
   * @copybrief solve().
   *
   * @param x0    Initial guess.
   * @param lams0 Initial Lagrange multipliers given separately for each
   * constraint.
   *
   */
  ConvergenceFlag solve(const ConstVectorRef &x0,
                        const ConstVectorRef &lams0 = VectorXs(0));

  void innerLoop(Workspace &workspace, Results &results);

  void assembleKktMatrix(Workspace &workspace);

  /// Iterative refinement of the KKT linear system.
  PROXSUITE_NLP_INLINE bool iterativeRefinement(Workspace &workspace) const;

  /// Update penalty parameter using the provided factor (with a safeguard
  /// ProxNLPSolverTpl::mu_lower).
  inline void updatePenalty();

  /// @brief Set the dual penalty weight for the merit function.
  void setDualPenalty(const Scalar beta) { pdal_beta_ = beta; }

  /// @brief Set penalty parameter, its inverse and update the merit function.
  /// @param new_mu The new penalty parameter.
  void setPenalty(const Scalar &new_mu) noexcept;

  /// Set proximal penalty parameter.
  void setProxParameter(const Scalar &new_rho) noexcept;

  /// @brief    Add a callback to the solver instance.
  inline void registerCallback(const CallbackPtr &cb) noexcept {
    callbacks_.push_back(cb);
  }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() noexcept { callbacks_.clear(); }

  /**
   * @brief Update primal-dual subproblem tolerances upon failure (insufficient
   * primal feasibility)
   *
   * This is called upon initialization of the solver.
   */
  void updateToleranceFailure() noexcept;

  /**
   * @brief Update primal-dual subproblem tolerances upon  successful outer-loop
   * iterate (good primal feasibility)
   */
  void updateToleranceSuccess() noexcept;

  inline void tolerancePostUpdate() noexcept;

  /// @brief  Accept Lagrange multiplier estimates.
  void acceptMultipliers(Results &results, Workspace &workspace) const;

  /**
   * Evaluate the problem data, as well as the proximal/projection operators,
   * and the first-order & primal-dual multiplier estimates.
   *
   * @param inner_lams_data Inner (SQP) dual variables
   * @param workspace       Problem workspace.
   */
  void computeMultipliers(const ConstVectorRef &inner_lams_data,
                          Workspace &workspace) const;

  /**
   * Evaluate the derivatives (cost gradient, Hessian, constraint Jacobians,
   * vector-Hessian products) of the problem data.
   *
   * @param x         Primal variable
   * @param workspace Problem workspace.
   * @param second_order Whether to compute the second-order information; set to
   * false for e.g. linesearch.
   */
  void computeProblemDerivatives(const ConstVectorRef &x, Workspace &workspace,
                                 boost::mpl::false_) const;
  /// @copydoc computeProblemDerivatives()
  void computeProblemDerivatives(const ConstVectorRef &x, Workspace &workspace,
                                 boost::mpl::true_) const;

  /**
   * Compute the primal residuals at the current primal-dual pair \f$(x,
   * \lambda^+)\f$, where the multipliers are chosen to be the predicted next
   * ALM iterate.
   *
   */
  void computePrimalResiduals(Workspace &workspace, Results &results) const;

  /**
   * Take a trial step.
   *
   * @param workspace Workspace
   * @param results   Contains the previous primal-dual point
   * @param alpha     Step size
   */
  void tryStep(Workspace &workspace, const Results &results, Scalar alpha);

  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (auto cb : callbacks_) {
      cb->call(workspace, results);
    }
  }
};

/// @brief Check the matrix has the desired inertia.
/// @param signature The computed inertia as a vector of ints valued -1, 0,
/// or 1.
/// @param delta     Scale factor for the identity matrix to add
inline InertiaFlag checkInertia(const int ndx, const int nc,
                                const Eigen::VectorXi &signature);

} // namespace nlp
} // namespace proxsuite

#include "proxsuite-nlp/prox-solver.hxx"

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/prox-solver.txx"
#endif
