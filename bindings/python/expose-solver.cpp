#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/prox-solver.hpp"
#include <eigenpy/std-unique-ptr.hpp>
#include <eigenpy/deprecation-policy.hpp>

namespace proxsuite {
namespace nlp {
namespace python {

void exposeSolver() {
  using context::Manifold;
  using context::Scalar;
  using ProxNLPSolver = context::ProxNLPSolverTpl;
  using context::BCLParams;
  using context::ConstVectorRef;
  using context::Problem;
  using context::VectorRef;
  using eigenpy::deprecated_member;
  using eigenpy::DeprecationType;
  using eigenpy::ReturnInternalStdUniquePtr;

  bp::enum_<VerboseLevel>("VerboseLevel", "Verbose level for the solver.")
      .value("QUIET", QUIET)
      .value("VERBOSE", VERBOSE)
      .value("VERYVERBOSE", VERYVERBOSE)
      .export_values();

  bp::enum_<LinesearchStrategy>(
      "LinesearchStrategy",
      "Linesearch strategy. Only Armijo linesearch is implemented for now.")
      .value("ARMIJO", LinesearchStrategy::ARMIJO);

  bp::enum_<HessianApprox>("HessianApprox",
                           "Type of approximation of the Lagrangian Hessian.")
      .value("HESSIAN_EXACT", HessianApprox::EXACT)
      .value("HESSIAN_GAUSS_NEWTON", HessianApprox::GAUSS_NEWTON)
      .export_values();

  bp::enum_<MultiplierUpdateMode>("MultiplierUpdateMode",
                                  "Type of multiplier update.")
      .value("MUL_NEWTON", MultiplierUpdateMode::NEWTON)
      .value("MUL_PRIMAL", MultiplierUpdateMode::PRIMAL)
      .value("MUL_PRIMAL_DUAL", MultiplierUpdateMode::PRIMAL_DUAL)
      .export_values();

  bp::enum_<LSInterpolation>("LSInterpolation",
                             "Linesearch interpolation scheme.")
      .value("BISECTION", LSInterpolation::BISECTION)
      .value("QUADRATIC", LSInterpolation::QUADRATIC)
      .value("CUBIC", LSInterpolation::CUBIC);

  bp::enum_<LDLTChoice>("LDLTChoice", "Choice of LDLT solver.")
      .value("LDLT_DENSE", LDLTChoice::DENSE)
      .value("LDLT_BUNCHKAUFMAN", LDLTChoice::BUNCHKAUFMAN)
      .value("LDLT_BLOCKSPARSE", LDLTChoice::BLOCKSPARSE)
      .value("LDLT_EIGEN", LDLTChoice::EIGEN)
      .value("LDLT_PROXSUITE", LDLTChoice::PROXSUITE)
      .export_values();

  using LinesearchOptions = Linesearch<Scalar>::Options;
  bp::class_<LinesearchOptions>("LinesearchOptions", "Linesearch options.",
                                bp::init<>(("self"_a), "Default constructor."))
      .def_readwrite("armijo_c1", &LinesearchOptions::armijo_c1)
      .def_readwrite("wolfe_c2", &LinesearchOptions::wolfe_c2)
      .def_readwrite(
          "dphi_thresh", &LinesearchOptions::dphi_thresh,
          "Threshold on the derivative at the initial point; the linesearch "
          "will be early-terminated if the derivative is below this threshold.")
      .def_readwrite("alpha_min", &LinesearchOptions::alpha_min,
                     "Minimum step size.")
      .def_readwrite("max_num_steps", &LinesearchOptions::max_num_steps)
      .def_readwrite("interp_type", &LinesearchOptions::interp_type,
                     "Interpolation type: bisection, quadratic or cubic.")
      .def_readwrite("contraction_min", &LinesearchOptions::contraction_min,
                     "Minimum step contraction.")
      .def_readwrite("contraction_max", &LinesearchOptions::contraction_max,
                     "Maximum step contraction.")
      .def(bp::self_ns::str(bp::self));

  using context::Results;
  using context::Workspace;
  using solve_std_vec_ins_t = ConvergenceFlag (ProxNLPSolver::*)(
      const ConstVectorRef &, const std::vector<VectorRef> &);
  using solve_eig_vec_ins_t = ConvergenceFlag (ProxNLPSolver::*)(
      const ConstVectorRef &, const ConstVectorRef &);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::class_<ProxNLPSolver, boost::noncopyable>(
      "ProxNLPSolver",
      "Semi-smooth Newton-based solver for nonlinear optimization using a "
      "primal-dual method of multipliers. This solver works by approximately "
      "solving the proximal subproblems in the method of multipliers.",
      bp::init<Problem &, Scalar, Scalar, Scalar, VerboseLevel, Scalar, Scalar,
               Scalar, Scalar, Scalar, LDLTChoice>(
          ("self"_a, "problem", "tol"_a = 1e-6, "mu_init"_a = 1e-2,
           "rho_init"_a = 0., "verbose"_a = VerboseLevel::QUIET,
           "mu_min"_a = 1e-9, "prim_alpha"_a = 0.1, "prim_beta"_a = 0.9,
           "dual_alpha"_a = 1., "dual_beta"_a = 1.,
           "ldlt_choice"_a = LDLTChoice::DENSE)))
      .add_property("problem",
                    bp::make_function(&ProxNLPSolver::problem,
                                      bp::return_internal_reference<>()),
                    "The general nonlinear program to solve.")
      .add_property("manifold",
                    bp::make_function(&ProxNLPSolver::manifold,
                                      bp::return_internal_reference<>()),
                    "The solver's working manifold.")
      .def_readwrite("hess_approx", &ProxNLPSolver::hess_approx)
      .def_readwrite("ls_strat", &ProxNLPSolver::ls_strat)
      .def("register_callback", &ProxNLPSolver::registerCallback,
           ("self"_a, "cb"), "Add a callback to the solver.")
      .def("clear_callbacks", &ProxNLPSolver::clearCallbacks,
           "Clear callbacks.", ("self"_a))
      .def_readwrite("verbose", &ProxNLPSolver::verbose,
                     "Solver verbose setting.")
      .def_readwrite("ldlt_choice", &ProxNLPSolver::ldlt_choice_,
                     "Use the BlockLDLT solver.")
      .def("setup", &ProxNLPSolver::setup, ("self"_a),
           "Initialize the solver workspace and results.")
      .def("getResults", &ProxNLPSolver::getResults, ("self"_a),
           deprecated_member<DeprecationType::DEPRECATION,
                             bp::return_internal_reference<>>(
               "This getter has been deprecated."),
           "Get a reference to the results object.")
      .def("getWorkspace", &ProxNLPSolver::getWorkspace, ("self"_a),
           deprecated_member<DeprecationType::DEPRECATION,
                             bp::return_internal_reference<>>(
               "This getter has been deprecated."),
           "Get a reference to the workspace object.")
      .add_property("workspace", bp::make_getter(&ProxNLPSolver::workspace_,
                                                 ReturnInternalStdUniquePtr{}))
      .add_property("results", bp::make_getter(&ProxNLPSolver::results_,
                                               ReturnInternalStdUniquePtr{}))
      .def<solve_std_vec_ins_t>(
          "solve", &ProxNLPSolver::solve, ("self"_a, "x0", "lams0"),
          "Run the solver (multiplier guesses given as a list).")
      .def<solve_eig_vec_ins_t>(
          "solve", &ProxNLPSolver::solve,
          ("self"_a, "x0", "lams0"_a = context::VectorXs(0)), "Run the solver.")
      .def("setPenalty", &ProxNLPSolver::setPenalty, ("self"_a, "mu"),
           "Set the augmented Lagrangian penalty parameter.")
      .def("setDualPenalty", &ProxNLPSolver::setDualPenalty,
           ("self"_a, "gamma"),
           "Set the dual variable penalty for the linesearch merit "
           "function.")
      .def("setProxParameter", &ProxNLPSolver::setProxParameter,
           ("self"_a, "rho"), "Set the primal proximal penalty parameter.")
      .def_readwrite("mu_init", &ProxNLPSolver::mu_init_,
                     "Initial AL parameter value.")
      .def_readwrite("rho_init", &ProxNLPSolver::rho_init_,
                     "Initial proximal parameter value.")
      .def_readwrite("mu_lower", &ProxNLPSolver::mu_lower_,
                     "Lower bound :math:`\\underline{\\mu} > 0` for the "
                     "AL parameter.")
      .def_readwrite("mu_upper", &ProxNLPSolver::mu_upper_,
                     "Upper bound :math:`\\bar{\\mu}` for the AL parameter. "
                     "The tolerances for "
                     "each subproblem will be updated as :math:`\\eta^0 (\\mu "
                     "/ \\bar{\\mu})^\\gamma`")
      // BCL parameters
      .def_readwrite("bcl_params", &ProxNLPSolver::bcl_params,
                     "BCL parameters.")
      .def_readwrite("target_tol", &ProxNLPSolver::target_tol,
                     "Target tolerance.")
      .def_readwrite("ls_options", &ProxNLPSolver::ls_options,
                     "Linesearch options.")
      .def_readwrite("mul_update_mode", &ProxNLPSolver::mul_update_mode,
                     "Type of multiplier update.")
      .def_readwrite("kkt_system", &ProxNLPSolver::kkt_system_,
                     "KKT system type.")
      .def_readwrite("max_refinement_steps",
                     &ProxNLPSolver::max_refinement_steps_,
                     "Maximum number of iterative refinement steps.")
      .def_readwrite("kkt_tolerance", &ProxNLPSolver::kkt_tolerance_,
                     "Acceptable tolerance for the KKT linear system "
                     "(threshold for iterative refinement).")
      .def_readwrite("max_iters", &ProxNLPSolver::max_iters,
                     "Maximum number of iterations.")
      .def_readwrite("max_al_iters", &ProxNLPSolver::max_al_iters,
                     "Max augmented Lagrangian iterations.")
      .def_readwrite("reg_init", &ProxNLPSolver::DELTA_INIT,
                     "Initial regularization.");
#pragma GCC diagnostic pop

  bp::enum_<KktSystem>("KktSystem")
      .value("KKT_CLASSIC", KKT_CLASSIC)
      .value("KKT_PRIMAL_DUAL", KKT_PRIMAL_DUAL)
      .export_values();

  bp::class_<BCLParams>("BCLParams",
                        "Parameters for the bound-constrained Lagrangian (BCL) "
                        "penalty update strategy.",
                        bp::init<>(("self"_a)))
      .def_readwrite("prim_alpha", &BCLParams::prim_alpha)
      .def_readwrite("prim_beta", &BCLParams::prim_beta)
      .def_readwrite("dual_alpha", &BCLParams::dual_alpha)
      .def_readwrite("dual_beta", &BCLParams::dual_beta)
      .def_readwrite("mu_factor", &BCLParams::mu_update_factor,
                     "Multiplier update factor.")
      .def_readwrite("rho_factor", &BCLParams::rho_update_factor,
                     "Proximal penalty update factor.");
}
} // namespace python
} // namespace nlp
} // namespace proxsuite
