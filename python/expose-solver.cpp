#include "proxnlp/python/fwd.hpp"
#include "proxnlp/solver-base.hpp"

namespace proxnlp {
namespace python {

void exposeSolver() {
  using context::Manifold;
  using context::Scalar;
  using Solver = context::Solver;
  using context::BCLParams;
  using context::ConstVectorRef;
  using context::Problem;
  using context::VectorRef;

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
      .value("LDLT_BLOCKED", LDLTChoice::BLOCKED)
      .value("LDLT_EIGEN", LDLTChoice::EIGEN)
      .value("LDLT_PROXSUITE", LDLTChoice::PROXSUITE)
      .export_values();

  using LinesearchOptions = Linesearch<Scalar>::Options;
  bp::class_<LinesearchOptions>(
      "LinesearchOptions", "Linesearch options.",
      bp::init<>(bp::args("self"), "Default constructor."))
      .def_readwrite("armijo_c1", &LinesearchOptions::armijo_c1)
      .def_readwrite("wolfe_c2", &LinesearchOptions::wolfe_c2)
      .def_readwrite(
          "dphi_thresh", &LinesearchOptions::dphi_thresh,
          "Threshold on the derivative at the initial point; the linesearch "
          "will be early-terminated if the derivative is below this threshold.")
      .def_readwrite("alpha_min", &LinesearchOptions::alpha_min,
                     "Minimum step size.")
      .def_readwrite("max_num_steps", &LinesearchOptions::max_num_steps)
      .def_readwrite("verbosity", &LinesearchOptions::verbosity)
      .def_readwrite("interp_type", &LinesearchOptions::interp_type,
                     "Interpolation type: bisection, quadratic or cubic.")
      .def_readwrite("contraction_min", &LinesearchOptions::contraction_min,
                     "Minimum step contraction.")
      .def_readwrite("contraction_max", &LinesearchOptions::contraction_max,
                     "Maximum step contraction.")
      .def(bp::self_ns::str(bp::self));

  using context::Results;
  using context::Workspace;
  using solve_std_vec_ins_t = ConvergenceFlag (Solver::*)(
      const ConstVectorRef &, const std::vector<VectorRef> &);
  using solve_eig_vec_ins_t = ConvergenceFlag (Solver::*)(
      const ConstVectorRef &, const ConstVectorRef &);

  bp::class_<Solver, boost::noncopyable>(
      "Solver", "The numerical solver.",
      bp::init<shared_ptr<Problem>, Scalar, Scalar, Scalar, VerboseLevel,
               Scalar, Scalar, Scalar, Scalar, Scalar, LDLTChoice>(
          (bp::arg("self"), bp::arg("problem"), bp::arg("tol") = 1e-6,
           bp::arg("mu_init") = 1e-2, bp::arg("rho_init") = 0.,
           bp::arg("verbose") = VerboseLevel::QUIET, bp::arg("mu_min") = 1e-9,
           bp::arg("prim_alpha") = 0.1, bp::arg("prim_beta") = 0.9,
           bp::arg("dual_alpha") = 1., bp::arg("dual_beta") = 1.,
           bp::arg("ldlt_choice") = LDLTChoice::DENSE)))
      .add_property("manifold",
                    bp::make_function(&Solver::manifold,
                                      bp::return_internal_reference<>()),
                    "The solver's working manifold.")
      .def_readwrite("hess_approx", &Solver::hess_approx)
      .def_readwrite("ls_strat", &Solver::ls_strat)
      .def("register_callback", &Solver::registerCallback,
           bp::args("self", "cb"), "Add a callback to the solver.")
      .def("clear_callbacks", &Solver::clearCallbacks, "Clear callbacks.",
           bp::args("self"))
      .def_readwrite("verbose", &Solver::verbose, "Solver verbose setting.")
      .def_readwrite("ldlt_choice", &Solver::ldlt_choice_,
                     "Use the BlockLDLT solver.")
      .def("setup", &Solver::setup, bp::args("self"),
           "Initialize the solver workspace and results.")
      .def("getResults", &Solver::getResults, bp::args("self"),
           bp::return_internal_reference<>(),
           "Get a reference to the results object.")
      .def("getWorkspace", &Solver::getWorkspace, bp::args("self"),
           bp::return_internal_reference<>(),
           "Get a reference to the workspace object.")
      .def<solve_std_vec_ins_t>(
          "solve", &Solver::solve, bp::args("self", "x0", "lams0"),
          "Run the solver (multiplier guesses given as a list).")
      .def<solve_eig_vec_ins_t>("solve", &Solver::solve,
                                (bp::arg("self"), bp::arg("x0"),
                                 bp::arg("lams0") = context::VectorXs(0)),
                                "Run the solver.")
      .def("setPenalty", &Solver::setPenalty, bp::args("self", "mu"),
           "Set the augmented Lagrangian penalty parameter.")
      .def("setDualPenalty", &Solver::setDualPenalty, bp::args("self", "gamma"),
           "Set the dual variable penalty for the linesearch merit "
           "function.")
      .def("setProxParameter", &Solver::setProxParameter,
           bp::args("self", "rho"),
           "Set the primal proximal penalty parameter.")
      .def_readwrite("mu_init", &Solver::mu_init_,
                     "Initial AL parameter value.")
      .def_readwrite("rho_init", &Solver::rho_init_,
                     "Initial proximal parameter value.")
      .def_readwrite("mu_lower", &Solver::mu_lower_,
                     "Lower bound :math:`\\underline{\\mu} > 0` for the "
                     "AL parameter.")
      .def_readwrite("mu_upper", &Solver::mu_upper_,
                     "Upper bound :math:`\\bar{\\mu}` for the AL parameter. "
                     "The tolerances for "
                     "each subproblem will be updated as :math:`\\eta^0 (\\mu "
                     "/ \\bar{\\mu})^\\gamma`")
      // BCL parameters
      .def_readwrite("bcl_params", &Solver::bcl_params, "BCL parameters.")
      .def_readwrite("target_tol", &Solver::target_tol, "Target tolerance.")
      .def_readwrite("ls_options", &Solver::ls_options, "Linesearch options.")
      .def_readwrite("mul_update_mode", &Solver::mul_update_mode,
                     "Type of multiplier update.")
      .def_readwrite("kkt_system", &Solver::kkt_system_, "KKT system type.")
      .def_readwrite("max_refinement_steps", &Solver::max_refinement_steps_,
                     "Maximum number of iterative refinement steps.")
      .def_readwrite("kkt_tolerance", &Solver::kkt_tolerance_,
                     "Acceptable tolerance for the KKT linear system "
                     "(threshold for iterative refinement).")
      .def_readwrite("max_iters", &Solver::max_iters,
                     "Maximum number of iterations.")
      .def_readwrite("max_al_iters", &Solver::max_al_iters,
                     "Max augmented Lagrangian iterations.")
      .def_readwrite("reg_init", &Solver::DELTA_INIT,
                     "Initial regularization.");
  bp::enum_<Solver::KktSystem>("KktSystem")
      .value("CLASSIC", Solver::KKT_CLASSIC)
      .value("PRIMAL_DUAL", Solver::KKT_PRIMAL_DUAL);

  bp::class_<BCLParams>("BCLParams",
                        "Parameters for the bound-constrained Lagrangian (BCL) "
                        "penalty update strategy.",
                        bp::init<>(bp::args("self")))
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
} // namespace proxnlp
