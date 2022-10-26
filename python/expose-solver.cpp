#include "proxnlp/python/fwd.hpp"
#include "proxnlp/solver-base.hpp"

namespace proxnlp {
namespace python {

void exposeSolver() {
  using context::Manifold;
  using context::Scalar;
  using Solver = context::Solver;
  using context::ConstVectorRef;
  using context::Problem;
  using context::VectorRef;

  bp::enum_<VerboseLevel>("VerboseLevel", "Verbose level for the solver.")
      .value("QUIET", QUIET)
      .value("VERBOSE", VERBOSE)
      .value("VERYVERBOSE", VERY)
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

  bp::class_<Solver>(
      "Solver", "The numerical solver.",
      bp::init<shared_ptr<Problem>, Scalar, Scalar, Scalar, VerboseLevel,
               Scalar, Scalar, Scalar, Scalar, Scalar>(
          (bp::arg("self"), bp::arg("problem"), bp::arg("tol") = 1e-6,
           bp::arg("mu_init") = 1e-2, bp::arg("rho_init") = 0.,
           bp::arg("verbose") = VerboseLevel::QUIET, bp::arg("mu_min") = 1e-9,
           bp::arg("prim_alpha") = 0.1, bp::arg("prim_beta") = 0.9,
           bp::arg("dual_alpha") = 1., bp::arg("dual_beta") = 1.)))
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
      .def("solve",
           (ConvergenceFlag(Solver::*)(context::Workspace &, context::Results &,
                                       const ConstVectorRef &,
                                       const std::vector<VectorRef> &)) &
               Solver::solve,
           bp::args("self", "workspace", "results", "x0", "lams0"),
           "Run the solver.")
      .def("solve",
           (ConvergenceFlag(Solver::*)(context::Workspace &, context::Results &,
                                       const ConstVectorRef &,
                                       const ConstVectorRef &)) &
               Solver::solve,
           bp::args("self", "workspace", "results", "x0", "lams0"),
           "Run the solver.")
      .def("solve",
           (ConvergenceFlag(Solver::*)(context::Workspace &, context::Results &,
                                       const ConstVectorRef &)) &
               Solver::solve,
           bp::args("self", "workspace", "results", "x0"),
           "Run the solver (without initial multiplier guess).")
      .def("setPenalty", &Solver::setPenalty, bp::args("self", "mu"),
           "Set the augmented Lagrangian penalty parameter.")
      .def("setDualPenalty", &Solver::setDualPenalty, bp::args("self", "gamma"),
           "Set the dual variable penalty for the linesearch merit function.")
      .def("setProxParameter", &Solver::setProxParameter,
           bp::args("self", "rho"),
           "Set the primal proximal penalty parameter.")
      .def_readwrite("mu_init", &Solver::mu_init_,
                     "Initial AL parameter value.")
      .def_readwrite("rho_init", &Solver::rho_init_,
                     "Initial proximal parameter value.")
      .def_readwrite(
          "mu_lower", &Solver::mu_lower_,
          "Lower bound :math:`\\underline{\\mu} > 0` for the AL parameter.")
      .def_readwrite("mu_upper", &Solver::mu_upper_,
                     "Upper bound :math:`\\bar{\\mu}` for the AL parameter. "
                     "The tolerances for "
                     "each subproblem will be updated as :math:`\\eta^0 (\\mu "
                     "/ \\bar{\\mu})^\\gamma`")
      // BCL parameters
      .def_readwrite("bcl_params", &Solver::bcl_params, "BCL parameters.")
      .def_readwrite("target_tol", &Solver::target_tol, "Target tolerance.")
      .def_readwrite("ls_options", &Solver::ls_options, "Linesearch options.")
      .def_readwrite("mul_update_mode", &Solver::mul_up_mode,
                     "Type of multiplier update.")
      .def_readwrite("max_iters", &Solver::max_iters,
                     "Maximum number of iterations.")
      .def_readwrite("reg_init", &Solver::DELTA_INIT,
                     "Initial regularization.");

  using BCLType = BCLParams<Scalar>;
  bp::class_<BCLType>("BCLParams",
                      "Parameters for the bound-constrained Lagrangian (BCL) "
                      "penalty update strategy.",
                      bp::init<>(bp::args("self")))
      .def_readwrite("prim_alpha", &BCLType::prim_alpha)
      .def_readwrite("prim_beta", &BCLType::prim_beta)
      .def_readwrite("dual_alpha", &BCLType::dual_alpha)
      .def_readwrite("dual_beta", &BCLType::dual_beta)
      .def_readwrite("mu_factor", &BCLType::mu_update_factor,
                     "Multiplier update factor.")
      .def_readwrite("rho_factor", &BCLType::rho_update_factor,
                     "Proximal penalty update factor.");
}
} // namespace python
} // namespace proxnlp
