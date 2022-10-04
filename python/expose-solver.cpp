#include "proxnlp/python/fwd.hpp"
#include "proxnlp/solver-base.hpp"

namespace proxnlp {
namespace python {

void exposeSolver() {
  using context::Manifold;
  using context::Scalar;
  using Solver = context::Solver;
  using context::ConstVectorRef;
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

  bp::enum_<LSInterpolation>("LSInterpolation",
                             "Linesearch interpolation scheme.")
      .value("BISECTION", LSInterpolation::BISECTION)
      .value("QUADRATIC", LSInterpolation::QUADRATIC)
      .value("CUBIC", LSInterpolation::CUBIC);

  using LSType = Linesearch<Scalar>;
  using LSOptions = LSType::Options;
  bp::class_<LSOptions>("LSOptions", "Linesearch options.",
                        bp::init<>(bp::args("self"), "Default constructor."))
      .def_readwrite("armijo_c1", &LSOptions::armijo_c1)
      .def_readwrite("wolfe_c2", &LSOptions::wolfe_c2)
      .def_readwrite(
          "dphi_thresh", &LSOptions::dphi_thresh,
          "Threshold on the derivative at the initial point; the linesearch "
          "will be early-terminated if the derivative is below this threshold.")
      .def_readwrite("alpha_min", &LSOptions::alpha_min, "Minimum step size.")
      .def_readwrite("max_num_steps", &LSOptions::max_num_steps)
      .def_readwrite("verbosity", &LSOptions::verbosity)
      .def_readwrite("interp_type", &LSOptions::interp_type,
                     "Interpolation type: bisection, quadratic or cubic.")
      .def_readwrite("contraction_min", &LSOptions::contraction_min,
                     "Minimum step contraction.")
      .def_readwrite("contraction_max", &LSOptions::contraction_max,
                     "Maximum step contraction.")
      .def(bp::self_ns::str(bp::self));

  bp::class_<Solver>(
      "Solver", "The numerical solver.",
      bp::init<shared_ptr<context::Problem> &, Scalar, Scalar, Scalar,
               VerboseLevel, Scalar, Scalar, Scalar, Scalar, Scalar>(
          (bp::arg("self"), bp::arg("problem"), bp::arg("tol") = 1e-6,
           bp::arg("mu_init") = 1e-2, bp::arg("rho_init") = 0.,
           bp::arg("verbose") = VerboseLevel::QUIET, bp::arg("mu_min") = 1e-9,
           bp::arg("prim_alpha") = 0.1, bp::arg("prim_beta") = 0.9,
           bp::arg("dual_alpha") = 1., bp::arg("dual_beta") = 1.)))
      .add_property("manifold",
                    bp::make_function(&Solver::manifold,
                                      bp::return_internal_reference<>()),
                    "The solver's working manifold.")
      .def_readwrite(
          "use_gauss_newton", &Solver::use_gauss_newton,
          "Whether to use a Gauss-Newton Hessian matrix approximation.")
      .def_readwrite("record_linesearch_process",
                     &Solver::record_linesearch_process)
      .def_readwrite("ls_strat", &Solver::ls_strat)
      .def_readwrite("ls_options", &Solver::ls_options, "Linesearch options.")
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
      .def("set_penalty", &Solver::setPenalty, bp::args("self", "mu"),
           "Set the augmented Lagrangian penalty parameter.")
      .def("set_prox_param", &Solver::setProxParameter, bp::args("self", "rho"),
           "Set the primal proximal penalty parameter.")
      .def("set_tolerance", &Solver::setTolerance, bp::args("self", "tol"),
           "Set the solver's target tolerance.")
      .def_readwrite("max_iters", &Solver::MAX_ITERS,
                     "Maximum number of iterations.")
      .def_readwrite("mu_factor", &Solver::mu_factor_,
                     "Multiplier update factor.")
      .def_readwrite("rho_factor", &Solver::rho_factor_,
                     "Proximal penalty update factor.")
      .def_readonly("prim_alpha", &Solver::prim_alpha_)
      .def_readonly("prim_beta", &Solver::prim_beta)
      .def_readonly("dual_alpha", &Solver::dual_alpha)
      .def_readonly("dual_beta", &Solver::dual_beta)
      .def_readonly("mu_min", &Solver::mu_lower_);
}
} // namespace python
} // namespace proxnlp
