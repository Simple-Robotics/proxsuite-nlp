#include "proxnlp/python/fwd.hpp"

#include "proxnlp/solver-base.hpp"


namespace proxnlp
{
  namespace python
  {

    void exposeSolver()
    {
      using context::Scalar;
      using context::Manifold;
      using Solver = context::Solver;
      using context::VectorRef;
      using context::ConstVectorRef;

      bp::enum_<VerboseLevel>("VerboseLevel", "Verbose level for the solver.")
        .value("QUIET", QUIET)
        .value("VERBOSE", VERBOSE)
        .value("VERYVERBOSE", VERY)
        .export_values()
        ;

      bp::enum_<LinesearchStrategy>("LinesearchStrategy")
        .value("ARMIJO", LinesearchStrategy::ARMIJO)
        .value("QUAD", LinesearchStrategy::QUADRATIC);

      bp::class_<Solver>(
        "Solver",
        "The numerical solver.",
        bp::init<const Manifold&,
                 shared_ptr<context::Problem>&,
                 Scalar,
                 Scalar,
                 Scalar,
                 VerboseLevel,
                 Scalar,
                 Scalar,
                 Scalar,
                 Scalar,
                 Scalar,
                 Scalar,
                 Scalar,
                 Scalar
                 >((  bp::arg("self")
                    , bp::arg("space")
                    , bp::arg("problem")
                    , bp::arg("tol") = 1e-6
                    , bp::arg("mu_init") = 1e-2
                    , bp::arg("rho_init") = 0.
                    , bp::arg("verbose") = VerboseLevel::QUIET
                    , bp::arg("mu_factor") = 0.1
                    , bp::arg("mu_min") = 1e-9
                    , bp::arg("prim_alpha") = 0.1
                    , bp::arg("prim_beta") = 0.9
                    , bp::arg("dual_alpha") = 1.
                    , bp::arg("dual_beta") = 1.
                    , bp::arg("alpha_min") = 1e-7
                    , bp::arg("armijo_c1") = 1e-4
                    ))
      )
        .def_readwrite("use_gauss_newton", &Solver::use_gauss_newton, "Whether to use a Gauss-Newton Hessian matrix approximation.")
        .def_readwrite("record_linesearch_process", &Solver::record_linesearch_process)
        .def("register_callback", &Solver::registerCallback, bp::args("self", "cb"), "Add a callback to the solver.")
        .def("clear_callbacks", &Solver::clearCallbacks, "Clear callbacks.")
        .def_readwrite("verbose", &Solver::verbose, "Solver verbose setting.")
        .def("solve",
             (ConvergenceFlag(Solver::*)(context::Workspace&, context::Results&,
                                         const ConstVectorRef&, const std::vector<VectorRef>&))&Solver::solve,
             bp::args("workspace", "results", "x0", "lams0"))
        .def("set_penalty",    &Solver::setPenalty,   bp::args("self", "mu"), "Set the augmented Lagrangian penalty parameter.")
        .def("set_prox_param", &Solver::setProxParameter, bp::args("self", "rho"), "Set the primal proximal penalty parameter.")
        .def("set_tolerance",  &Solver::setTolerance, bp::args("self", "tol"), "Set the solver's target tolerance.")
        .add_property("maxiters",
                      &Solver::getMaxIters,
                      &Solver::setMaxIters,
                      "Maximum number of iterations.")
        .def_readonly("prim_alpha", &Solver::prim_alpha)
        .def_readonly("prim_beta", &Solver::prim_beta)
        .def_readonly("dual_alpha", &Solver::dual_alpha)
        .def_readonly("dual_beta", &Solver::dual_beta)
        .def_readonly("mu_min", &Solver::mu_lower_)
        .def_readonly("alpha_min", &Solver::alpha_min)
        .def_readonly("armijo_c1", &Solver::armijo_c1)
        .def_readwrite("ls_beta", &Solver::ls_beta)
        .def_readwrite("ls_strat", &Solver::ls_strat)
        ;
    }
  } // namespace python
} // namespace proxnlp
