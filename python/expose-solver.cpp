#include "lienlp/python/fwd.hpp"

#include "lienlp/solver-base.hpp"


namespace lienlp
{
  namespace python
  {

    void exposeSolver()
    {
      using context::Scalar;
      using context::Manifold;
      using Solver = context::Solver;

      bp::class_<Solver>(
        "Solver", "The numerical solver.",
        bp::init<const Manifold&,
                 shared_ptr<context::Problem>&,
                 Scalar,
                 Scalar,
                 Scalar,
                 bool,
                 Scalar,
                 Scalar
                 >((  bp::arg("self")
                    , bp::arg("space")
                    , bp::arg("problem")
                    , bp::arg("tol") = 1e-6
                    , bp::arg("mu_init") = 1e-2
                    , bp::arg("rho_init") = 0.
                    , bp::arg("verbose") = false
                    , bp::arg("mu_factor") = 0.1
                    , bp::arg("mu_min") = 1e-9
                    ))
      )
        .def_readwrite("use_gauss_newton", &Solver::use_gauss_newton, "Whether to use a Gauss-Newton Hessian matrix approximation.")
        .def("register_callback", &Solver::registerCallback, bp::args("cb"), "Add a callback to the solver.")
        .def("clear_callbacks", &Solver::clearCallbacks, "Clear callbacks.")
        .def_readwrite("verbose", &Solver::verbose, "Solver verbose setting.")
        .def("solve", &Solver::solve,
             bp::args("workspace", "results", "x0", "lams0"))
        .def("set_penalty",    &Solver::setPenalty,   bp::args("self", "mu"), "Set the augmented Lagrangian penalty parameter.")
        .def("set_prox_param", &Solver::setProxParam, bp::args("self", "rho"), "Set the primal proximal penalty parameter.")
        .def("set_maxiters",   &Solver::setMaxIters,  bp::args("self", "n"), "Set the maximum number of iterations for the solver.")
        .def("set_tolerance",  &Solver::setTolerance, bp::args("self", "tol"), "Set the solver's target tolerance.")
        ;
    }
  } // namespace python
} // namespace lienlp
