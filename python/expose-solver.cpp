#include "lienlp/python/fwd.hpp"

#include "lienlp/solver-base.hpp"


namespace lienlp
{
  namespace python
  {

    void exposeSolver()
    {
      using context::Scalar;
      using context::ManifoldType;
      using Solver_t = Solver<Scalar>;

      bp::class_<Solver_t>(
        "Solver", "The numerical solver.",
        bp::init<const ManifoldType&,
                 shared_ptr<context::Problem_t>&
                 >((  bp::arg("self")
                    , bp::arg("space")
                    , bp::arg("problem")
                    // , bp::arg("tol") = 1e-6
                    // , bp::arg("mu_init") = 1e-2
                    // , bp::arg("rho") = 0.
                    // , bp::arg("mu_factor") = 0.1
                    ))
      )
        // .def_readwrite("use_gauss_newton", &Solver_t::use_gauss_newton, "Whether to use a Gauss-Newton Hessian matrix approximation.")
        // .def_readwrite("alpha_min", &Solver_t::alpha_min, "Set linesearch minimum step size.")
        // .def_readwrite("ls_c1", &Solver_t::ls_c1, "Linesearch Armijo rule decrease ratio.")
        .def_readwrite("verbose", &Solver_t::verbose, "Solver verbose setting.")
        .def("solve", &Solver_t::solve,
             bp::args("workspace", "results", "x0", "lams0"))
        .def("set_penalty",    &Solver_t::setPenalty,   bp::args("self", "mu"), "Set the augmented Lagrangian penalty parameter.")
        .def("set_prox_param", &Solver_t::setProxParam, bp::args("self", "rho"), "Set the primal proximal penalty parameter.")
        .def("set_maxiters",   &Solver_t::setMaxIters,  bp::args("self", "n"), "Set the maximum number of iterations for the solver.")
        .def("set_tolerance",  &Solver_t::setTolerance, bp::args("self", "tol"), "Set the solver's target tolerance.")
        ;
    }
  } // namespace python
} // namespace lienlp
