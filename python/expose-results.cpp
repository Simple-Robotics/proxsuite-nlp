#include "proxnlp/python/fwd.hpp"
#include "proxnlp/results.hpp"

namespace proxnlp {
namespace python {

void exposeResults() {
  using context::Results;

  bp::enum_<ConvergenceFlag>("ConvergenceFlag", "Convergence flag enum.")
      .value("success", ConvergenceFlag::SUCCESS)
      .value("max_iters_reached", ConvergenceFlag::MAX_ITERS_REACHED);

  bp::class_<Results>("Results", "Results holder struct.",
                      bp::init<context::Problem &>(bp::args("self", "problem")))
      .def_readonly("converged", &Results::converged)
      .def_readonly("merit", &Results::merit, "Merit function value.")
      .def_readonly("value", &Results::value)
      .def_readonly("xopt", &Results::x_opt)
      .def_readonly("data_lamsopt", &Results::data_lams_opt)
      .def_readonly("lamsopt", &Results::lams_opt)
      .def_readonly("activeset", &Results::active_set)
      .def_readonly("num_iters", &Results::num_iters)
      .def_readonly("mu", &Results::mu)
      .def_readonly("rho", &Results::rho)
      .def_readonly("dual_infeas", &Results::dual_infeas)
      .def_readonly("prim_infeas", &Results::prim_infeas)
      .def_readonly("constraint_errs", &Results::constraint_violations,
                    "Constraint violations.")
      .def(bp::self_ns::str(bp::self));
}

} // namespace python
} // namespace proxnlp
