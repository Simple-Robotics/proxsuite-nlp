#include "proxnlp/python/fwd.hpp"
#include "proxnlp/results.hpp"


namespace proxnlp
{
namespace python
{

  void exposeResults()
  {
    using context::Results;

    bp::enum_<ConvergenceFlag>("ConvergenceFlag", "Convergence flag enum.")
      .value("uninit", ConvergenceFlag::UNINIT)
      .value("success", ConvergenceFlag::SUCCESS)
      .value("max_iters_reached", ConvergenceFlag::MAX_ITERS_REACHED)
      ;

    bp::class_<Results>(
      "Results", "Results holder struct.",
      bp::init<int, context::Problem&>(bp::args("self", "nx", "problem")))
      .def_readonly("converged", &Results::converged)
      .def_readonly("merit", &Results::merit, "Merit function value.")
      .def_readonly("value", &Results::value)
      .def_readonly("xopt", &Results::xOpt)
      .def_readonly("lamsopt", &Results::lamsOpt)
      .def_readonly("activeset", &Results::activeSet)
      .def_readonly("numiters", &Results::numIters)
      .def_readonly("mu", &Results::mu)
      .def_readonly("rho", &Results::rho)
      .def_readonly("dual_infeas", &Results::dualInfeas)
      .def_readonly("primal_infeas", &Results::primalInfeas)
      .def(bp::self_ns::str(bp::self))
      ;
  }  

} // namespace python
} // namespace proxnlp

