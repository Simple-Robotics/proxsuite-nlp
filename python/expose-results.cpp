#include "lienlp/python/fwd.hpp"
#include "lienlp/results.hpp"


namespace lienlp
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
      .def_readonly("value", &Results::value)
      .def_readonly("xopt", &Results::xOpt)
      .def_readonly("lamsopt", &Results::lamsOpt)
      .def_readonly("activeset", &Results::activeSet)
      .def_readonly("numiters", &Results::numIters)
      .def_readonly("mu", &Results::mu)
      .def_readonly("rho", &Results::rho)
      ;
  }  

} // namespace python
} // namespace lienlp

