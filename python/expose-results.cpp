#include "lienlp/python/fwd.hpp"
#include "lienlp/results.hpp"


namespace lienlp
{
namespace python
{

  void exposeResults()
  {
    using context::Result;

    bp::enum_<ConvergenceFlag>("ConvergenceFlag", "Convergence flag enum.")
      .value("uninit", ConvergenceFlag::UNINIT)
      .value("success", ConvergenceFlag::SUCCESS)
      .value("max_iters_reached", ConvergenceFlag::MAX_ITERS_REACHED)
      ;

    bp::class_<Result>(
      "Results", "Results holder struct.",
      bp::init<int, context::Problem&>(bp::args("self", "nx", "problem")))
      .def_readonly("converged", &Result::converged)
      .def_readonly("value", &Result::value)
      .def_readonly("xopt", &Result::xOpt)
      .def_readonly("lamsopt", &Result::lamsOpt)
      .def_readonly("activeset", &Result::activeSet)
      .def_readonly("numiters", &Result::numIters)
      .def_readonly("mu", &Result::mu)
      .def_readonly("rho", &Result::rho)
      ;
  }  

} // namespace python
} // namespace lienlp

