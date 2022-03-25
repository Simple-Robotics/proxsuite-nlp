#include "lienlp/python/fwd.hpp"
#include "lienlp/results.hpp"


namespace lienlp
{
namespace python
{

  void exposeResults()
  {
    using context::Scalar;
    using context::Result_t;

    bp::enum_<ConvergedFlag>("ConvergedFlag", "Convergence flag enum.")
      .value("uninit", ConvergedFlag::UNINIT)
      .value("success", ConvergedFlag::SUCCESS)
      .value("too_many_iters", ConvergedFlag::TOO_MANY_ITERS)
      ;

    bp::class_<Result_t>(
      "Results", "Results holder struct.",
      bp::init<int, context::Problem_t&>())
      .def_readonly("converged", &Result_t::converged)
      .def_readonly("value", &Result_t::value)
      .def_readonly("xopt", &Result_t::xOpt)
      .def_readonly("lamsopt", &Result_t::lamsOpt)
      .def_readonly("activeset", &Result_t::activeSet)
      .def_readonly("numiters", &Result_t::numIters)
      .def_readonly("mu", &Result_t::mu)
      .def_readonly("rho", &Result_t::rho)
      ;
  }  

} // namespace python
} // namespace lienlp

