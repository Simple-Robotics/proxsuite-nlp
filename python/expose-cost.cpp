
#include "lienlp/python/fwd.hpp"
#include "lienlp/cost-function.hpp"

#include "lienlp/modelling/costs/squared-residual.hpp"


namespace lienlp
{
namespace python
{

  void exposeCost()
  {
    using context::Cost_t;
    using context::VectorXs;
    using context::MatrixXs;

    VectorXs (Cost_t::*compGrad1)(const context::ConstVectorRef&) const = &Cost_t::computeGradient;
    MatrixXs (Cost_t::*compHess1)(const context::ConstVectorRef&) const = &Cost_t::computeHessian;

    bp::class_<Cost_t, boost::noncopyable>(
      "CostFunctionBase", bp::no_init
    )
      .def("__call__", &Cost_t::operator(), bp::args("self", "x"))
      .def("computeGradient", compGrad1, bp::args("self", "x"))
      .def("computeHessian", compHess1, bp::args("self", "x"))
      ;

    bp::class_<QuadraticResidualCost<context::Scalar>, bp::bases<Cost_t>>(
      "QuadraticResidualCost", "Quadratic of a residual function",
      bp::init<shared_ptr<context::Residual_t>, MatrixXs>()
    );
  }  

} // namespace python
} // namespace lienlp

