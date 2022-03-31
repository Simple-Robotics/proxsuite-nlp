
#include "lienlp/python/fwd.hpp"
#include "lienlp/cost-function.hpp"

#include "lienlp/modelling/costs/quadratic-residual.hpp"
#include "lienlp/modelling/costs/squared-distance.hpp"


namespace lienlp
{
namespace python
{

  void exposeCost()
  {
    using context::Cost_t;
    using context::VectorXs;
    using context::MatrixXs;
    using context::ConstMatrixRef;
    using context::ConstVectorRef;
    using context::ManifoldType;

    VectorXs (Cost_t::*compGrad1)(const ConstVectorRef&) const = &Cost_t::computeGradient;
    MatrixXs (Cost_t::*compHess1)(const ConstVectorRef&) const = &Cost_t::computeHessian;

    bp::class_<Cost_t, shared_ptr<Cost_t>, bp::bases<context::DFunctor_t>, boost::noncopyable>(
      "CostFunctionBase", bp::no_init
    )
      .def("__call__", &Cost_t::call, bp::args("self", "x"))
      .def("computeGradient", compGrad1, bp::args("self", "x"))
      .def("computeHessian", compHess1, bp::args("self", "x"))
      ;

    bp::class_<QuadraticResidualCost<context::Scalar>, bp::bases<Cost_t>>(
      "QuadraticResidualCost", "Quadratic of a residual function",
      bp::init<shared_ptr<context::DFunctor_t>,
               const ConstMatrixRef&,
               const ConstVectorRef&,
               context::Scalar>(
                 (bp::arg("residual"),
                  bp::arg("weights"),
                  bp::arg("slope"),
                  bp::arg("constant") = 0.)
               )
    );

    bp::class_<QuadDistanceCost<context::Scalar>, bp::bases<Cost_t>>(
      "QuadDistanceCost", "Quadratic distance cost on the manifold.",
      bp::init<const ManifoldType&, const VectorXs&, const MatrixXs&>(
        bp::args("space", "target", "weights"))
    )
      .def(bp::init<const ManifoldType&, const VectorXs&>(
        bp::args("space", "target")))
      .def("update_target", &QuadDistanceCost<context::Scalar>::updateTarget, bp::args("new_target"))
    ;
  }

} // namespace python
} // namespace lienlp

