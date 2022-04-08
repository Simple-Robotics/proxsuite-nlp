
#include "lienlp/python/fwd.hpp"
#include "lienlp/cost-function.hpp"

#include "lienlp/modelling/costs/quadratic-residual.hpp"
#include "lienlp/modelling/costs/squared-distance.hpp"


namespace lienlp
{
namespace python
{
  namespace internal
  {
    
    struct CostWrapper : context::Cost_t, bp::wrapper<context::Cost_t>
    {
      LIENLP_FUNCTOR_TYPEDEFS(context::Scalar)

      CostWrapper(const int nx, const int ndx) : context::Cost_t(nx, ndx) {}

      context::Scalar call(const ConstVectorRef& x) const { return get_override("call")(x); }
      void computeGradient(const ConstVectorRef& x, VectorRef out) const { get_override("computeGradient")(x, out); }
      void computeHessian (const ConstVectorRef& x, MatrixRef out) const { get_override("computeHessian") (x, out); }

    };
  } // namespace internal
  

  void exposeCost()
  {
    using context::Cost_t;
    using context::VectorXs;
    using context::MatrixXs;
    using context::VectorRef;
    using context::ConstVectorRef;
    using context::MatrixRef;
    using context::ConstMatrixRef;
    using context::Manifold;

    void(Cost_t::*compGrad1)(const ConstVectorRef&, VectorRef) const = &Cost_t::computeGradient;
    void(Cost_t::*compHess1)(const ConstVectorRef&, MatrixRef) const = &Cost_t::computeHessian;
    VectorXs(Cost_t::*compGrad2)(const ConstVectorRef&) const = &Cost_t::computeGradient;

    bp::class_<internal::CostWrapper, bp::bases<context::C2Function_t>, boost::noncopyable>(
      "CostFunctionBase", bp::init<int,int>()
    )
      .def("call", bp::pure_virtual(&Cost_t::call), bp::args("self", "x"))
      .def("computeGradient", bp::pure_virtual(compGrad1), bp::args("self", "x", "gout"))
      .def("computeGradient", compGrad2, bp::args("self", "x"))
      .def("computeHessian",  bp::pure_virtual(compHess1), bp::args("self", "x", "Hout"))
      ;

    bp::class_<func_to_cost<context::Scalar>, bp::bases<Cost_t>>(
      "CostFromFunction",
      "Wrap a scalar-values C2 function into a cost function.",
      bp::init<const context::C2Function_t&>(bp::args("self", "func"))
    )
      ;

    bp::class_<QuadraticResidualCost<context::Scalar>, bp::bases<Cost_t>>(
      "QuadraticResidualCost", "Quadratic of a residual function",
      bp::init<shared_ptr<context::C2Function_t>,
               const ConstMatrixRef&,
               const ConstVectorRef&,
               context::Scalar>(
                 (bp::arg("residual"),
                  bp::arg("weights"),
                  bp::arg("slope"),
                  bp::arg("constant") = 0.)
               )
    );

    bp::class_<QuadraticDistanceCost<context::Scalar>, bp::bases<Cost_t>>(
      "QuadraticDistanceCost", "Quadratic distance cost on the manifold.",
      bp::init<const Manifold&, const VectorXs&, const MatrixXs&>(
        bp::args("space", "target", "weights"))
    )
      .def(bp::init<const Manifold&, const VectorXs&>(
        bp::args("space", "target")))
      .def("update_target", &QuadraticDistanceCost<context::Scalar>::updateTarget, bp::args("new_target"))
    ;
  }

} // namespace python
} // namespace lienlp

