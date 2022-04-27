
#include "proxnlp/python/fwd.hpp"
#include "proxnlp/cost-function.hpp"
#include "proxnlp/cost-sum.hpp"

#include "proxnlp/modelling/costs/quadratic-residual.hpp"
#include "proxnlp/modelling/costs/squared-distance.hpp"

#include "boost/python/operators.hpp"

namespace proxnlp
{
namespace python
{
  namespace internal
  {
    
    struct CostWrapper : context::Cost, bp::wrapper<context::Cost>
    {
      PROXNLP_FUNCTION_TYPEDEFS(context::Scalar)

      CostWrapper(const int nx, const int ndx) : context::Cost(nx, ndx) {}

      context::Scalar call(const ConstVectorRef& x) const { return get_override("call")(x); }
      void computeGradient(const ConstVectorRef& x, VectorRef out) const { get_override("computeGradient")(x, out); }
      void computeHessian (const ConstVectorRef& x, MatrixRef out) const { get_override("computeHessian") (x, out); }

    };
  } // namespace internal
  

  void exposeCost()
  {
    using context::Cost;
    using context::VectorXs;
    using context::MatrixXs;
    using context::VectorRef;
    using context::ConstVectorRef;
    using context::MatrixRef;
    using context::ConstMatrixRef;
    using context::Manifold;

    void(Cost::*compGrad1)(const ConstVectorRef&, VectorRef) const = &Cost::computeGradient;
    void(Cost::*compHess1)(const ConstVectorRef&, MatrixRef) const = &Cost::computeHessian;
    VectorXs(Cost::*compGrad2)(const ConstVectorRef&) const = &Cost::computeGradient;
    MatrixXs(Cost::*compHess2)(const ConstVectorRef&) const = &Cost::computeHessian;

    bp::class_<internal::CostWrapper, bp::bases<context::C2Function>, boost::noncopyable>(
      "CostFunctionBase", bp::init<int,int>()
    )
      .def("call", bp::pure_virtual(&Cost::call), bp::args("self", "x"))
      .def("computeGradient", bp::pure_virtual(compGrad1), bp::args("self", "x", "gout"))
      .def("computeGradient", compGrad2, bp::args("self", "x"), "Compute and return the gradient.")
      .def("computeHessian",  bp::pure_virtual(compHess1), bp::args("self", "x", "Hout"))
      .def("computeHessian",  compHess2, bp::args("self", "x"), "Compute and return the Hessian.")
      // define non-member operators
      .def(bp::self + bp::self)   // see cost_sum.hpp / returns CostSum<Scalar>
      .def(context::Scalar() * bp::self)   // see cost_sum.hpp / returns CostSum<Scalar>
      ;

    bp::class_<func_to_cost<context::Scalar>, bp::bases<Cost>>(
      "CostFromFunction",
      "Wrap a scalar-values C2 function into a cost function.",
      bp::init<const context::C2Function&>(bp::args("self", "func"))
    )
      ;

    using CostSum_t = CostSum<context::Scalar>;
    bp::class_<CostSum_t, bp::bases<Cost>>(
      "CostSum",
      "Sum of cost functions.",
      bp::init<int,
               int,
               const std::vector<CostSum_t::BasePtr>&,
               const std::vector<context::Scalar>&
               >(bp::args("nx", "ndx", "components", "weights"))
    )
      .def(bp::init<int, int>(bp::args("nx", "ndx")))
      .add_property("num_components", &CostSum_t::numComponents, "Number of components.")
      .def_readonly("weights", &CostSum_t::m_weights)
      .def("add_component", &CostSum_t::addComponent,
           ((bp::arg("self"), bp::arg("comp"),
             bp::arg("w") = 1.)),
           "Add a component to the cost."
           )
      // expose inplace operators
      .def(bp::self += bp::self)
      .def(bp::self += internal::CostWrapper(0, 0))  // declval doesn't work in context, use non-abstract wrapper
      .def(bp::self *= context::Scalar())
      // expose operator overloads 
      .def(bp::self + bp::self)
      .def(context::Scalar() * bp::self)
      .def(bp::self + internal::CostWrapper(0, 0))
      // printing
      .def(bp::self_ns::str(bp::self))
      ;

    /* Expose specific cost functions */

    bp::class_<QuadraticResidualCost<context::Scalar>, bp::bases<Cost>>(
      "QuadraticResidualCost", "Quadratic of a residual function",
      bp::init<const shared_ptr<context::C2Function>&,
               const ConstMatrixRef&,
               const ConstVectorRef&,
               context::Scalar>(
                 (bp::arg("self"),
                  bp::arg("residual"),
                  bp::arg("weights"),
                  bp::arg("slope"),
                  bp::arg("constant") = 0.)
               )
    )
      .def(bp::init<const shared_ptr<context::C2Function>&,
                    const ConstMatrixRef&,
                    context::Scalar>(
           (bp::arg("self"),
            bp::arg("residual"),
            bp::arg("weights"),
            bp::arg("constant") = 0.))
            )
    ;

    bp::class_<QuadraticDistanceCost<context::Scalar>, bp::bases<Cost>>(
      "QuadraticDistanceCost", "Quadratic distance cost `(1/2)r.T * Q * r + b.T * r + c` on the manifold.",
      bp::init<const Manifold&, const VectorXs&, const MatrixXs&>(
        bp::args("space", "target", "weights"))
    )
      .def(bp::init<const Manifold&, const VectorXs&>(
        bp::args("space", "target")))
      .def("update_target", &QuadraticDistanceCost<context::Scalar>::updateTarget, bp::args("new_target"))
    ;
  }

} // namespace python
} // namespace proxnlp

