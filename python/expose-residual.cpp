#include "lienlp/python/fwd.hpp"
#include "lienlp/residual-base.hpp"

#include "lienlp/modelling/residuals/linear.hpp"

#include <boost/python/overloads.hpp>


namespace lienlp
{
namespace python
{
  namespace bp = boost::python;
  using context::Residual_t;

  struct ResidualWrap : Residual_t, bp::wrapper<Residual_t>
  {
    void computeJacobian(
      const ConstVectorRef& x,
      Eigen::Ref<JacobianType> Jout) const
    {
      this->get_override("computeJacobian")(x, Jout);
    }
  };

  void exposeResidual()
  {
    using context::Scalar;
    using context::VectorXs;
    using context::MatrixXs;

    // define function pointer types and cast member functions
    context::MatFuncRet_t Residual_t::*compJac1 = &Residual_t::computeJacobian;
    context::MatFunc_t Residual_t::*compJac2 = &Residual_t::computeJacobian;

    bp::class_<Residual_t, boost::noncopyable>("ResidualBase", bp::no_init)
      .def("__call__", &Residual_t::operator(), bp::args("self", "z"))
      .def("computeJacobian", compJac1, bp::args("self", "x"))
      .def("computeJacobian", compJac2, bp::args("self", "x", "Jout"))
      // .def("vectorHessianProduct", &Residual_t::vectorHessianProduct, computeVhp1())
      .add_property("nx", &Residual_t::nx)
      .add_property("ndx", &Residual_t::ndx)
      .add_property("nr", &Residual_t::nr, "Residual codimension.")
      ;

    bp::class_<LinearResidual<Scalar>, bp::bases<Residual_t>>("LinearResidual",
      bp::init<MatrixXs, VectorXs>());
  }

} // namespace python
} // namespace lienlp
