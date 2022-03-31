#include "lienlp/python/fwd.hpp"
#include "lienlp/function-base.hpp"


namespace lienlp
{
namespace python
{

  void exposeFunctionTypes()
  {
    using context::Function_t;
    using context::C2Function_t;

    bp::class_<Function_t, boost::noncopyable>("BaseFunction", "Base class for functions.", bp::no_init)
      .def("__call__", &Function_t::operator(), bp::args("self", "z"))
      ;

    context::MatFunc_t C2Function_t::*compJac1 = &C2Function_t::computeJacobian;
    context::MatFuncRet_t C2Function_t::*compJac2 = &C2Function_t::computeJacobian;
    context::VHPFunc_t C2Function_t::*compHess1 = &C2Function_t::vectorHessianProduct;

    bp::class_<C2Function_t,
               bp::bases<Function_t>,
               boost::noncopyable
               >("C2Function", "Base class for twice-differentiable functions.", bp::no_init)
      .def("computeJacobian", compJac1, bp::args("self", "x", "Jout"))
      .def("computeJacobian", compJac2, bp::args("self", "x"))
      .def("vectorHessianProduct", compHess1, bp::args("self", "x", "v", "Hout"))
      .add_property("nx", &C2Function_t::nx, "Input dimension")
      .add_property("ndx", &C2Function_t::ndx, "Input tangent space dimension.")
      .add_property("nr", &C2Function_t::nr, "Function codimension.")
      ;
  }

} // namespace python
} // namespace lienlp
