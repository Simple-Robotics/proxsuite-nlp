#include "lienlp/python/fwd.hpp"
#include "lienlp/functor-base.hpp"


namespace lienlp
{
namespace python
{
  namespace bp = boost::python;

  void exposeFunctorTypes()
  {
    using context::Functor_t;
    using context::DFunctor_t;

    bp::class_<Functor_t, boost::noncopyable>("BaseFunctor", "Base class for functors.", bp::no_init)
      .def("__call__", &Functor_t::operator(), bp::args("self", "z"))
      ;

    context::MatFunc_t DFunctor_t::*compJac1 = &DFunctor_t::computeJacobian;
    context::MatFuncRet_t DFunctor_t::*compJac2 = &DFunctor_t::computeJacobian;
    context::VHPFunc_t DFunctor_t::*compHess1 = &DFunctor_t::vectorHessianProduct;

    bp::class_<DFunctor_t,
               bp::bases<Functor_t>,
               boost::noncopyable
               >("DifferentiableFunctor", "Base class for differentiable functors.", bp::no_init)
      .def("computeJacobian", compJac1, bp::args("self", "x", "Jout"))
      .def("computeJacobian", compJac2, bp::args("self", "x"))
      .def("vectorHessianProduct", compHess1, bp::args("self", "x", "v", "Hout"))
      .add_property("nx", &DFunctor_t::nx, "Input dimension")
      .add_property("ndx", &DFunctor_t::ndx, "Input tangent space dimension.")
      .add_property("nr", &DFunctor_t::nr, "Function codimension.")
      ;
  }

} // namespace python
} // namespace lienlp
