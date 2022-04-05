#include "lienlp/python/fwd.hpp"
#include "lienlp/function-base.hpp"


namespace lienlp
{
namespace python
{
  namespace internal
  {
    struct FunctionWrap : context::Function_t, bp::wrapper<context::Function_t>
    {
    public:
      LIENLP_FUNCTOR_TYPEDEFS(context::Scalar)

      FunctionWrap(const int nx, const int ndx, const int nr) : context::Function_t(nx, ndx, nr) {}

      ReturnType operator()(const ConstVectorRef& x) const override
      {
        return get_override("operator()")(x);
      }
    };

    struct C1FunctionWrap : context::C1Function_t, bp::wrapper<context::C1Function_t>
    {
      LIENLP_FUNCTOR_TYPEDEFS(context::Scalar)

      void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
      {
        get_override("computeJacobian")(x, Jout);
      }
    };

    struct C2FunctionWrap : context::C2Function_t, bp::wrapper<context::C2Function_t>
    {
      LIENLP_FUNCTOR_TYPEDEFS(context::Scalar)

      void computeVectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef& v, Eigen::Ref<JacobianType> Hout) const
      {
        get_override("vectorHessianProduct")(x, v, Hout);
      }
    };

  } // namespace internal
  

  void exposeFunctionTypes()
  {
    using context::Function_t;
    using context::C1Function_t;
    using context::C2Function_t;

    bp::class_<internal::FunctionWrap, boost::noncopyable>("BaseFunction", "Base class for functions.", bp::init<int, int, int>())
      .def("__call__", bp::pure_virtual(&Function_t::operator()), bp::args("self", "z"), "Call the function.")
      .add_property("nx", &Function_t::nx, "Input dimension")
      .add_property("ndx", &Function_t::ndx, "Input tangent space dimension.")
      .add_property("nr", &Function_t::nr, "Function codimension.")
      ;

    context::MatFunc_t C1Function_t::*compJac1 = &C1Function_t::computeJacobian;
    context::MatFuncRet_t C1Function_t::*compJac2 = &C1Function_t::computeJacobian;

    bp::class_<internal::C1FunctionWrap,
               bp::bases<Function_t>,
               boost::noncopyable>("C1Function", "Base class for differentiable functions", bp::no_init)
      .def("computeJacobian", bp::pure_virtual(compJac1), bp::args("self", "x", "Jout"))
      .def("computeJacobian", compJac2, bp::args("self", "x"))
      .def("to_base", &C1Function_t::toBase, "Downcast to the base function type.",
           bp::return_internal_reference<>())
      ;

    context::VHPFunc_t C2Function_t::*compHess1 = &C2Function_t::vectorHessianProduct;

    bp::class_<internal::C2FunctionWrap,
               bp::bases<C1Function_t>,
               boost::noncopyable
               >("C2Function", "Base class for twice-differentiable functions.", bp::no_init)
      .def("vectorHessianProduct", bp::pure_virtual(compHess1), bp::args("self", "x", "v", "Hout"))
      .def("to_c1_function", &C2Function_t::toC1, "Downcast to the C1 function type.",
           bp::return_internal_reference<>())
      ;
  }

} // namespace python
} // namespace lienlp
