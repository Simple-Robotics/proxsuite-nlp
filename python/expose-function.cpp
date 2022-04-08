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

      ReturnType operator()(const ConstVectorRef& x) const
      {
        bp::override f = get_override("__call__");
        return f(x);
      }
    };

    struct C1FunctionWrap : context::C1Function_t, bp::wrapper<context::C1Function_t>
    {
      LIENLP_FUNCTOR_TYPEDEFS(context::Scalar)

      C1FunctionWrap(const int nx, const int ndx, const int nr) : context::C1Function_t(nx, ndx, nr) {}

      ReturnType operator()(const ConstVectorRef& x) const
      {
        bp::override f = get_override("__call__");
        return f(x);
      }

      void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
      {
        Jout.resize(this->nr(), this->ndx());
        get_override("computeJacobian")(x, Jout);
      }
    };

    struct C2FunctionWrap : context::C2Function_t, bp::wrapper<context::C2Function_t>
    {
      LIENLP_FUNCTOR_TYPEDEFS(context::Scalar)

      C2FunctionWrap(const int nx, const int ndx, const int nr) : context::C2Function_t(nx, ndx, nr) {}

      ReturnType operator()(const ConstVectorRef& x) const
      {
        bp::override f = get_override("__call__");
        return f(x);
      }

      void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
      {
        Jout.resize(this->nr(), this->ndx());
        get_override("computeJacobian")(x, Jout);
      }

      void vectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef& v, Eigen::Ref<JacobianType> Hout) const
      {
        Hout.resize(this->ndx(), this->ndx());
        if (bp::override f = this->get_override("vectorHessianProduct"))
        {
          f(x, v, Hout);
          return;
        } else {
          return context::C2Function_t::vectorHessianProduct(x, v, Hout);
        }
      }

      void default_vhp(const ConstVectorRef& x, const ConstVectorRef& v, Eigen::Ref<JacobianType> Hout) const
      {
        return context::C2Function_t::vectorHessianProduct(x, v, Hout);
      }
    };

  } // namespace internal
  

  void exposeFunctionTypes()
  {
    using context::Function_t;
    using context::C1Function_t;
    using context::C2Function_t;

    bp::class_<internal::FunctionWrap,
               boost::noncopyable
               >("BaseFunction", "Base class for functions.", bp::init<int, int, int>())
      .def("__call__", bp::pure_virtual(&Function_t::operator()), bp::args("self", "z"), "Call the function.")
      .add_property("nx", &Function_t::nx, "Input dimension")
      .add_property("ndx",&Function_t::ndx,"Input tangent space dimension.")
      .add_property("nr", &Function_t::nr, "Function codimension.")
      ;

    context::MatFunc_t C1Function_t::*compJac1 = &C1Function_t::computeJacobian;
    context::MatFuncRet_t C1Function_t::*compJac2 = &C1Function_t::computeJacobian;

    bp::class_<internal::C1FunctionWrap,
               bp::bases<Function_t>,
               boost::noncopyable
               >("C1Function", "Base class for differentiable functions", bp::init<int, int, int>())
      .def("computeJacobian", bp::pure_virtual(compJac1), bp::args("self", "x", "Jout"))
      .def("get_jacobian", compJac2, bp::args("self", "x"), "Compute and return Jacobian.")
      ;

    context::VHPFunc_t C2Function_t::*compHess1 = &C2Function_t::vectorHessianProduct;
    context::VHPFuncRet_t C2Function_t::*compHess2 = &C2Function_t::vectorHessianProduct;

    bp::class_<internal::C2FunctionWrap,
               bp::bases<C1Function_t>,
               boost::noncopyable
               >("C2Function", "Base class for twice-differentiable functions.", bp::init<int,int,int>())
      .def("vectorHessianProduct", compHess1, &internal::C2FunctionWrap::default_vhp, bp::args("self", "x", "v", "Hout"))
      .def("get_vhp", compHess2, bp::args("self", "x", "v"), "Compute and return the vector-Hessian product.")
      ;
  }

} // namespace python
} // namespace lienlp
