#include "proxnlp/python/fwd.hpp"
#include "proxnlp/function-base.hpp"


namespace proxnlp
{
namespace python
{
  namespace internal
  {
    struct FunctionWrap : context::Function, bp::wrapper<context::Function>
    {
    public:
      PROXNLP_FUNCTION_TYPEDEFS(context::Scalar)

      FunctionWrap(const int nx, const int ndx, const int nr) : context::Function(nx, ndx, nr) {}

      ReturnType operator()(const ConstVectorRef& x) const
      {
        bp::override f = get_override("__call__");
        return f(x);
      }
    };

    struct C1FunctionWrap : context::C1Function, bp::wrapper<context::C1Function>
    {
      PROXNLP_FUNCTION_TYPEDEFS(context::Scalar)

      C1FunctionWrap(const int nx, const int ndx, const int nr) : context::C1Function(nx, ndx, nr) {}

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

    struct C2FunctionWrap : context::C2Function, bp::wrapper<context::C2Function>
    {
      PROXNLP_FUNCTION_TYPEDEFS(context::Scalar)

      C2FunctionWrap(const int nx, const int ndx, const int nr) : context::C2Function(nx, ndx, nr) {}

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
          return context::C2Function::vectorHessianProduct(x, v, Hout);
        }
      }

      void default_vhp(const ConstVectorRef& x, const ConstVectorRef& v, Eigen::Ref<JacobianType> Hout) const
      {
        return context::C2Function::vectorHessianProduct(x, v, Hout);
      }
    };

  } // namespace internal
  

  void exposeFunctionTypes()
  {
    using context::Function;
    using context::C1Function;
    using context::C2Function;

    bp::class_<internal::FunctionWrap,
               boost::noncopyable
               >("BaseFunction", "Base class for functions.",
                 bp::init<int, int, int>(bp::args("self", "nx", "ndx", "nr")))
      .def("__call__", bp::pure_virtual(&Function::operator()), bp::args("self", "x"), "Call the function.")
      .add_property("nx", &Function::nx, "Input dimension")
      .add_property("ndx",&Function::ndx,"Input tangent space dimension.")
      .add_property("nr", &Function::nr, "Function codimension.")
      ;

    context::MatFuncType C1Function::*compJac1 = &C1Function::computeJacobian;
    context::MatFuncRetType C1Function::*compJac2 = &C1Function::computeJacobian;

    bp::class_<internal::C1FunctionWrap,
               bp::bases<Function>,
               boost::noncopyable
               >("C1Function", "Base class for differentiable functions", bp::init<int, int, int>())
      .def("computeJacobian", bp::pure_virtual(compJac1), bp::args("self", "x", "Jout"))
      .def("get_jacobian", compJac2, bp::args("self", "x"), "Compute and return Jacobian.")
      ;

    context::VHPFuncType C2Function::*compHess1 = &C2Function::vectorHessianProduct;
    context::VHPFuncRetType C2Function::*compHess2 = &C2Function::vectorHessianProduct;

    bp::class_<internal::C2FunctionWrap,
               bp::bases<C1Function>,
               boost::noncopyable
               >("C2Function", "Base class for twice-differentiable functions.", bp::init<int,int,int>())
      .def("vectorHessianProduct", compHess1, &internal::C2FunctionWrap::default_vhp, bp::args("self", "x", "v", "Hout"))
      .def("get_vhp", compHess2, bp::args("self", "x", "v"), "Compute and return the vector-Hessian product.")
      ;
  }

} // namespace python
} // namespace proxnlp
