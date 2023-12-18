#include "proxnlp/python/fwd.hpp"

#include "proxnlp/function-base.hpp"

namespace proxnlp {
namespace python {

struct FunctionWrap : context::Function, bp::wrapper<context::Function> {
public:
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(context::Scalar);

  using context::Function::BaseFunctionTpl;

  VectorXs operator()(const ConstVectorRef &x) const {
    bp::override f = get_override("__call__");
    return f(x);
  }
};

struct C1FunctionWrap : context::C1Function, bp::wrapper<context::C1Function> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(context::Scalar);

  using context::C1Function::C1FunctionTpl;

  VectorXs operator()(const ConstVectorRef &x) const {
    bp::override f = get_override("__call__");
    return f(x);
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    Jout.resize(this->nr(), this->ndx());
    get_override("computeJacobian")(x, Jout);
  }
};

struct C2FunctionWrap : context::C2Function, bp::wrapper<context::C2Function> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(context::Scalar);

  using context::C2Function::C2FunctionTpl;

  VectorXs operator()(const ConstVectorRef &x) const {
    bp::override f = get_override("__call__");
    return f(x);
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    Jout.resize(this->nr(), this->ndx());
    get_override("computeJacobian")(x, Jout);
  }

  void vectorHessianProduct(const ConstVectorRef &x, const ConstVectorRef &v,
                            MatrixRef Hout) const {
    Hout.resize(this->ndx(), this->ndx());
    if (bp::override f = this->get_override("vectorHessianProduct")) {
      f(x, v, Hout);
    } else {
      context::C2Function::vectorHessianProduct(x, v, Hout);
    }
  }

  MatrixXs getVHP(const ConstVectorRef &x, const ConstVectorRef &v) const {
    using context::MatrixXs;
    MatrixXs Hout(this->ndx_, this->ndx_);
    this->vectorHessianProduct(x, v, Hout);
    return Hout;
  }

  void default_vhp(const ConstVectorRef &x, const ConstVectorRef &v,
                   MatrixRef Hout) const {
    context::C2Function::vectorHessianProduct(x, v, Hout);
  }
};

} // namespace python
} // namespace proxnlp
