/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/manifold-base.hpp"
#include "proxsuite-nlp/function-base.hpp"

#include <boost/core/demangle.hpp>
#include <ostream>

namespace proxsuite {
namespace nlp {

template <typename Scalar>
auto downcast_function_to_cost(const shared_ptr<C2FunctionTpl<Scalar>> &func)
    -> shared_ptr<CostFunctionBaseTpl<Scalar>> {
  if (func->nr() != 1) {
    PROXSUITE_NLP_RUNTIME_ERROR(
        "Function cannot be cast to cost (codimension nr != 1).");
  }
  return std::make_shared<func_to_cost<Scalar>>(func);
}

/** @brief    Base class for differentiable cost functions.
 *  @remark   Cost functions derive from differentiable functions,
 *            and implement the C2FunctionTpl<Scalar> API.
 *            As such, they can be used as constraints and composed.
 */
template <typename _Scalar>
struct CostFunctionBaseTpl : public C2FunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = C2FunctionTpl<Scalar>;

  CostFunctionBaseTpl(const int nx, const int ndx) : Base(nx, ndx, 1) {}
  explicit CostFunctionBaseTpl(const ManifoldAbstractTpl<Scalar> &manifold)
      : Base(manifold, 1) {}

  /* Define cost function-specific API */

  /// @brief Evaluate the cost function.
  virtual Scalar call(const ConstVectorRef &x) const = 0;
  virtual void computeGradient(const ConstVectorRef &x,
                               VectorRef out) const = 0;
  virtual void computeHessian(const ConstVectorRef &x, MatrixRef out) const = 0;

  /* Allocated versions */

  VectorXs computeGradient(const ConstVectorRef &x) const {
    VectorXs out(this->ndx());
    computeGradient(x, out);
    return out;
  }

  MatrixXs computeHessian(const ConstVectorRef &x) const {
    MatrixXs out(this->ndx(), this->ndx());
    computeHessian(x, out);
    return out;
  }

  /* Implement C2FunctionTpl interface. */

  VectorXs operator()(const ConstVectorRef &x) const {
    VectorXs out(1, 1);
    out << call(x);
    return out;
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    Eigen::Matrix<Scalar, 1, -1> gT = Jout.template topRows<1>();
    computeGradient(x, gT.transpose());
    Jout.row(0) = gT;
  }

  void vectorHessianProduct(const ConstVectorRef &x, const ConstVectorRef &v,
                            MatrixRef Hout) const {
    computeHessian(x, Hout);
    Hout *= v(0);
  }

  virtual ~CostFunctionBaseTpl() = default;

  friend std::ostream &operator<<(std::ostream &ostr,
                                  const CostFunctionBaseTpl<Scalar> &cost) {
    const std::string name = boost::core::demangle(typeid(cost).name());
    ostr << name;
    return ostr;
  }
};

template <typename _Scalar> struct func_to_cost : CostFunctionBaseTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostFunctionBaseTpl<Scalar>;
  using C2Function = C2FunctionTpl<Scalar>;

  /** @brief    Constructor.
   *  @details  This defines an implicit conversion from the C2FunctionTpl type.
   */
  func_to_cost(const shared_ptr<C2Function> &func)
      : Base(func->nx(), func->ndx()), underlying_(func) {}

  Scalar call(const ConstVectorRef &x) const { return underlying()(x)(0); }

  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    underlying().computeJacobian(x, out.transpose());
  }

  void computeHessian(const ConstVectorRef &x, MatrixRef Hout) const {
    VectorXs v = VectorXs::Ones(1);
    underlying().vectorHessianProduct(x, v, Hout);
  }

private:
  shared_ptr<C2Function> underlying_;
  const C2Function &underlying() const { return *underlying_; }
};

} // namespace nlp
} // namespace proxsuite

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/cost-function.txx"
#endif
