#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"

#include "lienlp/functor-base.hpp"


namespace lienlp
{

  /** @brief    Base class for differentiable cost functions.
   *  @remark   Cost functions derive from differentiable functors,
   *            and implement the DifferentiableFunctor<Scalar> API.
   *            As such, they can be used as constraints and composed.
   */
  template<typename _Scalar>
  struct CostFunctionBase : public DifferentiableFunctor<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)
    using Base = DifferentiableFunctor<Scalar>;

    CostFunctionBase(const int nx, const int ndx) : Base(nx, ndx, 1) {}

    /* Define cost function-specific API */

    /// @brief Evaluate the cost function.
    virtual Scalar call(const ConstVectorRef& x) const = 0;
    virtual void computeGradient(const ConstVectorRef& x, VectorRef out) const = 0;
    virtual void computeHessian(const ConstVectorRef& x, MatrixRef out) const = 0;

    VectorXs computeGradient(const ConstVectorRef& x) const
    {
      VectorXs out(this->ndx());
      computeGradient(x, out);
      return out;
    }

    MatrixXs computeHessian(const ConstVectorRef& x) const
    {
      MatrixXs out(this->ndx(), this->ndx());
      computeHessian(x, out);
      return out;
    }

    /* Defer parent class funcs to cost specific stuff */

    ReturnType operator()(const ConstVectorRef& x) const
    {
      ReturnType out(1, 1);
      out << call(x);
      return out;
    }

    void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
    {
      computeGradient(x, Jout.transpose());
    }

    void vectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef&, Eigen::Ref<JacobianType> Hout) const
    {
      computeHessian(x, Hout);
    }

    virtual ~CostFunctionBase<Scalar>() = default;

  };

}  // namespace lienlp
  
