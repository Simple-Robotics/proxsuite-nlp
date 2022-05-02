#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/function-base.hpp"

#include <boost/core/demangle.hpp>
#include <ostream>


namespace proxnlp
{
  template<typename Scalar>
  struct func_to_cost;

  /** @brief    Base class for differentiable cost functions.
   *  @remark   Cost functions derive from differentiable functions,
   *            and implement the C2FunctionTpl<Scalar> API.
   *            As such, they can be used as constraints and composed.
   */
  template<typename _Scalar>
  struct CostFunctionBaseTpl : public C2FunctionTpl<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)
    using Base = C2FunctionTpl<Scalar>;

    CostFunctionBaseTpl(const int nx, const int ndx) : Base(nx, ndx, 1) {}
    CostFunctionBaseTpl(const CostFunctionBaseTpl<Scalar>&) = default;

    /* Define cost function-specific API */

    /// @brief Evaluate the cost function.
    virtual Scalar call(const ConstVectorRef& x) const = 0;
    virtual void computeGradient(const ConstVectorRef& x, VectorRef out) const = 0;
    virtual void computeHessian (const ConstVectorRef& x, MatrixRef out) const = 0;

    /* Allocated versions */

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

    /* Implement C2FunctionTpl interface. */

    ReturnType operator()(const ConstVectorRef& x) const
    {
      ReturnType out(1, 1);
      out << call(x);
      return out;
    }

    void computeJacobian(const ConstVectorRef& x, MatrixRef Jout) const
    {
#if EIGEN_VERSION_AT_LEAST(3,3,8)
      computeGradient(x, Jout.transpose());
#else
      Eigen::Matrix<Scalar, 1, -1> gT = Jout.template topRows<1>();
      computeGradient(x, gT.transpose());
      Jout.row(0) = gT;
#endif
    }

    void vectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef& v, MatrixRef Hout) const
    {
      computeHessian(x, Hout);
      Hout *= v(0);
    }

    virtual ~CostFunctionBaseTpl<Scalar>() = default;

    /// @brief    Conversion from C2FunctionTpl.
    CostFunctionBaseTpl(const C2FunctionTpl<Scalar>& func)
      : CostFunctionBaseTpl<Scalar>(func_to_cost<Scalar>(func)) {}

    friend std::ostream& operator<<(std::ostream& ostr, const CostFunctionBaseTpl<Scalar>& cost)
    {
      const std::string name = boost::core::demangle(typeid(cost).name());
      ostr << name;
      return ostr;
    }
  };

  template<typename _Scalar>
  struct func_to_cost : CostFunctionBaseTpl<_Scalar>
  {
  private:
    const C2FunctionTpl<_Scalar>& underlying_;
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)

    /** @brief    Constructor.
     *  @details  This defines an implicit conversion from the C2FunctionTpl type.
     */
    func_to_cost(const C2FunctionTpl<Scalar>& func)
      : CostFunctionBaseTpl<Scalar>(func.nx(), func.ndx())
      , underlying_(func)
      {
        assert(func.nr() == 1);
      }

    const C2FunctionTpl<Scalar>& underlying() const
    { return underlying_; }


    Scalar call(const ConstVectorRef& x) const
    {
      return underlying_(x)(0);
    }

    void computeGradient(const ConstVectorRef& x, VectorRef out) const
    {
      underlying_.computeJacobian(x, out.transpose());
    }

    void computeHessian(const ConstVectorRef& x, MatrixRef Hout) const
    {
      VectorXs v = VectorXs::Ones(1);
      underlying_.vectorHessianProduct(x, v, Hout);
    }

  };

}  // namespace proxnlp
  
