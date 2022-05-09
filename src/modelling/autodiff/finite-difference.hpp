/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "proxnlp/function-base.hpp"


namespace proxnlp
{
namespace autodiff
{

  enum FDLevel
  {
    TOC1=0,   ///< Cast to a \f$C^1\f$ function.
    TOC2=1    ///< Cast to a \f$C^2\f$ function.
  };

  /// Type of finite differences: forward, central, or forward.
  ///
  enum FDType
  {
    BACKWARD,   ///< Backward finite differences\f$\frac{f_{i} - f_{i-1}}h\f$
    CENTRAL,    ///< Central finite differences\f$\frac{f_{i+1} - f_{i-1}}h\f$
    FORWARD     ///< Forward finite differences\f$\frac{f_{i+1} - f_i}h\f$
  };

  namespace internal
  {

    // fwd declare the implementation of finite difference algorithms.
    template<typename Scalar, FDLevel n, FDType = CENTRAL>
    struct finite_difference_impl;

    template<typename Scalar>
    struct finite_difference_impl<Scalar, TOC1> : virtual C1FunctionTpl<Scalar>
    {
      PROXNLP_FUNCTION_TYPEDEFS(Scalar);
      using FuncType = BaseFunctionTpl<Scalar>;
      using Base = C1FunctionTpl<Scalar>;
      using Base::computeJacobian;

      const ManifoldAbstractTpl<Scalar>& space;
      const FuncType& func;
      Scalar fd_eps;

      finite_difference_impl(const ManifoldAbstractTpl<Scalar>& space,
                             const FuncType& func,
                             const Scalar fd_eps)
         : C1FunctionTpl<Scalar>(func.nx(), func.ndx(), func.nr())
         , space(space)
         , func(func)
         , fd_eps(fd_eps)
         {}

      void computeJacobian(const ConstVectorRef& x,
                           MatrixRef Jout) const override
      {
        VectorXs ei(func.ndx());
        VectorXs xplus  = space.neutral();
        VectorXs xminus = space.neutral();
        ei.setZero();
        for (int i = 0; i < func.ndx(); i++)
        {
          ei(i) = fd_eps;
          space.integrate(x,  ei, xplus);
          space.integrate(x, -ei, xminus);
          Jout.col(i) = (func(xplus) - func(xminus)) / (2 * fd_eps);
          ei(i) = 0.;
        }
      }
    };

    template<typename Scalar>
    struct finite_difference_impl<Scalar, TOC2> : virtual C2FunctionTpl<Scalar>
    {
      PROXNLP_FUNCTION_TYPEDEFS(Scalar);
      using FuncType = C1FunctionTpl<Scalar>;
      using Base = C2FunctionTpl<Scalar>;
      using Base::vectorHessianProduct;

      const ManifoldAbstractTpl<Scalar>& space;
      const FuncType& func;
      Scalar fd_eps;

      finite_difference_impl(const ManifoldAbstractTpl<Scalar>& space,
                             const FuncType& func,
                             const Scalar fd_eps)
         : Base(func.nx(), func.ndx(), func.nr())
         , space(space)
         , func(func)
         , fd_eps(fd_eps)
         {}

      void vectorHessianProduct(const ConstVectorRef& x,
                                const ConstVectorRef& v,
                                MatrixRef Hout) const override
      {
        VectorXs ei(func.ndx());
        VectorXs xplus  = space.neutral();
        VectorXs xminus = space.neutral();
        MatrixXs Jplus(func.nr(),  func.ndx());
        MatrixXs Jminus(func.nr(), func.ndx());
        Jplus.setZero();
        Jminus.setZero();
        ei.setZero();

        for (int i = 0; i < func.ndx(); i++)
        {
          ei(i) = fd_eps;
          space.integrate(x,  ei, xplus);
          space.integrate(x, -ei, xminus);
          func.computeJacobian(xplus, Jplus);
          func.computeJacobian(xminus, Jminus);
          Hout.col(i) = ((Jplus - Jminus) / (2 * fd_eps)).transpose() * v;
          ei(i) = 0.;
        }
      }
    };

  } // namespace internal

  template<typename Scalar, FDLevel n = TOC1>
  struct finite_difference_wrapper;

  /** @brief    Approximate the derivatives of a given function 
   *            using finite differences, to downcast the function to a C1FunctionTpl.
   */
  template<typename _Scalar>
  struct finite_difference_wrapper<_Scalar, TOC1>
    : internal::finite_difference_impl<_Scalar, TOC1>
  {
    using Scalar = _Scalar;

    using InputType = BaseFunctionTpl<Scalar>;
    using OutType = C1FunctionTpl<Scalar>;
    using Base = internal::finite_difference_impl<Scalar, TOC1>;
    using Base::computeJacobian;

    PROXNLP_FUNCTION_TYPEDEFS(Scalar);

    finite_difference_wrapper(const ManifoldAbstractTpl<Scalar>& space,
                             const InputType& func,
                             const Scalar fd_eps)
      : OutType(func.nx(), func.ndx(), func.nr())
      , Base(space, func, fd_eps)
      {}

    ReturnType operator()(const ConstVectorRef& x) const { return this->func(x); }

  };


  /** @brief    Approximate the second derivatives of a given function using finite differences.
   * 
   *  @details  This class inherits from the C1 finite_difference_wrapper<_Scalar, TOC1>,
   *            and the C2 implementation.
   */
  template<typename _Scalar>
  struct finite_difference_wrapper<_Scalar, TOC2>
    : internal::finite_difference_impl<_Scalar, TOC1>
    , internal::finite_difference_impl<_Scalar, TOC2>
  {
    using Scalar = _Scalar;

    using InputType = C1FunctionTpl<Scalar>;
    using OutType = C2FunctionTpl<Scalar>;

    using Base1 = internal::finite_difference_impl<Scalar, TOC1>;
    using Base2 = internal::finite_difference_impl<Scalar, TOC2>;
    using Base1::computeJacobian;
    using Base2::vectorHessianProduct;

    PROXNLP_FUNCTION_TYPEDEFS(Scalar);

    finite_difference_wrapper(const ManifoldAbstractTpl<Scalar>& space,
                             const InputType& func,
                             const Scalar fd_eps)
      : C1FunctionTpl<Scalar>(func.nx(), func.ndx(), func.nr())
      , OutType(func.nx(), func.ndx(), func.nr())
      , Base1(space, func, fd_eps)
      , Base2(space, func, fd_eps)
      {}

    ReturnType operator()(const ConstVectorRef& x) const override { return static_cast<const Base1&>(*this).func(x); }
    void computeJacobian(const ConstVectorRef& x,
                         MatrixRef Jout) const override
    {
      Base1::computeJacobian(x, Jout);
    }

  };

} // namespace autodiff
} // namespace proxnlp
