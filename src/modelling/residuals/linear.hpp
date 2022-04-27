#pragma once


#include "proxnlp/function-base.hpp"
#include "proxnlp/function-ops.hpp"
#include "proxnlp/modelling/residuals/state-residual.hpp"


namespace proxnlp
{
  
  /**
   * @brief Linear residuals \f$r(x) = Ax + b\f$.
   */
  template<typename _Scalar>
  struct LinearFunction : C2FunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)

    using Base = C2FunctionTpl<Scalar>;
    using Base::computeJacobian;

    const MatrixXs mat;
    const VectorXs b;

    LinearFunction(const ConstMatrixRef& A, const ConstVectorRef& b)
      : Base((int)A.cols(), (int)A.cols(), (int)A.rows()),
        mat(A),
        b(b) {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      return mat * x + b;
    }

    void computeJacobian(const ConstVectorRef&, Eigen::Ref<JacobianType> Jout) const
    {
      Jout = mat;
    }
  };


  /** @brief    Linear function of difference vector on a manifold, of the form
   *            \f$ r(x) = A(x \ominus \bar{x}) + b \f$.
   */
  template<typename _Scalar>
  struct LinearFunctionDifferenceToPoint : ComposeFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    using Base = ComposeFunctionTpl<Scalar>;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

    using M = ManifoldAbstractTpl<Scalar>;

    LinearFunctionDifferenceToPoint(const M& manifold, const ConstVectorRef& target,
                        const ConstMatrixRef& A, const ConstVectorRef& b)
                        : Base(LinearFunction<Scalar>(A, b),
                               ManifoldDifferenceToPoint<Scalar>(manifold, target)
                               ) {}

  };



} // namespace proxnlp
