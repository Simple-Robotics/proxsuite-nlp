#pragma once


#include "lienlp/function-base.hpp"
#include "lienlp/function-ops.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"


namespace lienlp
{
  
  /**
   * @brief Linear residuals \f$r(x) = Ax + b\f$.
   */
  template<typename _Scalar>
  struct LinearFunction : C2FunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

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
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)

    using M = ManifoldAbstractTpl<Scalar>;

    LinearFunctionDifferenceToPoint(const M& manifold, const ConstVectorRef& target,
                        const ConstMatrixRef& A, const ConstVectorRef& b)
                        : Base(LinearFunction<Scalar>(A, b),
                               ManifoldDifferenceToPoint<Scalar>(manifold, target)
                               ) {}

  };



} // namespace lienlp
