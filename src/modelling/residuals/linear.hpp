#pragma once


#include "lienlp/residual-base.hpp"


namespace lienlp
{
  
  /**
   * @brief Linear residuals \f$r(x) = Ax + b\f$.
   */
  template<typename _Scalar>
  struct LinearResidual : ResidualBase<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    using Base = ResidualBase<Scalar>;
    using Base::computeJacobian;

    const MatrixXs mat;
    const VectorXs b;

    LinearResidual(const ConstMatrixRef& A, const ConstVectorRef& b)
      : Base(A.cols(), A.cols(), A.rows()), mat(A), b(b) {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      return mat * x + b;
    }

    void computeJacobian(const ConstVectorRef&, Eigen::Ref<JacobianType> Jout) const
    {
      Jout = mat;
    }
  };
} // namespace lienlp
