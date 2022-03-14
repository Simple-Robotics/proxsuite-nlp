#pragma once

#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"

namespace lienlp {
  
/**
 * Cost function which is a weighted squared residual.
 */
template<typename _Scalar>
class QuadResidualCost : public CostFunction<_Scalar>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Scalar = _Scalar;
  using CstrType = ConstraintFuncTpl<Scalar>;  // base constraint func to use
  LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
  using CostFunction<Scalar>::hessian;

  CstrType* m_residual;
  MatrixXs m_weights;

  QuadResidualCost(CstrType* residual,
                   const VectorXs& weightMatrix)
  : m_residual(residual), m_weights(weightMatrix)
  {}

  template<typename... ResidualArgs>
  QuadResidualCost(const VectorXs& weightMatrix,
                   ResidualArgs&... args)
                   : QuadResidualCost(new CstrType(args...),  weightMatrix)
  {}

  Scalar operator()(const VectorXs& x) const
  {
    VectorXs err = m_residual->operator()(x);
    return Scalar(0.5) * err.dot(m_weights * err);
  }

  VectorXs gradient(const VectorXs& x) const
  {
    MatrixXs Jres;
    m_residual->jacobian(x, Jres);
    VectorXs err = m_residual->operator()(x);
    return Jres * (m_weights * err);
  }

  void hessian(const VectorXs& x, MatrixXs& out) const
  {
    MatrixXs Jres;
    m_residual->jacobian(x, Jres);
    const int ndx = Jres.cols();
    out.resize(ndx, ndx);
    out.noalias() = Jres.transpose() * (m_weights * Jres);
  }

};

} // namespace lienlp
