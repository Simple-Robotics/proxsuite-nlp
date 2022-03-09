#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  /**
   * \brief   Base template for constraint functions.
   */
  template<class M, int NC=Eigen::Dynamic, typename... Args>
  struct ConstraintFuncTpl
  {
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)

    using C_t = Eigen::Matrix<Scalar, NC, 1, M::Options>;
    using Jacobian_t = Eigen::Matrix<Scalar, NC, Eigen::Dynamic, M::Options>;

    virtual C_t operator()(const VectorXs& x, const Args&...) const = 0;
    virtual Jacobian_t jacobian(const VectorXs& x, Jacobian_t& Jout, const Args&...) const = 0;

    /// TODO hvp (hessian vector product)

    virtual ~ConstraintFuncTpl<M, NC, Args...>() = default;

  };

  /**
   * \brief   Constraint format: negative/positive orthant, cones, etc...
   */
  template<class M>
  struct ConstraintFormatBaseTpl
  {
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)
    using C_t = Eigen::Matrix<Scalar, NC, 1, M::Options>;
    using Jacobian_t = Eigen::Matrix<Scalar, NC, Eigen::Dynamic, M::Options>;

    /// TODO hvp (hessian vector product)

    virtual C_t projection() = 0;
    virtual Jacobian_t Jprojection() = 0;
  };

}
