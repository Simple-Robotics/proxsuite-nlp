#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  #define LIENLP_CSTR_TYPES(Scalar, Options)                       \
    using C_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>; \
    using Jacobian_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;

  /**
   * @brief   Base template for constraint functions.
   */
  template<class _M>
  struct ConstraintFuncTpl
  {
    using M = _M;
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)
    LIENLP_CSTR_TYPES(Scalar, M::Options)

    virtual C_t operator()(const VectorXs& x) const = 0;
    /// @brief      Jacobian matrix of the constraint function.
    virtual void jacobian(const VectorXs& x, Jacobian_t& Jout) const = 0;

    /** @copybrief jacobian()
     * 
     * Allocated version of the jacobian() method.
     */
    Jacobian_t jacobian(const VectorXs& x) const
    {
      Jacobian_t Jout;
      jacobian(x, Jout);
      return Jout;
    }
    
    /// TODO hvp (hessian vector product)

    virtual ~ConstraintFuncTpl<M>() = default;

  };


  /**
   * @brief   Constraint format: negative/positive orthant, cones, etc...
   */
  template<class M>
  struct ConstraintFormatBaseTpl
  {
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)
    LIENLP_CSTR_TYPES(Scalar, M::Options)

    /// TODO hvp (hessian vector product)

    virtual C_t projection() = 0;
    virtual Jacobian_t Jprojection() = 0;
  };

}

#include "lienlp/constraint-base.hxx"
