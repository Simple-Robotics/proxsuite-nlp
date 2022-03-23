#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"
#include "lienlp/residual-base.hpp"


namespace lienlp {

  /**
   * @brief   Constraint format: negative/positive orthant, cones, etc...
   */
  template<typename _Scalar>
  struct ConstraintSetBase
  {
  public:
    using Scalar = _Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Active_t = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    using functor_t = ResidualBase<Scalar>;
    const functor_t& m_func;


    ConstraintSetBase<Scalar>(const functor_t& func)
      : m_func(func), eye(MatrixXs::Identity(func.nr(), func.nr()))
      {}

    inline ReturnType operator()(const ConstVectorRef& x) const
    {
      return m_func(x);
    }

    /// Get dimension of manifold element representation.
    int nx() const { return m_func.nx(); }
    /// Get dimension of constraint representation.
    int nr() const { return m_func.nr(); }
    /// Get tangent space dimension (no. of columns of Jacobian)
    int ndx() const { return m_func.ndx(); }

    /// Compute projection of variable @p z onto the constraint set.
    virtual ReturnType projection(const ConstVectorRef& z) const = 0;
    /// Compute projection of @p z onto the normal cone to the set.
    inline ReturnType dualProjection(const ConstVectorRef& z) const
    {
      return z - projection(z);
    }
    /// Compute the jacobian of the constraint set projection operator.
    virtual JacobianType Jprojection(const ConstVectorRef& z) const = 0;
    /// Compute the jacobian of the projection on the normal cone.
    inline JacobianType JdualProjection(const ConstVectorRef& z) const
    {
      
      return eye - Jprojection(z);
    }

    /// Update proximal parameter; this applies to when a proximal operator is derived.
    virtual void updateProxParameters(const Scalar mu) {};

    /// Compute the active set of the constraint.
    virtual void computeActiveSet(const ConstVectorRef& z, Eigen::Ref<Active_t> out) const = 0;
    virtual ~ConstraintSetBase<Scalar>() = default;
  private:
    const MatrixXs eye;
  };

}

#include "lienlp/constraint-base.hxx"
