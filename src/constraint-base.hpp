#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/function-base.hpp"


namespace proxnlp
{

  /**
   * @brief   Constraint format: negative/positive orthant, cones, etc...
   */
  template<typename _Scalar>
  struct ConstraintSetBase
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTOR_TYPEDEFS(Scalar)
    using ActiveType = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    using FunctionType = C2FunctionTpl<Scalar>;
    const FunctionType& m_func;

    explicit ConstraintSetBase<Scalar>(const FunctionType& func)
      : m_func(func)
      {}

    /// Do not use the vector-Hessian product in the Hessian
    /// for Gauss Newton.
    virtual bool disableGaussNewton() const { return true; }

    /// Get dimension of manifold element representation.
    int nx()  const { return m_func.nx(); }
    /// Get dimension of constraint representation.
    int nr()  const { return m_func.nr(); }
    /// Get tangent space dimension (no. of columns of Jacobian)
    int ndx() const { return m_func.ndx(); }

    /// Compute projection of variable @p z onto the constraint set.
    virtual ReturnType projection(const ConstVectorRef& z) const = 0;

    /** Compute projection of @p z onto the normal cone to the set.
     * The default implementation is just $\f\mathrm{id} - P\f$.
     */
    inline ReturnType normalConeProjection(const ConstVectorRef& z) const;

    /** Apply the jacobian of the constraint set projection operator.
     * @param[in]  z     Input vector (multiplier estimate)
     * @param[out] Jout  Output Jacobian matrix, which will be modifed in-place and returned.
     */
    virtual void applyProjectionJacobian(const ConstVectorRef& z, MatrixRef Jout) const;

    /** Apply the jacobian of the projection on the normal cone.
     * @param[in]  z     Input vector
     * @param[out] Jout  Output Jacobian matrix of shape \f$(nr, ndx)\f$, which will be modified in place.
     *                   The modification should be a row-wise operation.
     */
    virtual void applyNormalConeProjectionJacobian(const ConstVectorRef& z, MatrixRef Jout) const;

    /// Update proximal parameter; this applies to when this class is a proximal operator.
    virtual void updateProxParameters(const Scalar) {};

    /// Compute the active set of the constraint.
    virtual void computeActiveSet(const ConstVectorRef& z, Eigen::Ref<ActiveType> out) const = 0;
    virtual ~ConstraintSetBase<Scalar>() = default;

    bool operator==(const ConstraintSetBase<Scalar>& rhs)
    {
      return this == &rhs;
    }
  };

}

#include "proxnlp/constraint-base.hxx"
