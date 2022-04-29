#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/function-base.hpp"


namespace proxnlp
{

  /**
   * @brief   Base constraint set type.
   * 
   * @details Constraint sets can be the negative or positive orthant, the \f$\{0\}\f$ singleton, cones, etc...
   */
  template<typename _Scalar>
  struct ConstraintSetBase
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)
    using ActiveType = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    /// Do not use the vector-Hessian product in the Hessian
    /// for Gauss Newton.
    virtual bool disableGaussNewton() const { return true; }

    /// Compute projection of variable @p z onto the constraint set.
    virtual ReturnType projection(const ConstVectorRef& z) const = 0;

    /* Compute projection of @p z onto the normal cone to the set.
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


  /** @brief    Packs a ConstraintSetBase and C2FunctionTpl together.
   * 
   */
  template<typename _Scalar>
  struct ConstraintObject
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)

    using FunctionType = C2FunctionTpl<Scalar>;
    const FunctionType& m_func;

    shared_ptr<ConstraintSetBase<Scalar>> m_set;

    explicit ConstraintObject(const FunctionType& func, const shared_ptr<ConstraintSetBase<Scalar>>& set)
      : m_func(func)
      , m_set(set)
      {}

    explicit ConstraintObject(const FunctionType& func, ConstraintSetBase<Scalar>* set)
      : m_func(func)
      , m_set(set)
      {}

    /// Get dimension of constraint representation.
    int nr()  const { return m_func.nr(); }
  };

}

#include "proxnlp/constraint-base.hxx"
