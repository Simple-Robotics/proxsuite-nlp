/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/function-base.hpp"

namespace proxnlp {

/**
 * @brief   Base constraint set type.
 *
 * @details Constraint sets can be the negative or positive orthant, the
 * \f$\{0\}\f$ singleton, cones, etc... The expected inputs are constraint
 * values or shifted constraint values (as in ALM-type algorithms).
 */
template <typename _Scalar> struct ConstraintSetBase {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ActiveType = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

  /// Do not use the vector-Hessian product in the Hessian
  /// for Gauss Newton.
  virtual bool disableGaussNewton() const { return true; }

  /**
   * Provided the image @p zproj by the proximal/projection map, evaluate the
   * nonsmooth penalty or constraint set indicator function.
   * @note This will be 0 for projection operators.
   */
  virtual Scalar evaluate(const ConstVectorRef & /*zproj*/) const { return 0.; }

  /**
   * @brief Compute projection of variable @p z onto the constraint set.
   *
   * @param[in]   z     Input vector
   * @param[out]  zout  Output projection
   */
  virtual void projection(const ConstVectorRef &z, VectorRef zout) const = 0;

  /**
   * @brief Compute projection of @p z onto the normal cone to the set. The
   * default implementation is just $\f\mathrm{id} - P\f$.
   *
   * @param[in]   z     Input vector
   * @param[out]  zout  Output projection on the normal projection
   */
  virtual void normalConeProjection(const ConstVectorRef &z,
                                    VectorRef zout) const = 0;

  /**
   * Apply the jacobian of the constraint set projection operator. This carries
   * out the product \f$ \partial\prox . J \f$.
   *
   * @param[in]  z     Input vector (multiplier estimate)
   * @param[out] Jout  Output Jacobian matrix, which will be modifed in-place
   * and returned.
   */
  virtual void applyProjectionJacobian(const ConstVectorRef &z,
                                       MatrixRef Jout) const;

  /**
   * Apply the jacobian of the projection on the normal cone.
   *
   * @param[in]  z     Input vector
   * @param[out] Jout  Output Jacobian matrix of shape \f$(nr, ndx)\f$, which
   * will be modified in place. The modification should be a row-wise operation.
   */
  virtual void applyNormalConeProjectionJacobian(const ConstVectorRef &z,
                                                 MatrixRef Jout) const;

  /// Update proximal parameter; this applies to when this class is a proximal
  /// operator.
  virtual void setProxParameters(const Scalar){};

  /// Compute the active set of the constraint.
  /// Active means the Jacobian of the proximal operator is nonzero.
  virtual void computeActiveSet(const ConstVectorRef &z,
                                Eigen::Ref<ActiveType> out) const = 0;

  virtual ~ConstraintSetBase<Scalar>() = default;

  bool operator==(const ConstraintSetBase<Scalar> &rhs) { return this == &rhs; }

  /**
   * @brief Evaluate the Moreau envelope with parameter @p mu for the given
   * contraint set or nonsmooth penalty \f$P\f$ at point @p zin.
   *
   * @details    The envelope is
   *              \f[ P(\prox_{P/\mu}(z)) + \frac{1}{2\mu} \| z -
   * \prox_{P/\mu}(z)
   * \|^2. \f]
   *
   * @param cstr_set  The constraint set/nonsmooth penalty.
   * @param zin    		The input.
   * @param zproj     Projection of the input to the normal.
   * @param inv_mu    The inverse penalty parameter.
   */
  Scalar evaluateMoreauEnvelope(const ConstVectorRef &zin,
                                const ConstVectorRef &zproj,
                                const Scalar inv_mu) const {
    Scalar res = evaluate(zin - zproj);
    res += static_cast<Scalar>(0.5) * inv_mu * zproj.squaredNorm();
    return res;
  }

  /// @copybrief evaluateMoreauEnvelope(). This variant evaluates the prox map.
  Scalar computeMoreauEnvelope(const ConstVectorRef &zin, VectorRef zprojout,
                               const Scalar inv_mu) const {
    normalConeProjection(zin, zprojout);
    return evaluateMoreauEnvelope(zin, zprojout, inv_mu);
  }
};

/** @brief    Packs a ConstraintSetBase and C2FunctionTpl together.
 *
 */
template <typename _Scalar> struct ConstraintObject {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  using FunctionType = C2FunctionTpl<Scalar>;
  using ConstraintSet = ConstraintSetBase<Scalar>;

  shared_ptr<FunctionType> func_;
  shared_ptr<ConstraintSet> set_;

  const FunctionType &func() const { return *func_; }
  int nr() const { return func_->nr(); }

  ConstraintObject(shared_ptr<FunctionType> func, shared_ptr<ConstraintSet> set)
      : func_(func), set_(set) {}
};

} // namespace proxnlp

#include "proxnlp/constraint-base.hxx"
