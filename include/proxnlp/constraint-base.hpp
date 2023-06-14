/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/function-base.hpp"

namespace proxnlp {

///
/// @brief   Base constraint set type.
///
/// @details Constraint sets can be the negative or positive orthant, the
/// \f$\{0\}\f$ singleton, cones, etc... The expected inputs are constraint
/// values or shifted constraint values (as in ALM-type algorithms).
///
template <typename _Scalar> struct ConstraintSetBase {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ActiveType = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
  using Self = ConstraintSetBase<Scalar>;

  /// Do not use the vector-Hessian product in the Hessian
  /// for Gauss Newton.
  virtual bool disableGaussNewton() const { return true; }

  /// Provided the image @p zproj by the proximal/projection map, evaluate the
  /// nonsmooth penalty or constraint set indicator function.
  /// @note This will be 0 for projection operators.
  virtual Scalar evaluate(const ConstVectorRef & /*zproj*/) const { return 0.; }

  /// @brief Compute projection of variable @p z onto the constraint set.
  ///
  /// @param[in]   z     Input vector
  /// @param[out]  zout  Output projection
  virtual void projection(const ConstVectorRef &z, VectorRef zout) const = 0;

  /// @brief Compute projection of @p z onto the normal cone to the set. The
  /// default implementation is just \f$ \mathrm{id} - P\f$.
  ///
  /// @param[in]   z     Input vector
  /// @param[out]  zout  Output projection on the normal projection
  virtual void normalConeProjection(const ConstVectorRef &z,
                                    VectorRef zout) const = 0;

  /// @brief Apply a jacobian of the projection/proximal operator to a matrix.
  /// @details This carries out the product \f$PJ\f$, where \f$ P
  /// \in\partial_B\prox(z)\f$.
  ///
  /// @param[in]  z     Input vector (multiplier estimate)
  /// @param[out] Jout  Output Jacobian matrix, which will be modifed in-place
  /// and returned.
  virtual void applyProjectionJacobian(const ConstVectorRef &z,
                                       MatrixRef Jout) const;

  /// @brief Apply the jacobian of the projection on the normal cone.
  ///
  /// @param[in]  z     Input vector
  /// @param[out] Jout  Output Jacobian matrix of shape \f$(nr, ndx)\f$, which
  /// will be modified in place. The modification should be a row-wise
  /// operation.
  virtual void applyNormalConeProjectionJacobian(const ConstVectorRef &z,
                                                 MatrixRef Jout) const;

  /// @brief Update proximal parameter; this applies to when this class is a
  /// proximal operator that isn't a projection (e.g. \f$ \ell_1 \f$).
  void setProxParameters(const Scalar mu) {
    mu_ = mu;
    mu_inv_ = 1. / mu;
  };

  /// Compute the active set of the constraint.
  /// Active means the Jacobian of the proximal operator is nonzero.
  virtual void computeActiveSet(const ConstVectorRef &z,
                                Eigen::Ref<ActiveType> out) const = 0;

  virtual ~ConstraintSetBase<Scalar>() = default;

  bool operator==(const ConstraintSetBase<Scalar> &rhs) { return this == &rhs; }

  ///
  /// @brief Evaluate the Moreau envelope with parameter @p mu for the given
  /// contraint set or nonsmooth penalty \f$g\f$ at point @p zin.
  ///
  /// @details    The envelope is
  ///              \f[ M_{\mu g}(z) := g(\prox_{\mu g}(z)) + \frac{1}{2\mu} \| z
  ///              - \prox_{\mu g}(z) \|^2. \f]
  ///
  /// @param zin    		The input.
  /// @param zproj     Projection of the input to the normal.
  ///
  Scalar evaluateMoreauEnvelope(const ConstVectorRef &zin,
                                const ConstVectorRef &zproj) const {
    Scalar res = evaluate(zin - zproj);
    res += static_cast<Scalar>(0.5) * mu_inv_ * zproj.squaredNorm();
    return res;
  }

  /// @copybrief evaluateMoreauEnvelope(). This variant evaluates the prox map.
  /// @copydetails evaluateMoreauEnvelope
  Scalar computeMoreauEnvelope(const ConstVectorRef &zin,
                               VectorRef zprojout) const {
    normalConeProjection(zin, zprojout);
    return evaluateMoreauEnvelope(zin, zprojout);
  }

  Scalar mu() const { return mu_; }
  Scalar mu_inv() const { return mu_inv_; }

protected:
  Scalar mu_ = 0.;
  Scalar mu_inv_;
};

/** @brief    Packs a ConstraintSetBase and C2FunctionTpl together.
 *
 */
template <typename _Scalar> struct ConstraintObjectTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  using FunctionType = C2FunctionTpl<Scalar>;
  using ConstraintSet = ConstraintSetBase<Scalar>;

  shared_ptr<FunctionType> func_;
  shared_ptr<ConstraintSet> set_;

  const FunctionType &func() const { return *func_; }
  int nr() const { return func_->nr(); }

  ConstraintObjectTpl(shared_ptr<FunctionType> func,
                      shared_ptr<ConstraintSet> set)
      : func_(func), set_(set) {}

  bool operator==(const ConstraintObjectTpl &other) {
    return (func_ == other.func_) && (set_ && other.set_);
  }
};

} // namespace proxnlp

#include "proxnlp/constraint-base.hxx"
