/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/fwd.hpp"

namespace proxsuite {
namespace nlp {

/// Macro which brings manifold typedefs up into the constraint, cost type, etc.
#define PROXSUITE_NLP_DEFINE_MANIFOLD_TYPES(M)                                 \
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(typename M::Scalar);                          \
  using PointType = typename M::PointType;                                     \
  using TangentVectorType = typename M::TangentVectorType;

/**
 * Base class for manifolds, to use in cost funcs, solvers...
 */
template <typename _Scalar, int _Options> struct ManifoldAbstractTpl {
public:
  using Scalar = _Scalar; /// Scalar type
  enum { Options = _Options };

  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  using PointType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
  using TangentVectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;

  virtual ~ManifoldAbstractTpl() = default;

  /// @brief    Get manifold representation dimension.
  virtual int nx() const = 0;
  /// @brief    Get manifold tangent space dimension.
  virtual int ndx() const = 0;

  /// @brief    Get the neutral element \f$e \in M\f$ from the manifold (if this
  /// makes sense).
  virtual PointType neutral() const { return PointType::Zero(nx()); }
  /// @brief    Sample a random point \f$x \in M\f$ on the manifold.
  virtual PointType rand() const { return PointType::Random(nx()); }
  /// @brief    Check if the input vector @p x is a viable element of the
  /// manifold.
  virtual bool isNormalized(const ConstVectorRef & /*x*/) const { return true; }

  /// Typedef for the tangent space, as a manifold.
  using TangentSpaceType = VectorSpaceTpl<Scalar, Eigen::Dynamic, Options>;

  /// @brief    Return an object representing the tangent space as a manifold.
  const TangentSpaceType tangentSpace() const {
    return TangentSpaceType(this->ndx());
  }

  /// @name     Operations

  /// @brief Manifold integration operation \f$x \oplus v\f$
  void integrate(const ConstVectorRef &x, const ConstVectorRef &v,
                 VectorRef out) const;

  /// @brief    Jacobian of the integation operation.
  void Jintegrate(const ConstVectorRef &x, const ConstVectorRef &v,
                  MatrixRef Jout, int arg) const;

  /// @brief    Perform the parallel transport operation
  ///
  virtual void JintegrateTransport(const ConstVectorRef &x,
                                   const ConstVectorRef &v, MatrixRef Jout,
                                   int arg) const = 0;

  /// @brief Manifold difference/retraction operation \f$x_1 \ominus x_0\f$
  void difference(const ConstVectorRef &x0, const ConstVectorRef &x1,
                  VectorRef out) const;

  /// @brief    Jacobian of the retraction operation.
  void Jdifference(const ConstVectorRef &x0, const ConstVectorRef &x1,
                   MatrixRef Jout, int arg) const;

  void interpolate(const ConstVectorRef &x0, const ConstVectorRef &x1,
                   const Scalar &u, VectorRef out) const;

  /// \name Allocated overloads.
  /// \{

  /// @copybrief integrate()
  ///
  /// Out-of-place variant of integration operator.
  PointType integrate(const ConstVectorRef &x, const ConstVectorRef &v) const {
    PointType out(nx());
    integrate_impl(x, v, out);
    return out;
  }

  /// @copybrief difference()
  ///
  /// Out-of-place version of diff operator.
  TangentVectorType difference(const ConstVectorRef &x0,
                               const ConstVectorRef &x1) const {
    TangentVectorType out(ndx());
    difference_impl(x0, x1, out);
    return out;
  }

  /// @copybrief interpolate_impl()
  PointType interpolate(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        const Scalar &u) const {
    PointType out(nx());
    interpolate_impl(x0, x1, u, out);
    return out;
  }

  /// \}

protected:
  /// Perform the manifold integration operation.
  virtual void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                              VectorRef out) const = 0;

  virtual void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                               MatrixRef Jout, int arg) const = 0;

  /// Implementation of the manifold retraction operation.
  virtual void difference_impl(const ConstVectorRef &x0,
                               const ConstVectorRef &x1,
                               VectorRef out) const = 0;

  virtual void Jdifference_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, MatrixRef Jout,
                                int arg) const = 0;

  /// @brief    Interpolation operation.
  virtual void interpolate_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, const Scalar &u,
                                VectorRef out) const {
    // default implementation
    integrate(x0, u * difference(x0, x1), out);
  }
};

} // namespace nlp
} // namespace proxsuite

#include "proxsuite-nlp/manifold-base.hxx"

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/manifold-base.txx"
#endif
