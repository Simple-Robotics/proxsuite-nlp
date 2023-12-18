#pragma once

#include "proxnlp/manifold-base.hpp"

namespace proxnlp {

/**
 * @brief     Tangent bundle of a base manifold M.
 */
template <class Base>
struct TangentBundleTpl : public ManifoldAbstractTpl<typename Base::Scalar> {
protected:
  Base base_;

public:
  using Self = TangentBundleTpl<Base>;
  using Scalar = typename Base::Scalar;
  enum { Options = Base::Options };

  PROXSUITE_NLP_DEFINE_MANIFOLD_TYPES(Base)

  /// Constructor using base space instance.
  TangentBundleTpl(Base base) : base_(base){};
  /// Constructor using base space constructor.
  template <typename... BaseCtorArgs>
  TangentBundleTpl(BaseCtorArgs... args) : base_(Base(args...)) {}

  /// Declare implementations

  inline int nx() const { return base_.nx() + base_.ndx(); }
  inline int ndx() const { return 2 * base_.ndx(); }

  PointType neutral() const;
  PointType rand() const;
  bool isNormalized(const ConstVectorRef &x) const {
    auto p = getBasePoint(x);
    return base_.isNormalized(p);
  }

  const Base &getBaseSpace() const { return base_; }

  /// @name   Implementations of operators

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &dx,
                      VectorRef out) const;

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const;

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const;

  virtual void JintegrateTransport(const ConstVectorRef &x,
                                   const ConstVectorRef &v, MatrixRef Jout,
                                   int arg) const {
    const int nv_ = base_.ndx();
    base_.JintegrateTransport(getBasePoint(x), getBaseTangent(v),
                              Jout.topRows(nv_), arg);
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const;

  virtual void interpolate_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, const Scalar &u,
                                VectorRef out) const {
    base_.interpolate(getBasePoint(x0), getBasePoint(x1), u,
                      getBasePointWrite(out));
    out.tail(base_.ndx()) =
        (Scalar(1.) - u) * getBaseTangent(x0) + u * getBaseTangent(x1);
  }

  /// Get base point of an element of the tangent bundle.
  /// This map is exactly the natural projection.
  template <typename Point>
  typename Point::ConstSegmentReturnType getBasePoint(const Point &x) const {
    return x.head(base_.nx());
  }

  template <typename Point>
  typename Point::SegmentReturnType getBasePointWrite(const Point &x) const {
    return PROXSUITE_NLP_EIGEN_CONST_CAST(Point, x).head(base_.nx());
  }

  template <typename Tangent>
  typename Tangent::ConstSegmentReturnType
  getBaseTangent(const Tangent &v) const {
    return v.head(base_.ndx());
  }

  template <typename Tangent>
  typename Tangent::SegmentReturnType
  getTangentHeadWrite(const Tangent &v) const {
    return PROXSUITE_NLP_EIGEN_CONST_CAST(Tangent, v).head(base_.ndx());
  }

  template <typename Jac>
  Eigen::Block<Jac, Eigen::Dynamic, Eigen::Dynamic>
  getBaseJacobian(const Jac &J) const {
    return PROXSUITE_NLP_EIGEN_CONST_CAST(Jac, J).topLeftCorner(base_.ndx(),
                                                                base_.ndx());
  }
};

} // namespace proxnlp

#include "proxnlp/modelling/spaces/tangent-bundle.hxx"
