#pragma once

#include "proxnlp/manifold-base.hpp"

namespace proxnlp {

/**
 * @brief     Tangent bundle of a base manifold M.
 */
template <class Base>
struct TangentBundleTpl : public ManifoldAbstractTpl<typename Base::Scalar> {
protected:
  Base m_base;

public:
  using Self = TangentBundleTpl<Base>;
  using Scalar = typename Base::Scalar;
  enum { Options = Base::Options };

  PROXNLP_DEFINE_MANIFOLD_TYPES(Base)

  /// Constructor using base space instance.
  TangentBundleTpl(Base base) : m_base(base){};
  /// Constructor using base space constructor.
  template <typename... BaseCtorArgs>
  TangentBundleTpl(BaseCtorArgs... args) : m_base(Base(args...)) {}

  /// Declare implementations

  inline int nx() const { return m_base.nx() + m_base.ndx(); }
  inline int ndx() const { return 2 * m_base.ndx(); }

  PointType neutral() const;
  PointType rand() const;

  const Base &getBaseSpace() const { return m_base; }

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
    const int nv_ = m_base.ndx();
    m_base.JintegrateTransport(getBasePoint(x), getBaseTangent(v),
                               Jout.topRows(nv_), arg);
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const;

  virtual void interpolate_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, const Scalar &u,
                                VectorRef out) const {
    m_base.interpolate(getBasePoint(x0), getBasePoint(x1), u,
                       getBasePointWrite(out));
    out.tail(m_base.ndx()) =
        (Scalar(1.) - u) * getBaseTangent(x0) + u * getBaseTangent(x1);
  }

  /// Get base point of an element of the tangent bundle.
  /// This map is exactly the natural projection.
  template <typename Point>
  typename Point::ConstSegmentReturnType getBasePoint(const Point &x) const {
    return x.head(m_base.nx());
  }

  template <typename Point>
  typename Point::SegmentReturnType getBasePointWrite(const Point &x) const {
    return PROXNLP_EIGEN_CONST_CAST(Point, x).head(m_base.nx());
  }

  template <typename Tangent>
  typename Tangent::ConstSegmentReturnType
  getBaseTangent(const Tangent &v) const {
    return v.head(m_base.ndx());
  }

  template <typename Tangent>
  typename Tangent::SegmentReturnType
  getTangentHeadWrite(const Tangent &v) const {
    return PROXNLP_EIGEN_CONST_CAST(Tangent, v).head(m_base.ndx());
  }

  template <typename Jac>
  Eigen::Block<Jac, Eigen::Dynamic, Eigen::Dynamic>
  getBaseJacobian(const Jac &J) const {
    return PROXNLP_EIGEN_CONST_CAST(Jac, J).topLeftCorner(m_base.ndx(),
                                                          m_base.ndx());
  }
};

} // namespace proxnlp

#include "proxnlp/modelling/spaces/tangent-bundle.hxx"
