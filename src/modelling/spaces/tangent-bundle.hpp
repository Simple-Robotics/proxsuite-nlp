#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"

namespace lienlp
{

  /**
   * @brief     Tangent bundle of a base manifold M. This construction is recursive.
   */
  template<class Base>
  struct TangentBundleTpl : public ManifoldAbstractTpl<typename Base::Scalar>
  {
  protected:
    Base m_base;
  public:
    using Self = TangentBundleTpl<Base>;
    using Scalar = typename Base::Scalar;
    enum {
      Options = Base::Options
    };

    LIENLP_DEFINE_MANIFOLD_TYPES(Base)

    /// Constructor using base space instance.
    TangentBundleTpl(Base base) : m_base(base) {}; 
    /// Constructor using base space constructor.
    template<typename... BaseCtorArgs>
    TangentBundleTpl(BaseCtorArgs... args) : m_base(Base(args...)) {}

    /// Declare implementations

    inline int nx() const { return m_base.nx() + m_base.ndx(); }
    inline int ndx() const { return 2 * m_base.ndx(); }

    PointType neutral() const;
    PointType rand() const;

    const Base& getBaseSpace() const { return m_base; }

    /// @name   Implementations of operators

    void integrate_impl(const ConstVectorRef& x,
                        const ConstVectorRef& dx,
                        VectorRef out) const;

    void difference_impl(const ConstVectorRef& x0,
                         const ConstVectorRef& x1,
                         VectorRef out) const;

    void Jintegrate_impl(const ConstVectorRef& x,
                    const ConstVectorRef& v,
                    MatrixRef Jout,
                    int arg) const;

    void Jdifference_impl(const ConstVectorRef& x0,
                     const ConstVectorRef& x1,
                     MatrixRef Jout,
                     int arg) const;

    /// Get base point of an element of the tangent bundle.
    /// This map is exactly the natural projection.
    template<typename Point>
    typename Point::ConstSegmentReturnType
    getBasePoint(const Point& x) const
    {
      return x.head(m_base.nx());
    }

    template<typename Point>
    typename Point::SegmentReturnType
    getBasePointWrite(const Point& x) const
    {
      return LIENLP_EIGEN_CONST_CAST(Point, x).head(m_base.nx());
    }

    template<typename Tangent>
    typename Tangent::ConstSegmentReturnType
    getBaseTangent(const Tangent& v) const
    {
      return v.head(m_base.ndx());
    }

    template<typename Tangent>
    typename Tangent::SegmentReturnType
    getTangentHeadWrite(const Tangent& v) const
    {
      return LIENLP_EIGEN_CONST_CAST(Tangent, v).head(m_base.ndx());
    }

    template<typename Jac>
    Eigen::Block<Jac, Eigen::Dynamic, Eigen::Dynamic>
    getBaseJacobian(const Jac& J) const
    {
      return LIENLP_EIGEN_CONST_CAST(Jac, J).topLeftCorner(m_base.ndx(), m_base.ndx());
    }

  };

} // namespace lienlp

#include "lienlp/modelling/spaces/tangent-bundle.hxx"
