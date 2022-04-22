#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/modelling/spaces/vector-space.hpp"



namespace proxnlp
{
  /** @brief    The cartesian product of two manifolds.
   */
  template<typename _Scalar>
  struct CartesianProductTpl : ManifoldAbstractTpl<_Scalar>
  {
    using Scalar = _Scalar;

    using Base = ManifoldAbstractTpl<Scalar>;
    PROXNLP_DEFINE_MANIFOLD_TYPES(Base)

    const Base& left;
    const Base& right;

    CartesianProductTpl(const Base& left, const Base& right)
      : left(left), right(right) {}

    inline int nx()  const { return left.nx() + right.nx(); }
    inline int ndx() const { return left.ndx() + right.ndx(); }

    PointType neutral() const override
    {
      PointType out(this->nx());
      out.setZero();
      out.head(left.nx()) = left.neutral();
      out.tail(right.nx()) = right.neutral();
      return out;
    }

    PointType rand() const override
    {
      PointType out(this->nx());
      out.setZero();
      out.head(left.nx()) = left.rand();
      out.tail(right.nx()) = right.rand();
      return out;
    }

    void integrate_impl(const ConstVectorRef& x,
                        const ConstVectorRef& v,
                        VectorRef out) const
    {
      left .integrate(x.head(left .nx()), v.head(left.ndx()), out.head(left.nx()));
      right.integrate(x.head(right.nx()), v.head(right.ndx()), out.head(right.nx()));
    }

    void difference_impl(const ConstVectorRef& x0,
                         const ConstVectorRef& x1,
                         VectorRef out) const
    {
      left .difference(x0.head(left .nx()), x1.head(left.nx()), out.head(left.ndx()));
      right.difference(x0.head(right.nx()), x1.head(right.nx()), out.head(right.ndx()));
    }
    
    void Jintegrate_impl(
      const ConstVectorRef& x,
      const ConstVectorRef& v,
      MatrixRef Jout,
      int arg) const
    {
      const int nx1 = left .nx();
      const int nx2 = right.nx();
      const int ndx1 = left .ndx();
      const int ndx2 = right.ndx();
      left .Jintegrate(x.head(nx1),
                        v.head(ndx1),
                        Jout.topLeftCorner(ndx1, ndx1),
                        arg);
      right.Jintegrate(x.tail(nx2),
                        v.tail(ndx2),
                        Jout.bottomRightCorner(ndx2, ndx2),
                        arg);

    }

    void Jdifference_impl(
      const ConstVectorRef& x0,
      const ConstVectorRef& x1,
      MatrixRef Jout,
      int arg) const
    {
      const int nx1 = left .nx();
      const int nx2 = right.nx();
      const int ndx1 = left .ndx();
      const int ndx2 = right.ndx();
      left .Jdifference(x0.head(nx1),
                         x1.head(nx1),
                         Jout.topLeftCorner(ndx1, ndx1),
                         arg);
      right.Jdifference(x0.tail(nx2),
                         x1.tail(nx2),
                         Jout.bottomRightCorner(ndx2, ndx2),
                         arg);

    }

  };


  /// Direct product of two manifolds as a cartesian product.
  template<typename Scalar>
  CartesianProductTpl<Scalar>
  operator*(const ManifoldAbstractTpl<Scalar>& left,
            const ManifoldAbstractTpl<Scalar>& right)
  {
    return CartesianProductTpl<Scalar>(left, right);
  }

  
} // namespace proxnlp

