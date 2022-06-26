#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/modelling/spaces/vector-space.hpp"

#include <type_traits>


namespace proxnlp
{
  namespace {
    /// Typedef in anon namespace for use in rest of file.
    template<typename T>
    using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<T>>;
  }

  /** @brief    The cartesian product of two manifolds.
   */
  template<typename _Scalar>
  struct CartesianProductTpl : ManifoldAbstractTpl<_Scalar>
  {
    using Scalar = _Scalar;

    using Base = ManifoldAbstractTpl<Scalar>;
    PROXNLP_DEFINE_MANIFOLD_TYPES(Base)

    shared_ptr<const Base> left_, right_;
    const Base& left() const { return *left_; }
    const Base& right() const { return *right_; }

    template<typename U, typename V>
    CartesianProductTpl(const U& left, const V& right)
      : left_(std::make_shared<U>(left)), right_(std::make_shared<V>(right)) {}

    CartesianProductTpl(const shared_ptr<Base>& left, const shared_ptr<Base>& right)
      : left_(left), right_(right) {}

    inline int nx()  const { return left().nx() + right().nx(); }
    inline int ndx() const { return left().ndx() + right().ndx(); }

    PointType neutral() const
    {
      PointType out(this->nx());
      out.setZero();
      out.head(left().nx())  = left().neutral();
      out.tail(right().nx()) = right().neutral();
      return out;
    }

    PointType rand() const
    {
      PointType out(this->ndx());
      out.setZero();
      out.head(left().nx())  = left().rand();
      out.tail(right().nx()) = right().rand();
      return out;
    }

    void integrate_impl(const ConstVectorRef& x,
                        const ConstVectorRef& v,
                        VectorRef out) const
    {
      left() .integrate(x.head(left() .nx()), v.head(left() .ndx()), out.head(left() .nx()));
      right().integrate(x.head(right().nx()), v.head(right().ndx()), out.head(right().nx()));
    }

    void difference_impl(const ConstVectorRef& x0,
                         const ConstVectorRef& x1,
                         VectorRef out) const
    {
      left() .difference(x0.head(left() .nx()), x1.head(left() .nx()), out.head(left() .ndx()));
      right().difference(x0.head(right().nx()), x1.head(right().nx()), out.head(right().ndx()));
    }
    
    void Jintegrate_impl(
      const ConstVectorRef& x,
      const ConstVectorRef& v,
      MatrixRef Jout,
      int arg) const
    {
      const int nx1 = left().nx();
      const int nx2 = right().nx();
      const int ndx1 = left().ndx();
      const int ndx2 = right().ndx();
      left(). Jintegrate(x.head(nx1), v.head(ndx1), Jout.topLeftCorner    (ndx1, ndx1), arg);
      right().Jintegrate(x.tail(nx2), v.tail(ndx2), Jout.bottomRightCorner(ndx2, ndx2), arg);
    }

    void JintegrateTransport(
      const ConstVectorRef& x,
      const ConstVectorRef& v,
      MatrixRef Jout,
      int arg) const
    {
      const int nx1 = left().nx();
      const int nx2 = right().nx();
      const int ndx1 = left().ndx();
      const int ndx2 = right().ndx();
      left() .JintegrateTransport(x.head(nx1), v.head(ndx1), Jout.topRows   (ndx1), arg);
      right().JintegrateTransport(x.tail(nx2), v.head(ndx2), Jout.bottomRows(ndx2), arg);
    }

    void Jdifference_impl(
      const ConstVectorRef& x0,
      const ConstVectorRef& x1,
      MatrixRef Jout,
      int arg) const
    {
      const int nx1 = left().nx();
      const int nx2 = right().nx();
      const int ndx1 = left().ndx();
      const int ndx2 = right().ndx();
      left() .Jdifference(x0.head(nx1), x1.head(nx1), Jout.topLeftCorner    (ndx1, ndx1), arg);
      right().Jdifference(x0.tail(nx2), x1.tail(nx2), Jout.bottomRightCorner(ndx2, ndx2), arg);

    }

  };

  /// Direct product of two manifolds as a cartesian product.
  // template<typename U, typename V>
  // CartesianProductTpl<typename U::Scalar> operator*(const U& left, const V& right)
  // {
  //   using T = typename U::Scalar;
  //   static_assert(std::is_same<T, typename V::Scalar>::value, "Both arguments should have same Scalar template arg!");
  //   return CartesianProductTpl<T>(std::make_shared<U>(left), std::make_shared<V>(right));
  // }

  template<typename T>
  CartesianProductTpl<T> operator*(const ManifoldPtr<T>& left, const ManifoldPtr<T>& right)
  {
    return CartesianProductTpl<T>(left, right);
  }

  template<typename T>
  CartesianProductTpl<T> operator*(const CartesianProductTpl<T>& left, const ManifoldPtr<T>& right)
  {
    using out_t = CartesianProductTpl<T>;
    return out_t(std::make_shared<out_t>(left), right);
  }

  
} // namespace proxnlp

