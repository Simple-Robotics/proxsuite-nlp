#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/modelling/spaces/vector-space.hpp"

#include "proxnlp/exceptions.hpp"

#include <type_traits>

namespace proxnlp {
namespace {
/// Typedef in anon namespace for use in rest of file.
template <typename T> using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<T>>;
} // namespace

/** @brief    The cartesian product of two or more manifolds.
 */
template <typename _Scalar>
struct CartesianProductTpl : ManifoldAbstractTpl<_Scalar> {
  using Scalar = _Scalar;

  using Base = ManifoldAbstractTpl<Scalar>;
  PROXNLP_DEFINE_MANIFOLD_TYPES(Base)

  std::vector<shared_ptr<Base>> components;

  const Base &getComponent(std::size_t i) const { return *components[i]; }

  inline std::size_t numComponents() const { return components.size(); }

  inline void addComponent(const shared_ptr<Base> &c) {
    components.push_back(c);
  }

  CartesianProductTpl() {}

  CartesianProductTpl(const std::vector<shared_ptr<Base>> &components)
      : components(components) {}

  CartesianProductTpl(const std::initializer_list<shared_ptr<Base>> &components)
      : components(components) {}

  CartesianProductTpl(const shared_ptr<Base> &left,
                      const shared_ptr<Base> &right) {
    components.push_back(left);
    components.push_back(right);
  }

  template <typename U, typename V>
  CartesianProductTpl(const U &left, const V &right) {
    static_assert(!std::is_pointer<U>::value, "U should be pointer type.");
    static_assert(!std::is_pointer<V>::value, "V should be pointer type.");
    components.push_back(std::make_shared<U>(left));
    components.push_back(std::make_shared<V>(right));
  }

  inline int nx() const {
    int r = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      r += components[i]->nx();
    }
    return r;
  }

  inline int ndx() const {
    int r = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      r += components[i]->ndx();
    }
    return r;
  }

  PointType neutral() const {
    PointType out(this->nx());
    Eigen::Index c = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long n = components[i]->nx();
      out.segment(c, n) = components[i]->neutral();
      c += n;
    }
    return out;
  }

  PointType rand() const {
    PointType out(this->nx());
    Eigen::Index c = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long n = components[i]->nx();
      out.segment(c, n) = components[i]->rand();
      c += n;
    }
    return out;
  }

  std::vector<VectorRef> split(VectorRef x) const {
    proxnlp_dim_check(x, this->nx());
    std::vector<VectorRef> out = {};
    Eigen::Index c = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long n = components[i]->nx();
      out.push_back(x.segment(c, n));
      c += n;
    }
    return out;
  }

  std::vector<VectorRef> split_vector(VectorRef v) const {
    proxnlp_dim_check(v, this->ndx());
    std::vector<VectorRef> out = {};
    Eigen::Index c = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long n = components[i]->ndx();
      out.push_back(v.segment(c, n));
      c += n;
    }
    return out;
  }

  PointType merge(const std::vector<VectorXs> &xs) const {
    PointType out(this->nx());
    Eigen::Index c = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long n = components[i]->nx();
      out.segment(c, n) = xs[i];
      c += n;
    }
    return out;
  }

  TangentVectorType merge_vector(const std::vector<VectorXs> &vs) const {
    TangentVectorType out(this->ndx());
    Eigen::Index c = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long n = components[i]->ndx();
      out.segment(c, n) = vs[i];
      c += n;
    }
    return out;
  }

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const {
    assert(ndx() == out.size());
    Eigen::Index cq = 0, cv = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long nq = getComponent(i).nx();
      const long nv = getComponent(i).ndx();
      auto sx = x.segment(cq, nq);
      auto sv = v.segment(cv, nv);

      auto sout = out.segment(cq, nq);

      getComponent(i).integrate(sx, sv, sout);
      cq += nq;
      cv += nv;
    }
  }

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const {
    assert(ndx() == out.size());
    Eigen::Index cq = 0, cv = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long nq = getComponent(i).nx();
      const long nv = getComponent(i).ndx();
      auto sx0 = x0.segment(cq, nq);
      auto sx1 = x1.segment(cq, nq);

      auto sout = out.segment(cv, nv);

      getComponent(i).difference(sx0, sx1, sout);
      cq += nq;
      cv += nv;
    }
  }

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const {
    assert(ndx() == Jout.rows());
    Jout.setZero();
    Eigen::Index cq = 0, cv = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long nq = getComponent(i).nx();
      const long nv = getComponent(i).ndx();
      auto sx = x.segment(cq, nq);
      auto sv = v.segment(cv, nv);

      auto sJout = Jout.block(cv, cv, nv, nv);

      getComponent(i).Jintegrate(sx, sv, sJout, arg);
      cq += nq;
      cv += nv;
    }
  }

  void JintegrateTransport(const ConstVectorRef &x, const ConstVectorRef &v,
                           MatrixRef Jout, int arg) const {
    Eigen::Index cq = 0, cv = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long nq = components[i]->nx();
      auto sx = x.segment(cq, nq);
      cq += nq;

      const long nv = components[i]->ndx();
      auto sv = v.segment(cv, nv);
      auto sJout = Jout.middleRows(cv, nv);
      cv += nv;

      getComponent(i).JintegrateTransport(sx, sv, sJout, arg);
    }
    // const int nx1 = left().nx();
    // const int nx2 = right().nx();
    // const int ndx1 = left().ndx();
    // const int ndx2 = right().ndx();
    // left().JintegrateTransport(x.head(nx1), v.head(ndx1), Jout.topRows(ndx1),
    //                            arg);
    // right().JintegrateTransport(x.tail(nx2), v.head(ndx2),
    //                             Jout.bottomRows(ndx2), arg);
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const {
    assert(ndx() == Jout.rows());
    Jout.setZero();
    Eigen::Index cq = 0, cv = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      const long nq = components[i]->nx();
      auto sx0 = x0.segment(cq, nq);
      auto sx1 = x1.segment(cq, nq);
      cq += nq;

      const long nv = components[i]->ndx();
      auto sJout = Jout.block(cv, cv, nv, nv);
      cv += nv;

      getComponent(i).Jdifference(sx0, sx1, sJout, arg);
    }
    // const int nx1 = left().nx();
    // const int nx2 = right().nx();
    // const int ndx1 = left().ndx();
    // const int ndx2 = right().ndx();
    // left().Jdifference(x0.head(nx1), x1.head(nx1),
    //                    Jout.topLeftCorner(ndx1, ndx1), arg);
    // right().Jdifference(x0.tail(nx2), x1.tail(nx2),
    //                     Jout.bottomRightCorner(ndx2, ndx2), arg);
  }
};

/// Direct product of two manifolds as a cartesian product.
// template<typename U, typename V>
// CartesianProductTpl<typename U::Scalar> operator*(const U& left, const V&
// right)
// {
//   using T = typename U::Scalar;
//   static_assert(std::is_same<T, typename V::Scalar>::value, "Both arguments
//   should have same Scalar template arg!"); return
//   CartesianProductTpl<T>(std::make_shared<U>(left),
//   std::make_shared<V>(right));
// }

template <typename T>
CartesianProductTpl<T> operator*(const ManifoldPtr<T> &left,
                                 const ManifoldPtr<T> &right) {
  return CartesianProductTpl<T>({left, right});
}

template <typename T>
CartesianProductTpl<T> operator*(const shared_ptr<CartesianProductTpl<T>> &left,
                                 const ManifoldPtr<T> &right) {
  return (*left) * right;
}

template <typename T>
CartesianProductTpl<T> operator*(const CartesianProductTpl<T> &left,
                                 const ManifoldPtr<T> &right) {
  CartesianProductTpl<T> out(left);
  out.addComponent(right);
  return out;
}

template <typename T>
CartesianProductTpl<T> operator*(const ManifoldPtr<T> &left,
                                 const CartesianProductTpl<T> &right) {
  return right * left;
}

} // namespace proxnlp
