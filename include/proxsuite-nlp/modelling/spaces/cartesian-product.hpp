#pragma once

#include "proxsuite-nlp/manifold-base.hpp"
#include "proxsuite-nlp/third-party/polymorphic_cxx14.hpp"

#include <type_traits>

namespace proxsuite {
namespace nlp {
namespace {
using xyz::polymorphic;
/// Typedef in anon namespace for use in rest of file.
template <typename T>
using PolymorphicManifold = polymorphic<ManifoldAbstractTpl<T>>;
} // namespace

/** @brief    The cartesian product of two or more manifolds.
 */
template <typename _Scalar>
struct CartesianProductTpl : ManifoldAbstractTpl<_Scalar> {
  using Scalar = _Scalar;

  using Base = ManifoldAbstractTpl<Scalar>;
  PROXSUITE_NLP_DEFINE_MANIFOLD_TYPES(Base)

private:
  std::vector<polymorphic<Base>> m_components;

public:
  const Base &getComponent(std::size_t i) const { return *m_components[i]; }

  inline std::size_t numComponents() const { return m_components.size(); }

  template <class Concrete> inline void addComponent(const Concrete &c) {
    m_components.emplace_back(c);
  }

  inline void addComponent(const CartesianProductTpl &other) {
    for (const auto &c : other.m_components) {
      this->addComponent(c);
    }
  }

  CartesianProductTpl() {}

  CartesianProductTpl(const std::vector<polymorphic<Base>> &components)
      : m_components(components) {}

  CartesianProductTpl(
      const std::initializer_list<polymorphic<Base>> &components)
      : m_components(components) {}

  CartesianProductTpl(const polymorphic<Base> &left,
                      const polymorphic<Base> &right) {
    addComponent(left);
    addComponent(right);
  }

  inline int nx() const {
    int r = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      r += m_components[i]->nx();
    }
    return r;
  }

  inline int ndx() const {
    int r = 0;
    for (std::size_t i = 0; i < numComponents(); i++) {
      r += m_components[i]->ndx();
    }
    return r;
  }

  PointType neutral() const;
  PointType rand() const;
  bool isNormalized(const ConstVectorRef &x) const;

private:
  template <class VectorType, class U = std::remove_const_t<VectorType>>
  std::vector<U> split_impl(VectorType &x) const;

  template <class VectorType, class U = std::remove_const_t<VectorType>>
  std::vector<U> split_vector_impl(VectorType &v) const;

public:
  std::vector<VectorRef> split(VectorRef x) const {
    return split_impl<VectorRef>(x);
  }

  std::vector<ConstVectorRef> split(const ConstVectorRef &x) const {
    return split_impl<const ConstVectorRef>(x);
  }

  std::vector<VectorRef> split_vector(VectorRef v) const {
    return split_vector_impl<VectorRef>(v);
  }

  std::vector<ConstVectorRef> split_vector(const ConstVectorRef &v) const {
    return split_vector_impl<const ConstVectorRef>(v);
  }

  PointType merge(const std::vector<VectorXs> &xs) const;

  TangentVectorType merge_vector(const std::vector<VectorXs> &vs) const;

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const;

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const;

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const;

  void JintegrateTransport(const ConstVectorRef &x, const ConstVectorRef &v,
                           MatrixRef Jout, int arg) const;

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const;
};

template <typename T>
auto operator*(const PolymorphicManifold<T> &left,
               const PolymorphicManifold<T> &right) {
  return CartesianProductTpl<T>(left, right);
}

template <typename T>
auto operator*(const CartesianProductTpl<T> &left,
               const PolymorphicManifold<T> &right) {
  CartesianProductTpl<T> out(left);
  out.addComponent(right);
  return out;
}

template <typename T>
auto operator*(const PolymorphicManifold<T> &left,
               const CartesianProductTpl<T> &right) {
  return right * left;
}

template <typename T>
auto operator*(const CartesianProductTpl<T> &left,
               const CartesianProductTpl<T> &right) {
  CartesianProductTpl<T> out{left};
  out.addComponent(right);
  return out;
}

} // namespace nlp
} // namespace proxsuite

// implementation details
#include "proxsuite-nlp/modelling/spaces/cartesian-product.hxx"

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/modelling/spaces/cartesian-product.txx"
#endif
