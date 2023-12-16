#pragma once

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

private:
  std::vector<shared_ptr<Base>> components;

public:
  const Base &getComponent(std::size_t i) const { return *components[i]; }

  inline std::size_t numComponents() const { return components.size(); }

  inline void addComponent(const shared_ptr<Base> &c) {
    components.push_back(c);
  }

  inline void addComponent(const shared_ptr<CartesianProductTpl> &other) {
    for (const auto &c : other->components) {
      this->addComponent(c);
    }
  }

  CartesianProductTpl() {}

  CartesianProductTpl(const std::vector<shared_ptr<Base>> &components)
      : components(components) {}

  CartesianProductTpl(const std::initializer_list<shared_ptr<Base>> &components)
      : components(components) {}

  CartesianProductTpl(const shared_ptr<Base> &left,
                      const shared_ptr<Base> &right) {
    addComponent(left);
    addComponent(right);
  }

  template <typename U, typename V>
  CartesianProductTpl(const U &left, const V &right) {
    static_assert(!(std::is_pointer<U>::value || std::is_pointer<V>::value),
                  "Ctor operators on non-pointer types.");
    addComponent(std::make_shared<U>(left));
    addComponent(std::make_shared<V>(right));
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
auto operator*(const ManifoldPtr<T> &left, const ManifoldPtr<T> &right) {
  return std::make_shared<CartesianProductTpl<T>>(left, right);
}

template <typename T>
auto operator*(const shared_ptr<CartesianProductTpl<T>> &left,
               const ManifoldPtr<T> &right) {
  auto out = std::make_shared<CartesianProductTpl<T>>(*left);
  out->addComponent(right);
  return out;
}

template <typename T>
auto operator*(const ManifoldPtr<T> &left,
               const shared_ptr<CartesianProductTpl<T>> &right) {
  auto out = std::make_shared<CartesianProductTpl<T>>();
  out->addComponent(left);
  out->addComponent(right);
  return out;
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

// implementation details
#include "proxnlp/modelling/spaces/cartesian-product.hxx"

#ifdef PROXNLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxnlp/modelling/spaces/cartesian-product.txx"
#endif
