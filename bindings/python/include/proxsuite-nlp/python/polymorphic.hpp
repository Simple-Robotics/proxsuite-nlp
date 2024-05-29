#pragma once
#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/third-party/polymorphic_cxx14.hpp"
#include <eigenpy/utils/traits.hpp>

namespace proxsuite::nlp {
namespace python {
namespace detail {
template <class held_type> struct PolymorphicHolder;

/// @brief Owning holder for the xyz::polymorphic<T, A> polymorphic value type.
template <class _T, class _A>
struct PolymorphicHolder<polymorphic<_T, _A>> : bp::instance_holder {
  static_assert(std::is_polymorphic_v<_T>, "Held type should be polymorphic.");
  using polymorphic_t = polymorphic<_T, _A>;
  using value_type = _T;
  using allocator_type = _A;

  PolymorphicHolder(polymorphic_t obj) : m_p(std::move(obj)) {}

  template <typename U>
  PolymorphicHolder(PyObject *, U u) : m_p(std::move(u)) {}

private:
  void *holds(bp::type_info dst_t, bool /*null_ptr_only*/) override {
    if (dst_t == bp::type_id<polymorphic_t>()) {
      // return pointer to the held data
      return &this->m_p;
    }
    value_type *p = boost::get_pointer(m_p);
    if (dst_t == bp::type_id<value_type>())
      return p;

    bp::type_info src_t = bp::type_id<value_type>();
    return src_t == dst_t ? p : bp::objects::find_dynamic_type(p, src_t, dst_t);
  }

  polymorphic_t m_p;
};

struct make_polymorphic_reference_holder {
  template <class T, class A>
  static PyObject *execute(const polymorphic<T, A> &p) {
    using pointer_holder = bp::objects::pointer_holder<T *, T>;
    T *q = const_cast<T *>(boost::get_pointer(p));
    return bp::objects::make_ptr_instance<T, pointer_holder>::execute(q);
  }
};

} // namespace detail

/// @brief Expose a polymorphic value type, e.g. xyz::polymorphic<T, A>.
template <class Poly> void register_polymorphic_to_python() {
  using value_type = typename Poly::value_type;
  bp::objects::class_value_wrapper<
      Poly, bp::objects::make_ptr_instance<value_type,
                                           detail::PolymorphicHolder<Poly>>>();
}

} // namespace python
} // namespace proxsuite::nlp

namespace boost {
namespace python {

/// Use the same trick from <eigenpy/eigen-to-python.hpp> to specialize the
/// template for both const and non-const
template <class X, class MakeHolder> struct to_python_indirect_poly {
  using poly_type = boost::remove_cv_ref_t<X>;
  template <class U> PyObject *operator()(U const &x) const {
    if (x.valueless_after_move()) {
      return detail::none();
    }
    return ::proxsuite::nlp::python::detail::make_polymorphic_reference_holder::
        execute(const_cast<U &>(x));
  }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
  PyTypeObject const *get_pytype() {
    return converter::registered_pytype<poly_type>::get_pytype();
  }
#endif
};

template <class T, class A, class MakeHolder>
struct to_python_indirect<xyz::polymorphic<T, A> &, MakeHolder>
    : to_python_indirect_poly<xyz::polymorphic<T, A> &, MakeHolder> {};

template <class T, class A, class MakeHolder>
struct to_python_indirect<const xyz::polymorphic<T, A> &, MakeHolder>
    : to_python_indirect_poly<const xyz::polymorphic<T, A> &, MakeHolder> {};

} // namespace python
} // namespace boost
