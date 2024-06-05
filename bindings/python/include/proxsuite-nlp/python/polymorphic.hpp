#pragma once
#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/third-party/polymorphic_cxx14.hpp"
#include <eigenpy/utils/traits.hpp>

// Required class template specialization for
// boost::python::register_ptr_to_python<> to work.
namespace boost::python {
template <class T, class A> struct pointee<xyz::polymorphic<T, A>> {
  typedef T type;
};
} // namespace boost::python

namespace proxsuite::nlp {
namespace python {
namespace detail {

struct make_polymorphic_reference_holder {
  template <class T, class A>
  static PyObject *execute(const polymorphic<T, A> &p) {
    if (p.valueless_after_move())
      return bp::detail::none();
    T *q = const_cast<T *>(boost::get_pointer(p));
    assert(q);
    // C++17 for the win
    if constexpr (std::is_polymorphic_v<T>) {
      // Detect if the polymorphic<T> object contains a
      // boost::python::wrapper<T> callback class. If so, extract the owning
      // PyObject* and return it.
      if (bp::detail::wrapper_base *t =
              dynamic_cast<bp::detail::wrapper_base *>(q)) {
        // The following call extracts the underlying PyObject.
        // see <boost/python/to_python_indirect.hpp>, execute() overload
        PyObject *o = bp::detail::wrapper_base_::get_owner(*t);
        assert(o);
        if (o)
          return bp::incref(o);
      }
    }
    using pointer_holder = bp::objects::pointer_holder<T *, T>;
    return bp::objects::make_ptr_instance<T, pointer_holder>::execute(q);
  }
};

} // namespace detail

/// @brief Expose a polymorphic value type, e.g. xyz::polymorphic<T, A>.
/// @details Just an alias for bp::register_ptr_to_python<>().
template <class Poly> inline void register_polymorphic_to_python() {
  using X = typename bp::pointee<Poly>::type;
  bp::objects::class_value_wrapper<
      Poly, bp::objects::make_ptr_instance<
                X, bp::objects::pointer_holder<Poly, X>>>();
}

template <class Poly> struct PolymorphicVisitor;

/// Does the same thing as boost::python::implicitly_convertible<>(),
/// except the conversion is placed at the top of the conversion chain.
/// This ensures that Boost.Python attempts to convert to Poly BEFORE
/// any parent class!
template <class Base, class A>
struct PolymorphicVisitor<xyz::polymorphic<Base, A>>
    : bp::def_visitor<PolymorphicVisitor<xyz::polymorphic<Base, A>>> {
  using Poly = xyz::polymorphic<Base, A>;
  static_assert(std::is_polymorphic_v<Base>, "Type should be polymorphic!");

  template <class PyClass> void visit(PyClass &) const {
    using T = typename PyClass::wrapped_type;
    using meta = typename PyClass::metadata;
    using held = typename meta::held_type;
    typedef bp::converter::implicit<held, Poly> functions;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-abstract-non-virtual-dtor"
    bp::converter::registry::insert(
        &functions::convertible, &functions::construct, bp::type_id<Poly>(),
        &bp::converter::expected_from_python_type_direct<T>::get_pytype);
  }
#pragma GCC diagnostic pop
};

} // namespace python
} // namespace proxsuite::nlp

namespace boost {
namespace python {

/// Use the same trick from <eigenpy/eigen-to-python.hpp> to specialize the
/// template for both const and non-const
template <class X, class MakeHolder> struct to_python_indirect_poly {
  using poly_type = boost::remove_cv_ref_t<X>;
  template <class U> PyObject *operator()(U const &x) const {
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
