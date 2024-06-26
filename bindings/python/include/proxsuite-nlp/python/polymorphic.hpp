#pragma once
#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/third-party/polymorphic_cxx14.hpp"
#include "proxsuite-nlp/macros.hpp"
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

    PROXSUITE_NLP_COMPILER_DIAGNOSTIC_PUSH
    PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
    bp::converter::registry::insert(
        &functions::convertible, &functions::construct, bp::type_id<Poly>(),
        &bp::converter::expected_from_python_type_direct<T>::get_pytype);
    PROXSUITE_NLP_COMPILER_DIAGNOSTIC_POP
  }
};

/// Like boost::python::with_custodian_and_ward but the ward must be a list
/// and with_custodian_and_ward memory management will be applied on each
/// element of contained in the ward.
template <std::size_t custodian, std::size_t ward,
          class BasePolicy_ = bp::default_call_policies>
struct with_custodian_and_ward_list_content : BasePolicy_ {
  BOOST_STATIC_ASSERT(custodian != ward);
  BOOST_STATIC_ASSERT(custodian > 0);
  BOOST_STATIC_ASSERT(ward > 0);

  template <class ArgumentPackage>
  static bool precall(ArgumentPackage const &args_) {
    unsigned arity_ = bp::detail::arity(args_);
    if (custodian > arity_ || ward > arity_) {
      PyErr_SetString(
          PyExc_IndexError,
          "with_custodian_and_ward_list_content: argument index out of range");
      return false;
    }

    PyObject *patient = bp::detail::get_prev<ward>::execute(args_);
    PyObject *nurse = bp::detail::get_prev<custodian>::execute(args_);

    // Check if patient is a list
    if (!PyList_Check(patient)) {
      PyErr_SetString(
          PyExc_TypeError,
          "with_custodian_and_ward_list_content: ward must be a list");
      return false;
    }

    // Retrieve the underlying list.
    // TODO I'm not sure the bp::handle is mandatory here.
    // I don't think we will call Python interpreter some how and
    // delete the patient reference (See
    // https://docs.python.org/3/extending/extending.html#thin-ice).
    bp::object patient_obj(bp::handle<>(bp::borrowed(patient)));
    bp::list patient_list(patient_obj);

    // Add life_support for each element of the patient_list
    bp::ssize_t list_size = bp::len(patient_list);
    std::vector<PyObject *> life_support_list;
    life_support_list.reserve(list_size);
    for (bp::ssize_t k = 0; k < list_size; ++k) {
      auto l = patient_list[k];
      PyObject *life_support = bp::objects::make_nurse_and_patient(
          nurse, bp::object(patient_list[k]).ptr());
      if (life_support == 0) {
        for (PyObject *ls : life_support_list) {
          Py_DECREF(ls);
        }
        return false;
      }
      life_support_list.push_back(life_support);
    }

    bool result = BasePolicy_::precall(args_);

    if (!result) {
      for (PyObject *ls : life_support_list) {
        Py_DECREF(ls);
      }
    }

    return result;
  }
};

} // namespace python
} // namespace proxsuite::nlp

namespace boost {
namespace python {

/// Use the same trick from <eigenpy/eigen-to-python.hpp> to specialize the
/// template for both const and non-const.
/// This specialization do the same thing than to_python_indirect with the
/// following differences:
/// - Return the content of the xyz::polymorphic
/// - Don't incref owner Python object if the xyz::polymorphic content inherit
///   of bp::wrapper to avoid memory leak
/// \warning This converter should only be called by
/// bp::return_internal_reference, because return_internal_reference will link
/// returned object lifetime to the owner object lifetime.
template <class poly_ref, class MakeHolder> struct to_python_indirect_poly {
  using poly_type = remove_cv_ref_t<poly_ref>;
  template <class U> PyObject *operator()(U const &ref) const {
    return this->execute(const_cast<U &>(ref), detail::is_pointer<U>());
  }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
  PyTypeObject const *get_pytype() {
    return converter::registered_pytype<poly_type>::get_pytype();
  }
#endif

private:
  template <class T, class A>
  inline PyObject *execute(xyz::polymorphic<T, A> *ptr, detail::true_) const {
    // No special NULL treatment for references
    if (ptr == 0)
      return detail::none();
    else
      return this->execute(*ptr, detail::false_());
  }

  template <class T, class A>
  inline PyObject *execute(xyz::polymorphic<T, A> const &x,
                           detail::false_) const {
    if (x.valueless_after_move())
      return detail::none();
    T *const p = const_cast<T *>(boost::get_pointer(x));
    if (PyObject *o = detail::wrapper_base_::owner(p)) {
      // to_python_indirect call incref on o, but this create a memory leak
      // when used with bp::return_internal_reference.
      return o;
    }
    return MakeHolder::execute(p);
  }
};

template <class T, class A, class MakeHolder>
struct to_python_indirect<xyz::polymorphic<T, A> &, MakeHolder>
    : to_python_indirect_poly<xyz::polymorphic<T, A> &, MakeHolder> {};

template <class T, class A, class MakeHolder>
struct to_python_indirect<const xyz::polymorphic<T, A> &, MakeHolder>
    : to_python_indirect_poly<const xyz::polymorphic<T, A> &, MakeHolder> {};

} // namespace python
} // namespace boost
