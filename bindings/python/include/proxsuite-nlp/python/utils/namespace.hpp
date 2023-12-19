#pragma once

#include <boost/python.hpp>

namespace proxsuite {
namespace nlp {
namespace python {
namespace bp = boost::python;

inline std::string get_scope_name(bp::scope scope) {
  return std::string(bp::extract<const char *>(scope.attr("__name__")));
}

/**
 * @brief   Create or retrieve a Python scope (that is, a class or module
 * namespace).
 *
 * @returns The submodule with the input name.
 */
inline bp::object get_namespace(const std::string &name) {
  bp::scope cur_scope; // current scope
  const std::string complete_name = get_scope_name(cur_scope) + "." + name;
  bp::object submodule(bp::borrowed(PyImport_AddModule(complete_name.c_str())));
  cur_scope.attr(name.c_str()) = submodule;
  return submodule;
}

} // namespace python

} // namespace nlp
} // namespace proxsuite
