#include <boost/python/return_value_policy.hpp>
#include <boost/python/return_by_value.hpp>
#include <boost/python/return_internal_reference.hpp>

#pragma once

namespace proxsuite {
namespace nlp {
namespace python {

namespace bp = boost::python;

namespace policies {
constexpr auto return_by_value = bp::return_value_policy<bp::return_by_value>();
constexpr auto return_internal_reference = bp::return_internal_reference<>();
} // namespace policies

} // namespace python
} // namespace nlp
} // namespace proxsuite
