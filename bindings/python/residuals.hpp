#include "proxsuite-nlp/python/fwd.hpp"

namespace proxsuite {
namespace nlp {
namespace python {

/// Expose a differentiable residual (subclass of C2FunctionTpl).
template <typename T, class Init>
auto expose_function(const char *name, const char *docstring, Init init) {
  return bp::class_<T, bp::bases<context::C2Function>>(name, docstring, init);
}

} // namespace python
} // namespace nlp
} // namespace proxsuite
