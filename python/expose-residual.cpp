#include "lienlp/python/fwd.hpp"
#include "lienlp/residual-base.hpp"

namespace lienlp {
namespace python {
  namespace bp = boost::python;

  void exposeResidual()
  {
    using Residual_t = ResidualBase<context::Scalar>;
    bp::class_<Residual_t, boost::noncopyable>("ResidualBase", bp::no_init)
      .def("__call__", &Residual_t::operator());
  }

} // namespace python
} // namespace lienlp
