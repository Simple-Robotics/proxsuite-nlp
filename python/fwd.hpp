#pragma once
#include <eigenpy/eigenpy.hpp>


namespace lienlp {

namespace python {
  namespace bp = boost::python;

  void exposeManifold();
  void exposeResidual();
  void exposeResults();

} // namespace python

} // namespace lienlp

