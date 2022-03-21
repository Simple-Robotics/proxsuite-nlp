#pragma once

#include "lienlp/python/context.hpp"

#include <eigenpy/eigenpy.hpp>


namespace lienlp {

namespace python {
  namespace bp = boost::python;

  void exposeManifold();
  void exposeProblem();
  void exposeResidual();
  void exposeResults();

} // namespace python

} // namespace lienlp

