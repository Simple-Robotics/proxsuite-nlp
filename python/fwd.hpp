#pragma once

#include "lienlp/python/context.hpp"

#include <eigenpy/eigenpy.hpp>


namespace lienlp {

namespace python {
  namespace bp = boost::python;

  void exposeManifold();
  void exposeCost();
  void exposeResidual();
  void exposeProblem();
  void exposeResults();

} // namespace python

} // namespace lienlp

