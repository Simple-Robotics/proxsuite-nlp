#pragma once

#include "lienlp/python/context.hpp"

#include <eigenpy/eigenpy.hpp>


namespace lienlp {

namespace python {
  namespace bp = boost::python;

  void exposeManifold();
  void exposeResidual();
  void exposeCost();
  void exposeConstraint();
  void exposeProblem();
  void exposeResults();

} // namespace python

} // namespace lienlp

