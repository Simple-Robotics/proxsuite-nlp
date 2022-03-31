#pragma once

#include "lienlp/python/context.hpp"

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>


namespace lienlp
{

namespace python
{
  namespace bp = boost::python;

  void exposeFunctionTypes();
  void exposeManifold();
  /// Expose defined residuals for modelling
  void exposeResiduals();
  void exposeCost();
  void exposeConstraints();
  void exposeProblem();
  void exposeResults();
  void exposeWorkspace();
  void exposeSolver();
  void exposeCallbacks();

} // namespace python

} // namespace lienlp

