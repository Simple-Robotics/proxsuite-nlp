#pragma once

#include "lienlp/python/context.hpp"

#include <eigenpy/eigenpy.hpp>


namespace lienlp
{

namespace python
{
  namespace bp = boost::python;

  void exposeManifold();
  void exposeFunctorTypes();
  /// Expose defined residuals for modelling
  void exposeResiduals();
  void exposeCost();
  void exposeConstraint();
  void exposeProblem();
  void exposeResults();

} // namespace python

} // namespace lienlp

