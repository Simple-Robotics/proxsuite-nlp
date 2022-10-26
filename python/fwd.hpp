#pragma once

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>

#ifdef byte
#undef byte
#endif
#include "proxnlp/python/context.hpp"

namespace proxnlp {

/// @brief Python bindings.
namespace python {
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
void exposeAutodiff();

} // namespace python

} // namespace proxnlp
