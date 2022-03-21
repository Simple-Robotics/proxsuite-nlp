#include "lienlp/python/fwd.hpp"
#include <eigenpy/eigenpy.hpp>

using namespace lienlp::python;

BOOST_PYTHON_MODULE(pylienlp)
{
  bp::docstring_options module_docstring_options(true, true, false);

  eigenpy::enableEigenPy();

  bp::import("warnings");

  exposeManifold();
  exposeProblem();
  exposeResidual();
  exposeResults();
}
