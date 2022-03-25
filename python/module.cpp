#include "lienlp/python/fwd.hpp"
#include <eigenpy/eigenpy.hpp>


using namespace lienlp::python;

BOOST_PYTHON_MODULE(pylienlp)
{
  bp::docstring_options module_docstring_options(true, true, true);

  eigenpy::enableEigenPy();

  bp::import("warnings");

  exposeManifold();
  exposeFunctorTypes();
  exposeResidual();
  exposeCost();
  exposeConstraint();
  exposeProblem();
  exposeResults();
}
