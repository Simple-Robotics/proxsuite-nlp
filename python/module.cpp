#include "lienlp/python/fwd.hpp"
#include "lienlp/python/util.hpp"

#include <eigenpy/eigenpy.hpp>


using namespace lienlp::python;

BOOST_PYTHON_MODULE(pylienlp)
{
  bp::docstring_options module_docstring_options(true, true, true);

  eigenpy::enableEigenPy();

  bp::import("warnings");

  {
    bp::scope man_scope = get_namespace("manifolds");
    exposeManifold();
  }
  exposeFunctorTypes();
  exposeResiduals();
  exposeCost();
  exposeConstraint();
  exposeProblem();
  exposeResults();
}
