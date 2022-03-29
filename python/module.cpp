#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

#include "lienlp/python/fwd.hpp"
#include "lienlp/python/util.hpp"

#include <eigenpy/eigenpy.hpp>


using namespace lienlp::python;


/// Expose some useful container types
void exposeContainerTypes()
{
  namespace pp = pinocchio::python;

  pp::StdVectorPythonVisitor<std::vector<int>, true>::expose("StdVec_int");
  pp::StdVectorPythonVisitor<context::VectorOfVectors, false>::expose("StdVec_Vector");
  pp::StdVectorPythonVisitor<std::vector<context::VectorXBool>, false>::expose("StdVec_VecBool");
}


BOOST_PYTHON_MODULE(pylienlp)
{
  bp::docstring_options module_docstring_options(true, true, true);

  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<context::VectorXBool>();

  bp::import("warnings");

  exposeFunctorTypes();
  exposeContainerTypes();
  {
    bp::scope man_scope = get_namespace("manifolds");
    exposeManifold();
  }
  {
    bp::scope res_cope = get_namespace("residuals");
    exposeResiduals();
  }
  {
    bp::scope cost_scope = get_namespace("costs");
    exposeCost();
  }
  {
    bp::scope cstr_scope = get_namespace("constraints");
    exposeConstraints();
  }
  exposeProblem();
  exposeResults();
  exposeWorkspace();
  exposeSolver();
}
