#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

#include "lienlp/python/fwd.hpp"
#include "lienlp/python/util.hpp"
#include "lienlp/version.hpp"

#include <eigenpy/eigenpy.hpp>


using namespace lienlp::python;


namespace lienlp
{
namespace python
{
  /// Expose some useful container types
  void exposeContainerTypes()
  {
    namespace pp = pinocchio::python;

    pp::StdVectorPythonVisitor<std::vector<int>, true>::expose("StdVec_int");
    pp::StdVectorPythonVisitor<std::vector<context::Scalar>, true>::expose("StdVec_Scalar");
    pp::StdVectorPythonVisitor<context::VectorOfVectors, true>::expose("StdVec_Vector");
    pp::StdVectorPythonVisitor<context::VectorOfRef, true>::expose("StdVec_VecRef");
    pp::StdVectorPythonVisitor<std::vector<context::VectorXBool>, false>::expose("StdVec_VecBool");
  }

}
}


BOOST_PYTHON_MODULE(pylienlp)
{
  bp::docstring_options module_docstring_options(true, true, true);

  bp::scope().attr("__version__") = lienlp::printVersion();
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<context::VectorXBool>();

  bp::import("warnings");

  exposeContainerTypes();
  exposeFunctionTypes();
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
  exposeCallbacks();
}
