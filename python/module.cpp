#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

#include "proxnlp/python/fwd.hpp"
#include "proxnlp/python/util.hpp"
#include "proxnlp/version.hpp"


namespace proxnlp
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
    pp::StdVectorPythonVisitor<std::vector<context::MatrixXs>, true>::expose("StdVec_Matrix");
    pp::StdVectorPythonVisitor<std::vector<context::VectorXBool>, false>::expose("StdVec_VecBool");
    pp::StdVectorPythonVisitor<context::VectorOfRef, true>::expose("StdVec_VecRef");
    pp::StdVectorPythonVisitor<std::vector<context::MatrixRef>, true>::expose("StdVec_MatRef");
  }

}
}


BOOST_PYTHON_MODULE(pyproxnlp)
{
  using namespace proxnlp::python;

  bp::docstring_options module_docstring_options(true, true, true);

  bp::scope().attr("__version__") = proxnlp::printVersion();
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
  {
    bp::scope in_scope = get_namespace("helpers");
    exposeCallbacks();
  }
  {
    bp::scope autodiff_scope = get_namespace("autodiff");
    exposeAutodiff();
  }
}
