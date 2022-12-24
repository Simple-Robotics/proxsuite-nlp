// #include <pinocchio/fwd.hpp>
// #include <pinocchio/bindings/python/utils/std-vector.hpp>
#include <eigenpy/std-vector.hpp>

#include "proxnlp/python/fwd.hpp"
#include "proxnlp/python/utils/namespace.hpp"
#include "proxnlp/version.hpp"

namespace proxnlp {
namespace python {
/// Expose some useful container types
void exposeContainerTypes() {
  // using pinocchio::python::StdVectorPythonVisitor;
  using eigenpy::StdVectorPythonVisitor;

  StdVectorPythonVisitor<std::vector<int>, true>::expose("StdVec_int");
  StdVectorPythonVisitor<std::vector<context::Scalar>, true>::expose(
      "StdVec_Scalar");
  StdVectorPythonVisitor<context::VectorOfVectors, true>::expose(
      "StdVec_Vector");
  StdVectorPythonVisitor<std::vector<context::MatrixXs>, true>::expose(
      "StdVec_Matrix");
  StdVectorPythonVisitor<std::vector<context::VectorXBool>, false>::expose(
      "StdVec_VecBool");
  StdVectorPythonVisitor<context::VectorOfRef, true>::expose("StdVec_VecRef");
  StdVectorPythonVisitor<std::vector<context::MatrixRef>, true>::expose(
      "StdVec_MatRef");
}

} // namespace python
} // namespace proxnlp

BOOST_PYTHON_MODULE(pyproxnlp) {
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
    exposeManifolds();
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
