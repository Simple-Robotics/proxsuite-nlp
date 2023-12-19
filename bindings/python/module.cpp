#include <eigenpy/std-vector.hpp>

#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/python/utils/namespace.hpp"
#include "proxsuite-nlp/version.hpp"

namespace context = proxsuite::nlp::context;

namespace proxsuite {
namespace nlp {
namespace python {
/// Expose some useful container types
void exposeContainerTypes() {
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
} // namespace nlp
} // namespace proxsuite

BOOST_PYTHON_MODULE(MODULE_NAME) {
  using namespace proxsuite::nlp::python;

  bp::docstring_options module_docstring_options(true, true, true);

  bp::scope().attr("__version__") = proxsuite::nlp::printVersion();
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<context::VectorXBool>();

  bp::import("warnings");
#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
  bp::import("pinocchio");
#endif

  exposeContainerTypes();
  exposeFunctionTypes();
  {
    bp::scope man_scope = get_namespace("manifolds");
    exposeManifolds();
  }
  {
    bp::scope res_cope = get_namespace("residuals");
    exposeResiduals();
#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
    exposePinocchioResiduals();
#endif
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
  exposeLdltRoutines();
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
