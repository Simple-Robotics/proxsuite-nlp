#include "proxnlp/python/fwd.hpp"
#include "proxnlp/python/policies.hpp"

#include "proxnlp/ldlt-allocator.hpp"

namespace proxnlp {
namespace python {

void exposeLdltRoutines() {
  using context::Scalar;
  using LDLTBase = linalg::ldlt_base<Scalar>;
  bp::class_<LDLTBase, boost::noncopyable>(
      "LDLTBase", "Base class for LDLT solvers.", bp::no_init)
      .def("compute", &LDLTBase::compute, policies::return_internal_reference,
           bp::args("self", "mat"))
      .def("solveInPlace", &LDLTBase::solveInPlace, bp::args("self", "rhsAndX"))
      .def("matrixLDLT", &LDLTBase::matrixLDLT, policies::return_by_value,
           "Get the current value of the decomposition matrix. This makes a "
           "copy.");
  bp::class_<linalg::DenseLDLT<Scalar>, bp::bases<LDLTBase>>("DenseLDLT",
                                                             bp::no_init);
  using BlockLDLT = linalg::BlockLDLT<Scalar>;
  bp::class_<BlockLDLT, bp::bases<LDLTBase>>("BlockLDLT", bp::no_init)
      .def("print_sparsity", &BlockLDLT::print_sparsity, bp::args("self"),
           "Print the sparsity pattern of the matrix to factorize.");
#ifdef PROXNLP_ENABLE_PROXSUITE_LDLT
  bp::class_<linalg::ProxSuiteLDLTWrapper<Scalar>, bp::bases<LDLTBase>>(
      "ProxSuiteLDLT", "Wrapper around ProxSuite's custom LDLT.", bp::no_init);
#endif
}

} // namespace python
} // namespace proxnlp
