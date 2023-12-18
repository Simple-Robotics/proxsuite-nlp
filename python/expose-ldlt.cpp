#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/python/policies.hpp"

#include "proxsuite-nlp/ldlt-allocator.hpp"

namespace proxnlp {
namespace python {

template <class LDLTtype>
struct LDLTVisitor : bp::def_visitor<LDLTVisitor<LDLTtype>> {

  static LDLTtype &compute_proxy(LDLTtype &fac,
                                 const context::ConstMatrixRef &mat) {
    return fac.compute(mat);
  }

  static bool solveInPlace_proxy(const LDLTtype &fac,
                                 context::MatrixRef rhsAndX) {
    return fac.solveInPlace(rhsAndX);
  }

  template <typename... Args> void visit(bp::class_<Args...> &cl) const {
    cl.def("compute", compute_proxy, policies::return_internal_reference,
           ("self"_a, "mat"))
        .def("solveInPlace", solveInPlace_proxy, ("self"_a, "rhsAndX"))
        .def("matrixLDLT", &LDLTtype::matrixLDLT, policies::return_by_value,
             "self"_a,
             "Get the current value of the decomposition matrix. This makes a "
             "copy.");
  }
};

void exposeLdltRoutines() {
  using context::Scalar;
  using context::VectorXs;

  using DenseLDLT = linalg::DenseLDLT<Scalar>;
  bp::class_<DenseLDLT>("DenseLDLT", bp::no_init).def(LDLTVisitor<DenseLDLT>());

  using BunchKaufman_t = Eigen::BunchKaufman<context::MatrixXs, Eigen::Lower>;
  bp::class_<BunchKaufman_t>("BunchKaufman", bp::no_init)
      .def(bp::init<>("self"_a))
      .def(LDLTVisitor<BunchKaufman_t>())
      .def("pivots", &BunchKaufman_t::pivots, bp::return_internal_reference<>())
      .def(
          "subdiag",
          +[](const BunchKaufman_t &bk) -> VectorXs { return bk.subdiag(); });

  using BlockLDLT = linalg::BlockLDLT<Scalar>;
  bp::class_<BlockLDLT>("BlockLDLT", bp::no_init)
      .def(LDLTVisitor<BlockLDLT>())
      .def("print_sparsity", &BlockLDLT::print_sparsity, "self"_a,
           "Print the sparsity pattern of the matrix to factorize.");
#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT
  using ProxSuiteLDLT = linalg::ProxSuiteLDLTWrapper<Scalar>;
  bp::class_<ProxSuiteLDLT>(
      "ProxSuiteLDLT", "Wrapper around ProxSuite's custom LDLT.", bp::no_init)
      .def(LDLTVisitor<ProxSuiteLDLT>());
#endif
}

} // namespace python
} // namespace proxnlp
