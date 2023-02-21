/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include <proxsuite/linalg/dense/ldlt.hpp>

#include "proxnlp/linalg/ldlt-base.hpp"

namespace {

namespace veg = proxsuite::linalg::veg;
namespace psdense = proxsuite::linalg::dense;
using byte = proxsuite::linalg::veg::mem::byte;
using StackType = proxsuite::linalg::veg::Vec<byte>;
using StackReq = proxsuite::linalg::veg::dynstack::StackReq;

template <typename Mat, typename Rhs> void solve_impl(Mat &&ld_, Rhs &&rhs_) {
  auto ld = psdense::util::to_view(ld_);
  auto rhs = psdense::util::to_view_dyn_rows(rhs_);

  auto l = ld.template triangularView<Eigen::UnitLower>();
  auto lt =
      psdense::util::trans(ld).template triangularView<Eigen::UnitUpper>();
  auto d = psdense::util::diagonal(ld);
  l.solveInPlace(rhs);
  rhs = d.asDiagonal().inverse() * rhs;
  lt.solveInPlace(rhs);
}

} // anonymous namespace

namespace proxnlp {
namespace linalg {

/// @brief Use the LDLT from proxsuite.
template <typename Scalar> struct ProxSuiteLDLTWrapper : ldlt_base<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ldlt_base<Scalar>;
  using psldlt_t = psdense::Ldlt<Scalar>;
  using DView = typename Base::DView;

  psldlt_t m_ldlt;
  StackType ldl_stack;
  using Base::m_info;

  ProxSuiteLDLTWrapper(isize nr, isize rhs_max_cols) : m_ldlt{} {
    m_ldlt.reserve_uninit(nr);

    // quick runthrough: no block insertions, no diagonal updates
    // just need space for solve in place with nr rows, rhs_max_cols columns.
    auto req_mat = psldlt_t::solve_in_place_req(rhs_max_cols * nr);

    StackReq req{psldlt_t::factorize_req(nr) | req_mat};

    ldl_stack.resize_for_overwrite(req.alloc_req());
  }

  ProxSuiteLDLTWrapper &compute(const ConstMatrixRef &mat) {
    using veg::dynstack::DynStackMut;

    DynStackMut stack{veg::from_slice_mut, ldl_stack.as_mut()};

    m_ldlt.factorize(mat, stack);

    m_info = Eigen::ComputationInfo::Success;
    return *this;
  }

  bool solveInPlace(MatrixRef rhs) const {
    using veg::dynstack::DynStackMut;
    isize nrows = rhs.rows();
    isize ncols = rhs.cols();
    auto perm = m_ldlt.p();
    auto perm_inv = m_ldlt.pt();
    DynStackMut stack{veg::from_slice_mut,
                      const_cast<StackType &>(ldl_stack).as_mut()};
    LDLT_TEMP_MAT_UNINIT(Scalar, work, nrows, ncols, stack);

    // for (isize i = 0; i < nrows; ++i) {
    //   work.row(i) = rhs.row(perm[i]);
    // }
    work = perm_inv * rhs;

    solve_impl(m_ldlt.ld_col(), work);

    rhs = perm * work;

    // for (isize i = 0; i < nrows; ++i) {
    //   rhs.row(i) = work.row(perm_inv[i]);
    // }
    return true;
  }

  inline DView vectorD() const {
    return psdense::util::diagonal(m_ldlt.ld_col());
  }

  MatrixXs reconstructedMatrix() const {

    auto L = m_ldlt.l();
    auto D = m_ldlt.d();
    auto U = m_ldlt.lt();
    MatrixXs r = U;
    r = D.asDiagonal() * r;
    r = L * r;
    r = r * m_ldlt.pt();
    r = m_ldlt.p() * r;

    return r;
  }
};

} // namespace linalg
} // namespace proxnlp
