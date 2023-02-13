/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "./blocks.hpp"

namespace proxnlp {
namespace linalg {

template <typename Scalar>
void BlockLDLT<Scalar>::setPermutation(isize const *new_perm) {
  auto in = m_structure.copy();
  const isize n = m_structure.nsegments();
  if (new_perm != nullptr)
    std::copy_n(new_perm, n, m_perm.data());
  m_structure.performed_llt = false;
  symbolic_deep_copy(in, m_structure, m_perm.data());
  analyzePattern(); // call manually
}

template <typename Scalar>
BlockLDLT<Scalar> &BlockLDLT<Scalar>::findSparsifyingPermutation() {
  auto in = m_structure.copy();
  m_structure.brute_force_best_permutation(in, m_perm.data(), m_iwork.data());
  symbolic_deep_copy(in, m_structure, m_perm.data());
  analyzePattern();
  return *this;
}

template <typename Scalar>
BlockLDLT<Scalar> &
BlockLDLT<Scalar>::updateBlockPermutationMatrix(const SymbolicBlockMatrix &in) {
  const isize *row_segs = in.segment_lens;
  using IndicesType = PermutationType::IndicesType;
  IndicesType &indices = m_permutation.indices();
  isize idx = 0;
  for (std::size_t i = 0; i < nblocks(); ++i) {
    m_idx[i] = idx;
    idx += row_segs[i];
  }

  idx = 0;
  for (std::size_t i = 0; i < nblocks(); ++i) {
    auto len = row_segs[m_perm[i]];
    auto s = indices.segment(idx, len);
    isize i0 = m_idx[m_perm[i]];
    s.setLinSpaced(i0, i0 + len - 1);
    idx += len;
  }
  m_permutation = m_permutation.transpose();
  return *this;
}

template <typename Scalar>
typename BlockLDLT<Scalar>::MatrixXs
BlockLDLT<Scalar>::reconstructedMatrix() const {
  MatrixXs res(m_matrix.rows(), m_matrix.cols());
  res.setIdentity();
  backend::dense_ldlt_reconstruct<Scalar>(m_matrix, res);
  res.noalias() = res * permutationP();
  res.noalias() = permutationP().transpose() * res;
  return res;
}

template <typename Scalar>
bool BlockLDLT<Scalar>::solveInPlace(MatrixRef b) const {

  b.noalias() = permutationP() * b;
  PROXNLP_NOMALLOC_BEGIN;
  BlockTriL mat_blk_L(m_matrix, m_structure);
  bool flag = mat_blk_L.solveInPlace(b);

  using std::abs;
  auto vecD(m_matrix.diagonal());
  const Scalar tol = std::numeric_limits<Scalar>::min();
  for (isize i = 0; i < vecD.size(); ++i) {
    if (abs(vecD(i)) > tol)
      b.row(i) /= vecD(i);
    else
      b.row(i).setZero();
  }

  /// TODO: fixcaching this variable somewhere w/ update to m_structure
  auto struct_tr = m_structure.transpose();
  BlockTriU mat_blk_U(m_matrix.transpose(), struct_tr);
  flag |= mat_blk_U.solveInPlace(b);
  delete[] struct_tr.data;
  delete[] struct_tr.segment_lens;

  PROXNLP_NOMALLOC_END;
  b.noalias() = permutationP().transpose() * b;
  return flag;
}

} // namespace linalg
} // namespace proxnlp
