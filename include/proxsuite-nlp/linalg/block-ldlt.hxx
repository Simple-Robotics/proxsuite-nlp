/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "./block-ldlt.hpp"

namespace proxsuite {
namespace nlp {
namespace linalg {

template <typename Scalar>
void BlockLDLT<Scalar>::setBlockPermutation(isize const *new_perm) {
  SymbolicBlockMatrix in(m_structure.copy());
  const isize n = m_structure.nsegments();
  if (new_perm != nullptr)
    std::copy_n(new_perm, n, m_perm.data());
  m_structure.performed_llt = false;
  symbolic_deep_copy(in, m_structure, m_perm.data());
  analyzePattern(); // call manually
  updateBlockPermutationMatrix(in);
}

template <typename Scalar>
BlockLDLT<Scalar> &BlockLDLT<Scalar>::findSparsifyingPermutation() {
  SymbolicBlockMatrix in(m_structure.copy());
  m_structure.brute_force_best_permutation(in, m_perm.data(), m_iwork.data());
  symbolic_deep_copy(in, m_structure, m_perm.data());
  analyzePattern();
  updateBlockPermutationMatrix(in);
  return *this;
}

template <typename Scalar> bool BlockLDLT<Scalar>::analyzePattern() {
  if (m_structure.performed_llt)
    return true;
  bool flag = m_structure.llt_in_place();
  m_struct_tr = m_structure.transpose();
  return flag;
}

template <typename Scalar>
BlockLDLT<Scalar> &BlockLDLT<Scalar>::updateBlockPermutationMatrix(
    const SymbolicBlockMatrix &init) {
  PermIdxType &indices = m_permutation.indices();
  computeStartIndices(init);

  isize idx = 0;
  for (isize i = 0; i < (isize)nblocks(); ++i) {
    auto j = (usize)m_perm[(usize)i];
    isize len = init.segment_lens[j];
    isize i0 = m_start_idx[j];
    indices.segment(idx, len).setLinSpaced(i0, i0 + len - 1);
    idx += len;
  }
  return *this;
}

template <typename Scalar>
typename BlockLDLT<Scalar>::MatrixXs
BlockLDLT<Scalar>::reconstructedMatrix() const {
  MatrixXs res(m_matrix.rows(), m_matrix.cols());
  res.setIdentity();
  MatrixXs mat = m_matrix.template selfadjointView<Eigen::Lower>();
  backend::dense_ldlt_reconstruct<Scalar>(mat, res);
  res.noalias() = permutationP() * res;
  res.noalias() = res * permutationP().transpose();
  return res;
}

template <typename Scalar>
template <typename Derived>
bool BlockLDLT<Scalar>::solveInPlace(Eigen::MatrixBase<Derived> &b) const {

  b.noalias() = permutationP().transpose() * b;
  PROXSUITE_NLP_NOMALLOC_BEGIN;
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
  BlockTriU mat_blk_U(m_matrix.transpose(), m_struct_tr);
  flag |= mat_blk_U.solveInPlace(b);

  PROXSUITE_NLP_NOMALLOC_END;
  b.noalias() = permutationP() * b;
  return flag;
}

} // namespace linalg
} // namespace nlp
} // namespace proxsuite
