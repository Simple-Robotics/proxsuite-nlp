/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxnlp/linalg/block-kind.hpp"

#include <numeric>
#include <iostream>

namespace proxnlp {
namespace linalg {

/// BlockKind of the transpose of a matrix.
BlockKind trans(BlockKind a) noexcept {
  switch (a) {
  case TriL:
    return TriU;
    break;
  case TriU:
    return TriL;
    break;
  default:
    return a;
    break;
  }
  return a;
}

/// BlockKind of the addition of two matrices - given by their BlockKind.
BlockKind add(BlockKind a, BlockKind b) noexcept {
  if (a == Dense || b == Dense || int(a) + int(b) == int(TriL) + int(TriU)) {
    return Dense;
  }
  return std::max(a, b);
}

/// BlockKind of the product of two matrices.
BlockKind mul(BlockKind a, BlockKind b) noexcept {
  if (a == Zero || b == Zero) {
    return Zero;
  }
  return linalg::add(a, b);
}

/* IMPLS FOR SYMBOLIC MATRIX */

SymbolicBlockMatrix SymbolicBlockMatrix::submatrix(isize i,
                                                   isize n) const noexcept {
  return {ptr(i, i), segment_lens + i, n, outer_stride, performed_llt};
}

SymbolicBlockMatrix SymbolicBlockMatrix::copy() const {
  BlockKind *data = new BlockKind[std::size_t(segments_count * outer_stride)];
  isize *segment_lens = new isize[std::size_t(segments_count)];
  SymbolicBlockMatrix out{data, segment_lens, segments_count, outer_stride};
  symbolic_deep_copy(*this, out);
  return out;
}

void symbolic_deep_copy(const SymbolicBlockMatrix &in, SymbolicBlockMatrix &out,
                        isize const *perm) noexcept {
  const isize n = out.nsegments();

  for (isize i = 0; i < n; ++i) {
    out.segment_lens[i] = in.segment_lens[(perm != nullptr) ? perm[i] : i];
  }
  for (isize i = 0; i < n; ++i) {
    for (isize j = 0; j < n; ++j) {
      if (perm == nullptr) {
        out(i, j) = in(i, j);
      } else {
        out(i, j) = in(perm[i], perm[j]);
      }
    }
  }
  out.performed_llt = in.performed_llt;
}

Eigen::ComputationInfo SymbolicBlockMatrix::brute_force_best_permutation(
    SymbolicBlockMatrix const &in, isize *best_perm, isize *iwork) {
  isize n = in.nsegments();
  std::iota(iwork, iwork + n, isize(0));

  bool first_iter = true;
  isize best_perm_nnz = 0;

  // find best permutation
  do {
    symbolic_deep_copy(in, *this, iwork);
    performed_llt = false;
    if (!llt_in_place()) {
      return Eigen::NumericalIssue;
    }

    isize nnz = count_nnz();

    if (first_iter || nnz < best_perm_nnz) {
      std::memcpy(best_perm, iwork, std::size_t(n) * sizeof(isize));
      best_perm_nnz = nnz;
    }

    first_iter = false;
  } while (std::next_permutation(iwork, iwork + n));
  return Eigen::Success;
}

bool SymbolicBlockMatrix::check_if_symmetric() const noexcept {
  const isize n = nsegments();
  const auto &self = *this;
  for (isize i = 0; i < n; ++i) {
    for (isize j = 0; j < i; j++) {
      if (self(i, j) != self(j, i)) {
        return false;
      }
    }
  }
  return true;
}

isize SymbolicBlockMatrix::count_nnz() const noexcept {
  auto &self = *this;
  isize nnz = 0;
  isize n = nsegments();

  for (isize i = 0; i < n; ++i) {
    for (isize j = 0; j < n; ++j) {
      switch (self(i, j)) {
      case Zero:
        break;
      case Diag: {
        nnz += self.segment_lens[i];
        break;
      }
      case TriL:
      case TriU: {
        isize k = self.segment_lens[i];
        nnz += (k * (k + 1)) / 2;
        break;
      }
      case Dense: {
        nnz += self.segment_lens[i] * self.segment_lens[j];
      }
      }
    }
  }
  return nnz;
}

bool SymbolicBlockMatrix::llt_in_place() noexcept {
  // assume `*this` is symmetric
  if (segments_count == 0) {
    performed_llt = true;
    return performed_llt;
  }

  auto &self = *this;

  isize n = segments_count;

  // zero triu part
  for (isize j = 1; j < n; ++j) {
    self(0, j) = BlockKind::Zero;
  }

  switch (self(0, 0)) {
  case TriL:
  case TriU:
  case Zero:
    return false;
  case Dense: {
    self(0, 0) = TriL;
    for (isize i = 1; i < n; ++i) {
      switch (self(i, 0)) {
      case Zero:
      case Diag: {
        self(i, 0) = TriU;
        break;
      }
      case TriL: {
        self(i, 0) = Dense;
        break;
      }
      case TriU:
      case Dense:
        break;
      }
    }
  }
  case Diag: {
    // l00 unchanged
    // l10 unchanged
  }
  }

  // update l11
  for (isize i = 1; i < n; ++i) {
    self(i, i) = linalg::add(self(i, i),
                             linalg::mul( //
                                 self(i, 0), linalg::trans(self(i, 0))));

    for (isize j = i + 1; j < n; ++j) {

      self(i, j) = linalg::add(self(i, j),
                               linalg::mul( //
                                   self(i, 0), linalg::trans(self(j, 0))));

      self(j, i) = linalg::trans(self(i, j));
    }
  }

  performed_llt = submatrix(1, n - 1).llt_in_place();
  return performed_llt;
}

/* UTILS */

void print_sparsity_pattern(const SymbolicBlockMatrix &smat) noexcept {
  isize nrows = 0;
  for (isize i = 0; i < smat.segments_count; ++i) {
    nrows += smat.segment_lens[i];
  }

  isize ncols = nrows;

  Eigen::Matrix<bool, -1, -1> buf(nrows, ncols);
  buf.setConstant(false);

  isize handled_rows = 0;
  for (isize i = 0; i < smat.segments_count; ++i) {
    isize handled_cols = 0;
    for (isize j = 0; j < smat.segments_count; ++j) {
      switch (smat(i, j)) {
      case Zero:
        break;
      case Diag: {
        auto blk = buf.block(handled_rows, handled_cols, smat.segment_lens[i],
                             smat.segment_lens[j]);
        blk.diagonal().setConstant(true);
        break;
      }
      case TriL: {
        for (isize ii = 0; ii < smat.segment_lens[i]; ++ii) {
          for (isize jj = 0; jj <= ii; ++jj) {
            buf(handled_rows + ii, handled_cols + jj) = true;
          }
        }
        break;
      }
      case TriU: {
        for (isize ii = 0; ii < smat.segment_lens[i]; ++ii) {
          for (isize jj = ii; jj < smat.segment_lens[j]; ++jj) {
            buf(handled_rows + ii, handled_cols + jj) = true;
          }
        }
        break;
      }
      case Dense: {
        auto blk = buf.block(handled_rows, handled_cols, smat.segment_lens[i],
                             smat.segment_lens[j]);
        blk.setConstant(true);
        break;
      }
      }
      handled_cols += smat.segment_lens[j];
    }
    handled_rows += smat.segment_lens[i];
  }

  for (isize i = 0; i < nrows; ++i) {
    for (isize j = 0; j < ncols; ++j) {
      if (buf(i, j)) {
        std::cout << "██";
      } else {
        std::cout << "░░";
      }
    }
    std::cout << '\n';
  }
}

} // namespace linalg
} // namespace proxnlp
