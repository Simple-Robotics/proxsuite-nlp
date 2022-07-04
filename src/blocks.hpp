/**
 * @file blocks.hpp
 * @author Sarah El-Kazdadi
 * @brief Routines for block-sparse (notably, KKT-type) matrix LDLT factorisation.
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxnlp/math.hpp"
#include <type_traits>

#include <vector>
#include <algorithm>
#include <numeric>

#include <iostream>

namespace proxnlp {
/// @brief	Block-wise Cholesky or LDLT factorisation routines.
namespace block_chol {

using Scalar = double;
using MatrixRef = proxnlp::math_types<Scalar>::MatrixRef;

using usize = std::size_t;
using isize = typename std::make_signed<usize>::type;

enum BlockKind {
  Zero,
  Diag,
  TriL,
  TriU,
  Dense,
};

auto trans(BlockKind a) noexcept -> BlockKind {
  if (a == TriL) {
    return TriU;
  }
  if (a == TriU) {
    return TriL;
  }
  return a;
}

auto add(BlockKind a, BlockKind b) noexcept -> BlockKind {
  if (a == Dense || b == Dense || int(a) + int(b) == int(TriL) + int(TriU)) {
    return Dense;
  }
  return std::max(a, b);
}

auto mul(BlockKind a, BlockKind b) noexcept -> BlockKind {
  if (a == Zero || b == Zero) {
    return Zero;
  }
  return block_chol::add(a, b);
}

struct SymbolicBlockMatrix {
  struct Raw {
    BlockKind *data;
    isize *segment_lens;
    isize segments_count;
    isize outer_stride;
  } _;

  SymbolicBlockMatrix /* NOLINT */ (Raw raw) noexcept : _{raw} {}

  auto nsegments() const noexcept -> isize { return _.segments_count; }
  auto ptr(isize i, isize j) const noexcept -> BlockKind * {
    return _.data + (i + j * _.outer_stride);
  }
  auto submatrix(isize i, isize n) const noexcept -> SymbolicBlockMatrix {

    return {
        Raw{
            ptr(i, i),
            _.segment_lens + i,
            n,
            _.outer_stride,
        },
    };
  }
  auto operator()(isize i, isize j) const noexcept -> BlockKind & {

    return *ptr(i, j);
  }

  void deep_copy(SymbolicBlockMatrix in,
                 isize const *perm = nullptr) const noexcept {
    auto self = *this;

    isize n = self.nsegments();

    for (isize i = 0; i < n; ++i) {
      self._.segment_lens[i] =
          in._.segment_lens[(perm != nullptr) ? perm[i] : i];
    }
    for (isize i = 0; i < n; ++i) {
      for (isize j = 0; j < n; ++j) {
        if (perm == nullptr) {
          self(i, j) = in(i, j);
        } else {
          self(i, j) = in(perm[i], perm[j]);
        }
      }
    }
  }

  // work has length `in.nsegments()`
  void brute_force_best_permutation(SymbolicBlockMatrix in, isize *best_perm,
                                    isize *iwork) const {
    isize n = in.nsegments();
    std::iota(iwork, iwork + n, isize(0));

    bool first_iter = true;
    isize best_perm_nnz = 0;

    // find best permutation
    do {
      deep_copy(in, iwork);
      llt_in_place();

      isize nnz = count_nnz();

      if (first_iter || nnz < best_perm_nnz) {
        std::memcpy(best_perm, iwork, usize(n) * sizeof(isize));
        best_perm_nnz = nnz;
      }

      first_iter = false;
    } while (std::next_permutation(iwork, iwork + n));
  }

  auto count_nnz() const noexcept -> isize {
    auto self = *this;
    isize nnz = 0;
    isize n = nsegments();

    for (isize i = 0; i < n; ++i) {
      for (isize j = 0; j < n; ++j) {
        switch (self(i, j)) {
        case Zero:
          break;
        case Diag: {
          nnz += self._.segment_lens[i];
          break;
        }
        case TriL:
        case TriU: {
          isize k = self._.segment_lens[i];
          nnz += (k * (k + 1)) / 2;
          break;
        }
        case Dense: {
          nnz += self._.segment_lens[i] * self._.segment_lens[j];
        }
        }
      }
    }
    return nnz;
  }

  void llt_in_place() const noexcept {
    // assume `*this` is symmetric
    if (_.segments_count == 0) {
      return;
    }

    auto self = *this;

    isize n = _.segments_count;

    // zero triu part
    for (isize j = 1; j < n; ++j) {
      self(0, j) = BlockKind::Zero;
    }

    switch (self(0, 0)) {
    case TriL:
    case TriU:
    case Zero:
      std::terminate();
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
      self(i, i) =
          block_chol::add(self(i, i),
                          block_chol::mul( //
                              self(i, 0), block_chol::trans(self(i, 0))));

      for (isize j = i + 1; j < n; ++j) {

        self(i, j) =
            block_chol::add(self(i, j),
                            block_chol::mul( //
                                self(i, 0), block_chol::trans(self(j, 0))));

        self(j, i) = block_chol::trans(self(i, j));
      }
    }

    submatrix(1, n - 1).llt_in_place();
  }

  void dump() const noexcept {
    isize nrows = 0;
    for (isize i = 0; i < _.segments_count; ++i) {
      nrows += _.segment_lens[i];
    }

    isize ncols = nrows;

    std::vector<bool> buf;
    buf.resize(usize(nrows * ncols));

    isize handled_rows = 0;
    for (isize i = 0; i < _.segments_count; ++i) {
      isize handled_cols = 0;
      for (isize j = 0; j < _.segments_count; ++j) {
        switch ((*this)(i, j)) {
        case Zero:
          break;
        case Diag: {
          for (isize ii = 0; ii < _.segment_lens[i]; ++ii) {
            buf[usize((handled_rows + ii) * ncols + handled_cols + ii)] = true;
          }
          break;
        }
        case TriL: {
          for (isize ii = 0; ii < _.segment_lens[i]; ++ii) {
            for (isize jj = 0; jj <= ii; ++jj) {
              buf[usize((handled_rows + ii) * ncols + handled_cols + jj)] =
                  true;
            }
          }
          break;
        }
        case TriU: {
          for (isize ii = 0; ii < _.segment_lens[i]; ++ii) {
            for (isize jj = ii; jj < _.segment_lens[j]; ++jj) {
              buf[usize((handled_rows + ii) * ncols + handled_cols + jj)] =
                  true;
            }
          }
          break;
        }
        case Dense: {
          for (isize ii = 0; ii < _.segment_lens[i]; ++ii) {
            for (isize jj = 0; jj < _.segment_lens[j]; ++jj) {
              buf[usize((handled_rows + ii) * ncols + handled_cols + jj)] =
                  true;
            }
          }
          break;
        }
        }
        handled_cols += _.segment_lens[j];
      }
      handled_rows += _.segment_lens[i];
    }

    for (isize i = 0; i < nrows; ++i) {
      for (isize j = 0; j < ncols; ++j) {
        if (buf[usize(i * ncols + j)]) {
          std::cout << "█";
        } else {
          std::cout << "░";
        }
      }
      std::cout << '\n';
    }
  }
};

namespace backend {
template <typename M> auto ref(M &mat) noexcept -> MatrixRef {
  static_assert(M::InnerStrideAtCompileTime == 1, ".");
  return mat;
}

auto ref_submatrix(MatrixRef a, isize i, isize j, isize nrows,
                   isize ncols) noexcept -> MatrixRef {
  return a.block(i, j, nrows, ncols);
}

void ldlt_in_place_unblocked(MatrixRef a) {
  isize n = a.rows();
  if (n == 0) {
    return;
  }

  isize j = 0;
  while (true) {
    auto l10 = a.row(j).head(j);
    auto d0 = a.diagonal().head(j);
    auto work = a.col(n - 1).head(j);

    work = l10.transpose().cwiseProduct(d0);
    a(j, j) -= work.dot(l10);

    if (j + 1 == n) {
      return;
    }

    isize rem = n - j - 1;

    auto l20 = a.bottomLeftCorner(rem, j);
    auto l21 = a.col(j).tail(rem);

    l21.noalias() -= l20 * work;
    l21 *= 1 / a(j, j);
    ++j;
  }
}

void ldlt_in_place_recursive(MatrixRef const &a) {
  isize n = a.rows();
  if (n <= 128) {
    backend::ldlt_in_place_unblocked(a);
  } else {
    isize bs = (n + 1) / 2;
    isize rem = n - bs;

    auto l00 = backend::ref_submatrix(a, 0, 0, bs, bs);
    auto l10 = backend::ref_submatrix(a, bs, 0, rem, bs);
    auto l11 = backend::ref_submatrix(a, bs, bs, rem, rem);

    backend::ldlt_in_place_recursive(l00);
    auto d0 = l00.diagonal();

    l00.transpose()
        .template triangularView<Eigen::UnitUpper>()
        .template solveInPlace<Eigen::OnTheRight>(l10);

    auto work = backend::ref_submatrix(a, 0, rem, rem, bs);
    work = l10;
    l10 = l10 * d0.asDiagonal().inverse();

    l11.template triangularView<Eigen::Lower>() -= l10 * work.transpose();

    backend::ldlt_in_place_recursive(l11);
  }
}

void ldlt_in_place(MatrixRef const &a) { ldlt_in_place_recursive(a); }

template <BlockKind LHS, BlockKind RHS> struct GemmT;

template <BlockKind LHS> struct GemmT<LHS, Zero> {
  static void fn(MatrixRef const & /*dst*/, MatrixRef const & /*lhs*/,
                 MatrixRef const & /*rhs*/, Scalar /*alpha*/) {}
};
template <BlockKind RHS> struct GemmT<Zero, RHS> {
  static void fn(MatrixRef const & /*dst*/, MatrixRef const & /*lhs*/,
                 MatrixRef const & /*rhs*/, Scalar /*alpha*/) {}
};
template <> struct GemmT<Zero, Zero> {
  static void fn(MatrixRef const & /*dst*/, MatrixRef const & /*lhs*/,
                 MatrixRef const & /*rhs*/, Scalar /*alpha*/) {}
};

template <> struct GemmT<Diag, Diag> {
  // dst is diagonal
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.diagonal() +=
        alpha * lhs.diagonal().cwiseProduct(rhs.transpose().diagonal());
  }
};

template <> struct GemmT<Diag, TriL> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Upper>() +=
    // 		alpha * (lhs.diagonal().asDiagonal() *
    //              rhs.template triangularView<Eigen::Lower>().transpose());

    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).head(j + 1) += alpha * lhs.diagonal().cwiseProduct(
                                            rhs.transpose().col(j).head(j + 1));
    }
  }
};

template <> struct GemmT<Diag, TriU> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    // 		alpha * (lhs.diagonal().asDiagonal() *
    //              rhs.template triangularView<Eigen::Upper>().transpose());

    isize m = dst.rows();
    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).tail(m - j) += alpha * lhs.diagonal().cwiseProduct(
                                            rhs.transpose().col(j).tail(m - j));
    }
  }
};

template <> struct GemmT<Diag, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst += alpha * (lhs.diagonal().asDiagonal() * rhs.transpose());
  }
};

template <> struct GemmT<TriL, Diag> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Lower>() *
    //              rhs.diagonal().asDiagonal());

    isize m = dst.rows();
    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).tail(m - j) += (alpha * rhs(j, j)) * lhs.col(j).tail(m - j);
    }
  }
};

template <> struct GemmT<TriL, TriL> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template triangularView<Eigen::Upper>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<TriL, TriU> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template triangularView<Eigen::Lower>());
    dst.triangularView<Eigen::Lower>() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <> struct GemmT<TriL, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        lhs.template triangularView<Eigen::Lower>() * (alpha * rhs.transpose());
  }
};

template <> struct GemmT<TriU, Diag> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Lower>() *
    //              rhs.diagonal().asDiagonal());

    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).head(j + 1) += (alpha * rhs(j, j)) * lhs.col(j).head(j + 1);
    }
  }
};

template <> struct GemmT<TriU, TriL> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // PERF
    // dst.template triangularView<Eigen::Upper>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Upper>() *
    //              rhs.transpose().triangularView<Eigen::Upper>());
    dst.template triangularView<Eigen::Upper>() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<TriU, TriU> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // PERF
    // dst.noalias() += alpha * (lhs.template triangularView<Eigen::Upper>() *
    //                           rhs.transpose().template
    //                           triangularView<Eigen::Lower>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <> struct GemmT<TriU, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() += (lhs.template triangularView<Eigen::Upper>() *
                      (alpha * rhs.transpose()));
  }
};

template <> struct GemmT<Dense, Diag> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() += alpha * (lhs * rhs.transpose().diagonal().asDiagonal());
  }
};

template <> struct GemmT<Dense, TriL> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<Dense, TriU> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Lower>());
  }
};

template <> struct GemmT<Dense, Dense> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() += alpha * lhs * rhs.transpose();
  }
};

void gemmt(MatrixRef const &dst, MatrixRef const &lhs, MatrixRef const &rhs,
           BlockKind lhs_kind, BlockKind rhs_kind, Scalar alpha) {
  // dst += alpha * lhs * rhs.Scalar
  switch (lhs_kind) {
  case Zero: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Zero, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Zero, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Zero, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Zero, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Zero, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case Diag: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Diag, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Diag, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Diag, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Diag, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Diag, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case TriL: {
    switch (rhs_kind) {
    case Zero:
      GemmT<TriL, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<TriL, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<TriL, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<TriL, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<TriL, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case TriU: {
    switch (rhs_kind) {
    case Zero:
      GemmT<TriU, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<TriU, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<TriU, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<TriU, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<TriU, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case Dense: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Dense, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Dense, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Dense, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Dense, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Dense, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  }
}
} // namespace backend

/// @brief Block matrix data structure with LDLT algos.
struct BlockMatrix {

  MatrixRef storage;
  SymbolicBlockMatrix structure;

  void permute(BlockMatrix in, isize const *perm) const {
    auto mat = this->storage;

    structure.deep_copy(in.structure, perm);

    isize nblocks = in.structure.nsegments();

    isize out_offset_i = 0;
    for (isize i = 0; i < nblocks; ++i) {
      auto bsi = structure._.segment_lens[i];

      isize in_offset_i = 0;
      for (isize ii = 0; ii < perm[i]; ++ii) {
        in_offset_i += in.structure._.segment_lens[ii];
      }

      isize out_offset_j = 0;
      for (isize j = 0; j < nblocks; ++j) {
        auto bsj = structure._.segment_lens[j];

        isize in_offset_j = 0;
        for (isize jj = 0; jj < perm[j]; ++jj) {
          in_offset_j += in.structure._.segment_lens[jj];
        }

        for (isize i_inner = 0; i_inner < bsi; ++i_inner) {
          for (isize j_inner = 0; j_inner < bsj; ++j_inner) {
            mat(out_offset_i + i_inner, out_offset_j + j_inner) =
                in.storage(in_offset_i + i_inner, in_offset_j + j_inner);
          }
        }

        out_offset_j += bsj;
      }

      out_offset_i += bsi;
    }
  }

  void ldlt_in_place_impl() const {

    isize nblocks = structure.nsegments();
    isize n = storage.rows();
    if (nblocks == 0) {
      return;
    }

    auto structure_00 = structure(0, 0);
    auto bs = structure._.segment_lens[0];
    auto rem = n - bs;
    auto l00 = backend::ref_submatrix(storage, 0, 0, bs, bs);
    auto l11 = backend::ref_submatrix(storage, bs, bs, rem, rem);
    auto d0 = l00.diagonal();

    auto work = backend::ref_submatrix(storage, 0, rem, rem, bs);

    switch (structure_00) {
    case Zero:
    case TriU:
    case Dense:
      std::terminate();

    case TriL: {
      // compute l00
      backend::ldlt_in_place(l00);

      isize offset = bs;

      for (isize i = 1; i < nblocks; ++i) {
        auto bsi = structure._.segment_lens[i];
        auto li0 = backend::ref_submatrix(storage, offset, 0, bsi, bs);
        auto li0_copy = backend::ref_submatrix(work, offset - bs, 0, bsi, bs);

        switch (structure(i, 0)) {
        case Diag:
        case TriL:
          std::terminate();
        case TriU: {
          auto li0_u = li0.template triangularView<Eigen::Upper>();
          // PERF: replace li0.Scalar by li0_tl
          // auto li0_tl = li0.transpose().template
          // triangularView<Eigen::Lower>();
          l00.template triangularView<Eigen::UnitLower>().solveInPlace(
              li0.transpose());
          li0_copy.template triangularView<Eigen::Upper>() = li0_u;
          li0_u = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Dense: {
          l00.template triangularView<Eigen::UnitLower>().solveInPlace(
              li0.transpose());
          li0_copy = li0;
          li0 = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Zero:
          break;
        }
      }

      break;
    }
    case Diag: {
      // l00 is unchanged
      isize offset = bs;

      for (isize i = 1; i < nblocks; ++i) {
        auto bsi = structure._.segment_lens[i];
        auto li0 = backend::ref_submatrix(storage, offset, 0, bsi, bs);
        auto li0_copy = backend::ref_submatrix(work, offset - bs, 0, bsi, bs);

        switch (structure(i, 0)) {
        case TriL:
          std::terminate();
        case Diag: {
          auto li0_d = li0.diagonal();
          li0_copy.diagonal() = li0_d;
          li0_d = li0_d.cwiseQuotient(d0);
          break;
        }
        case TriU: {
          auto li0_u = li0.template triangularView<Eigen::Upper>();
          li0_copy.template triangularView<Eigen::Upper>() = li0_u;
          li0_u = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Dense: {
          li0_copy = li0;
          li0 = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Zero:
          break;
        }
        offset += bsi;
      }

      break;
    }
    }

    isize offset_i = bs;
    for (isize i = 1; i < nblocks; ++i) {
      auto bsi = structure._.segment_lens[i];
      auto li0 = backend::ref_submatrix(storage, offset_i, 0, bsi, bs);
      auto li0_prev = backend::ref_submatrix(work, offset_i - bs, 0, bsi, bs);

      auto target_ii =
          backend::ref_submatrix(storage, offset_i, offset_i, bsi, bsi);

      // target_ii -= li0 * li0_prev.Scalar;
      backend::gemmt(target_ii, li0, li0_prev, structure(i, 0), structure(i, 0),
                     Scalar(-1));

      isize offset_j = offset_i + bsi;
      for (isize j = i + 1; j < nblocks; ++j) {
        // target_ji -= lj0 * li0_prev.Scalar

        auto bsj = structure._.segment_lens[j];
        auto lj0 = backend::ref_submatrix(storage, offset_j, 0, bsj, bs);
        auto target_ji =
            backend::ref_submatrix(storage, offset_j, offset_i, bsj, bsi);

        backend::gemmt(target_ji, lj0, li0_prev, structure(j, 0),
                       structure(i, 0), Scalar(-1));

        offset_j += bsj;
      }

      offset_i += bsi;
    }

    BlockMatrix{
        l11,
        structure.submatrix(1, nblocks - 1),
    }
        .ldlt_in_place_impl();
  }

  void ldlt_in_place() { ldlt_in_place_impl(); }
};

} // namespace block_chol
} // namespace proxnlp
