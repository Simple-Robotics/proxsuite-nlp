#pragma once

#include "proxnlp/linalg/block-ldlt.hpp"

using namespace proxnlp;
using linalg::BlockKind;
using linalg::isize;
using linalg::SymbolicBlockMatrix;
using Scalar = double;

PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

inline MatrixXs get_block_matrix(SymbolicBlockMatrix const &sym) {
  isize *row_segments = sym.segment_lens;
  auto n = std::size_t(sym.nsegments());
  isize size = 0;
  for (std::size_t i = 0; i < n; ++i)
    size += row_segments[i];

  MatrixXs mat(size, size);
  mat.setZero();

  isize startRow = 0;
  isize startCol = 0;
  for (unsigned i = 0; i < n; ++i) {
    isize blockRows = row_segments[i];
    startCol = 0;
    for (unsigned j = 0; j <= i; ++j) {
      isize blockCols = row_segments[j];
      const BlockKind kind = sym(i, j);
      auto block = mat.block(startRow, startCol, blockRows, blockCols);
      switch (kind) {
      case BlockKind::Zero:
        block.setZero();
        break;
      case BlockKind::Dense:
        block.setRandom();
        break;
      case BlockKind::Diag:
        block.diagonal().setRandom();
        break;
      case BlockKind::TriL:
        block.setRandom();
        block.triangularView<Eigen::StrictlyUpper>().setZero();
        break;
      case BlockKind::TriU:
        block.setRandom();
        block.triangularView<Eigen::StrictlyLower>().setZero();
        break;
      default:
        break;
      }
      startCol += blockCols;
    }
    startRow += blockRows;
  }
  mat = mat.selfadjointView<Eigen::Lower>();
  return mat;
}
