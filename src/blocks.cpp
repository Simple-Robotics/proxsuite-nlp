#include "proxnlp/blocks.hpp"

namespace proxnlp {
namespace block_chol {

/// BlockKind of the transpose of a matrix.
BlockKind trans(BlockKind a) noexcept {
  if (a == TriL) {
    return TriU;
  }
  if (a == TriU) {
    return TriL;
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
  return block_chol::add(a, b);
}

void dump(const SymbolicBlockMatrix &smat) noexcept {
  isize nrows = 0;
  auto &raw = smat._;
  for (isize i = 0; i < raw.segments_count; ++i) {
    nrows += raw.segment_lens[i];
  }

  isize ncols = nrows;

  std::vector<bool> buf;
  buf.resize(std::size_t(nrows * ncols));

  isize handled_rows = 0;
  for (isize i = 0; i < raw.segments_count; ++i) {
    isize handled_cols = 0;
    for (isize j = 0; j < raw.segments_count; ++j) {
      switch (smat(i, j)) {
      case Zero:
        break;
      case Diag: {
        for (isize ii = 0; ii < raw.segment_lens[i]; ++ii) {
          buf[std::size_t((handled_rows + ii) * ncols + handled_cols + ii)] =
              true;
        }
        break;
      }
      case TriL: {
        for (isize ii = 0; ii < raw.segment_lens[i]; ++ii) {
          for (isize jj = 0; jj <= ii; ++jj) {
            buf[std::size_t((handled_rows + ii) * ncols + handled_cols + jj)] =
                true;
          }
        }
        break;
      }
      case TriU: {
        for (isize ii = 0; ii < raw.segment_lens[i]; ++ii) {
          for (isize jj = ii; jj < raw.segment_lens[j]; ++jj) {
            buf[std::size_t((handled_rows + ii) * ncols + handled_cols + jj)] =
                true;
          }
        }
        break;
      }
      case Dense: {
        for (isize ii = 0; ii < raw.segment_lens[i]; ++ii) {
          for (isize jj = 0; jj < raw.segment_lens[j]; ++jj) {
            buf[std::size_t((handled_rows + ii) * ncols + handled_cols + jj)] =
                true;
          }
        }
        break;
      }
      }
      handled_cols += raw.segment_lens[j];
    }
    handled_rows += raw.segment_lens[i];
  }

  for (isize i = 0; i < nrows; ++i) {
    for (isize j = 0; j < ncols; ++j) {
      if (buf[std::size_t(i * ncols + j)]) {
        std::cout << "█";
      } else {
        std::cout << "░";
      }
    }
    std::cout << '\n';
  }
}

} // namespace block_chol
} // namespace proxnlp
