#pragma once

#include "proxnlp/math.hpp"
#include "proxnlp/macros.hpp"

namespace proxnlp {
namespace linalg {

constexpr int choose_row_or_col_major(int Options) {
  return Options & Eigen::RowMajor ? Eigen::RowMajor : Eigen::ColMajor;
}

template <typename VectorType, typename OutType>
void cumsum(const Eigen::MatrixBase<VectorType> &in,
            Eigen::MatrixBase<OutType> &out) {
  static_assert(VectorType::IsVectorAtCompileTime, ".");
  static_assert(OutType::IsVectorAtCompileTime, ".");
  eigen_assert(out.size() == in.size());
  typedef typename VectorType::Index Index;
  out[0] = 0;
  for (Index i = 0; i < in.size() - 1; ++i) {
    out[i + 1] = out[i] + in[i];
  }
}

static constexpr int DYN = Eigen::Dynamic;

/// Type for a block-diagonal matrix. This should be constructable online i.e.
/// along a factorization procedure.
/// Storage layout is row-wise in the diagonal blocks (makes block data
/// contiguous).
template <typename _Scalar, int _MaxBlockSize = Eigen::Dynamic,
          int _Options = Eigen::ColMajor>
struct BlockDiagonalMatrix {
  using Scalar = _Scalar;
  using Index = Eigen::DenseIndex;
  static constexpr int Options = _Options;
  static constexpr int MaxBlockSize = _MaxBlockSize;
  using IndexVector = Eigen::Matrix<Index, DYN, 1, 0>;
  using StorageKind = Eigen::Dense;
  using CoefficientsType =
      Eigen::Matrix<Scalar, DYN, DYN, choose_row_or_col_major(Options)>;

  static constexpr int get_storage_num_rows(Index size) {
    return MaxBlockSize != DYN ? static_cast<Index>(MaxBlockSize) : size;
  }

  Index size() const { return m_size; }
  Index rows() const { return m_size; }
  Index cols() const { return m_size; }

  Index nblocks() { return m_blocksizes.size(); }
  const IndexVector &blockSizes() const { m_blocksizes; }
  Index blockSize(Index i) const { return blockSizes()[i]; }
  const IndexVector &blockStarts() const { m_blockstarts; }

  const CoefficientsType &coeffs() const { return m_coeffs; }
  CoefficientsType &coeffs() { return m_coeffs; }

  /// @brief Constructor using provided block sizes.
  explicit BlockDiagonalMatrix(const IndexVector &sizes)
      : m_size(sizes.sum()), m_nblocks(sizes.size()), m_cumul(-1),
        m_coeffs(get_storage_num_rows(m_size), m_size), m_blocksizes(sizes),
        m_blockstarts(m_nblocks) {
    cumsum(sizes, m_blockstarts);
  }

  BlockDiagonalMatrix(Index size = 0)
      : m_size(size), m_nblocks(0), m_cumul(0), m_coeffs(m_size, m_size),
        m_blocksizes(), m_blockstarts() {}

  inline bool is_full() const { return m_cumul == m_size; }

  /// Add block, by its size.
  void add_block(Index size) {
    eigen_assert(!is_full() && "Entire matrix already allocated to blocks.");
    eigen_assert(m_cumul + size <= m_size && "Size of added block too large.");
    m_blocksizes[m_nblocks] = size;
    m_blockstarts[m_nblocks] = m_cumul;
    m_cumul += size;
    ++m_nblocks;
    return block_impl(m_nblocks - 1);
  }

  template <typename Derived>
  void add_block(const Eigen::MatrixBase<Derived> &matrix) {
    assert(matrix.rows() == matrix.cols());
    add_block(matrix.size()) = matrix;
  }

  auto block(Index i) { return block_impl(i); }
  auto block(Index i) const { return block_impl(i); }

protected:
  Eigen::Block<CoefficientsType> block_impl(Index i) {
    return m_coeffs.block(0, m_blockstarts[i], m_blocksizes[i],
                          m_blocksizes[i]);
  }

  /// Matrix size
  Index m_size;
  /// Number of blocks
  Index m_nblocks;
  /// Current running total size of blocks. Used for bookkeeping.
  Index m_cumul;
  /// Matrix coefficients
  CoefficientsType m_coeffs;
  /// Block sizes
  IndexVector m_blocksizes;
  /// Block start indices
  IndexVector m_blockstarts;
};

} // namespace linalg
} // namespace proxnlp
