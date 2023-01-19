#include <Eigen/Core>

namespace proxnlp {
namespace linalg {

namespace {
using ::Eigen::TranspositionsBase;
}

// fwd declare
template <typename _IndicesType> class TranspositionsWrapper;
} // namespace linalg
} // namespace proxnlp

namespace Eigen {
namespace internal {
template <typename _IndicesType>
struct traits<proxnlp::linalg::TranspositionsWrapper<_IndicesType>>
    : traits<::Eigen::TranspositionsWrapper<_IndicesType>> {};
} // namespace internal
} // namespace Eigen

namespace proxnlp {
namespace linalg {

/// @brief   A fixed version of Eigen::TranspositionsWrapper.
/// @see     The original @file Eigen/src/Core/Transpositions.h
/// @details This only fixes getting a mutable reference to the indices.
template <typename _IndicesType>
class TranspositionsWrapper
    : public TranspositionsBase<TranspositionsWrapper<_IndicesType>> {
  typedef ::Eigen::internal::traits<TranspositionsWrapper> Traits;

public:
  typedef TranspositionsBase<TranspositionsWrapper> Base;
  typedef typename Traits::IndicesType IndicesType;
  typedef typename IndicesType::Scalar StorageIndex;

  explicit inline TranspositionsWrapper(IndicesType &indices)
      : m_indices(indices) {}

  /** Copies the \a other transpositions into \c *this */
  template <typename OtherDerived>
  TranspositionsWrapper &
  operator=(const TranspositionsBase<OtherDerived> &other) {
    return Base::operator=(other);
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** This is a special case of the templated operator=. Its purpose is to
   * prevent a default operator= from hiding the templated operator=.
   */
  TranspositionsWrapper &operator=(const TranspositionsWrapper &other) {
    m_indices = other.m_indices;
    return *this;
  }
#endif

  /** const version of indices(). */
  const IndicesType &indices() const { return m_indices; }

  /** \returns a reference to the stored array representing the transpositions.
   */
  IndicesType &indices() { return const_cast<IndicesType &>(m_indices); }

protected:
  typename IndicesType::Nested m_indices;
};

} // namespace linalg
} // namespace proxnlp
