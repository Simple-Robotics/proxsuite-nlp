#pragma once

#include "lienlp/macros.hpp"


namespace lienlp {

  /** Workspace class, which holds the necessary intermediary data
   * for the solver to function.
   */
  template<typename _Scalar>
  struct SWorkspace
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(_Scalar)


    /// KKT iteration matrix; of dynamic size.
    MatrixXs kktMatrix;
    /// KKT iteration right-hand side.
    VectorXs kktRhs;
  }


} // namespace lienlp

