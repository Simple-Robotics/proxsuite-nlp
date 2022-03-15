/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <vector>

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"
#include "lienlp/meritfuncs/pdal.hpp"
#include "lienlp/workspace.hpp"


namespace lienlp {

  template<typename M>  
  class Solver
  {
  public:
    using Scalar = typename M::Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;
    using Merit_t = PDALFunction<Scalar>;

    shared_ptr<Prob_t> m_problem;
    Merit_t m_merit;
    M& m_manifold;

    Solver(M& man, shared_ptr<Prob_t>& prob)
      : m_manifold(man), m_problem(prob)
    {}

    void solve(SWorkspace& workspace, const VectorXs& x0, const VectorOfVectors& lams0)
    {
      workspace.x_prev = x0;
      workspace.lams_prev = lams0;
    }

  };

} // namespace lienlp

