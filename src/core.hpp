/**
 * Copyright (c) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include <Eigen/Core>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <vector>
#include <boost/shared_ptr.hpp>

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"
#include "lienlp/meritfuncs/pdal.hpp"


namespace lienlp {

  using boost::shared_ptr;

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

    Solver(shared_ptr<Prob_t>& prob)
      : m_problem(prob)
    {}

  };

} // namespace lienlp

