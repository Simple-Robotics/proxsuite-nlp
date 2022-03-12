#pragma once

#include "lienlp/manifold-base.hpp"
#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"
#include "lienlp/constraints/equality-constraint.hpp"

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>


namespace lienlp {
  
  using boost::shared_ptr;

  template<typename _Scalar>
  struct Problem
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    /// Generic constraint type
    using Cstr_t = ConstraintFormatBaseTpl<Scalar>;
    using CstrPtr = shared_ptr<Cstr_t>;
    /// Equality constraint type
    using Equality_t = EqualityConstraint<Scalar>;
    /// Cost function type
    using Cost_t = CostFunction<Scalar>;

    /// The cost functional.
    const Cost_t& m_cost;
    /// List of equality constraints.
    const std::vector<CstrPtr> m_cstrs;

    const CstrPtr getCstr(const std::size_t& i) const
    {
      return m_cstrs[i];
    }

    std::size_t getNumConstraints() const
    {
      return m_cstrs.size();
    }

    void allocateMultipliers(VectorList& out) const
    {
      out.resize(getNumConstraints());
      for (std::size_t i = 0; i < getNumConstraints(); i++)
      {
        CstrPtr cur_cstr = m_cstrs[i];
        out[i] = VectorXs::Zero(cur_cstr->getDim());
      }
    }

    VectorList allocateMultipliers() const
    {
      VectorList out_;
      allocateMultipliers(out_);
      return out_;
    }

    Problem(const Cost_t& cost)
            : m_cost(cost) {}

    Problem(const Cost_t& cost,
            std::vector<CstrPtr> eq_cstrs)
            : m_cost(cost), m_cstrs(eq_cstrs)
            {}

  };

} // namespace lienlp


