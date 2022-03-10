#pragma once

#include "lienlp/manifold-base.hpp"
#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"
#include "lienlp/constraints/equality-constraint.hpp"

#include <vector>


namespace lienlp {
  
  template<typename _Scalar>
  struct Problem
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    /// Generic constraint type
    using Cstr_t = ConstraintFormatBaseTpl<Scalar>;
    /// Equality constraint type
    using Equality_t = EqualityConstraint<Scalar>;
    /// Cost function type
    using Cost_t = CostFunction<Scalar>;

    /// The cost functional.
    const Cost_t& m_cost;
    /// List of equality constraints.
    std::vector<Equality_t*> m_eq_cstrs;

    inline const Equality_t* getEqCs(const std::size_t& i) const
    {
      return m_eq_cstrs[i];
    }

    std::size_t getNumConstraints() const
    {
      return m_eq_cstrs.size();
    }

    void allocateMultipliers(VectorList& out) const
    {
      out.resize(getNumConstraints());
      for (std::size_t i = 0; i < getNumConstraints(); i++)
      {
        auto cur_cstr = getEqCs(i);
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
            std::vector<Equality_t*> eq_cstrs)
            : m_cost(cost), m_eq_cstrs(eq_cstrs)
            {}

  };

} // namespace lienlp


