#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/cost-function.hpp"
#include "proxnlp/constraint-base.hpp"
#include "proxnlp/modelling/constraints/equality-constraint.hpp"

#include <vector>


namespace proxnlp
{

  template<typename _Scalar>
  struct ProblemTpl
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

    /// Generic constraint type
    using ConstraintType = ConstraintSetBase<Scalar>;
    using ConstraintPtr = shared_ptr<ConstraintType>;
    /// Equality constraint type
    using EqualityType = EqualityConstraint<Scalar>;
    /// Cost function type
    using CostType = CostFunctionBaseTpl<Scalar>;

    /// The cost functional
    const CostType& m_cost;
    /// Get a pointer to the \f$i\f$-th constraint pointer
    const ConstraintPtr getConstraint(const std::size_t& i) const
    {
      return m_cstrs[i];
    }

    /// @brief Get the number of constraint blocks.
    std::size_t getNumConstraints() const
    {
      return m_cstrs.size();
    }

    int getTotalConstraintDim() const
    {
      return m_nc_total;
    }

    std::vector<int> getConstraintDims() const
    {
      return m_ncs;
    }

    /// Get dimension of constraint \p i.
    int getConstraintDim(int i) const
    {
      return m_ncs[i];
    }

    std::size_t nx()  const { return m_cost.nx(); }
    std::size_t ndx() const { return m_cost.ndx(); }

    ProblemTpl(const CostType& cost) : m_cost(cost), m_nc_total(0) {}

    ProblemTpl(const CostType& cost, const std::vector<ConstraintPtr>& constraints)
            : m_cost(cost), m_cstrs(constraints), m_nc_total(0)
    {
      reset_constraint_dim_vars();
    }

    /// Add a constraint to the problem, after initialization.
    void addConstraint(const ConstraintPtr& cstr)
    {
      m_cstrs.push_back(cstr);
      reset_constraint_dim_vars();
    }    

    std::vector<int> getIndices() const
    {
      return m_indices;
    }

    int getIndex(int i) const
    {
      return m_indices[i];
    }

  protected:
    /// Vector of equality constraints.
    std::vector<ConstraintPtr> m_cstrs;
    /// Total number of constraints
    int m_nc_total;
    std::vector<int> m_ncs;
    std::vector<int> m_indices;

    /// Set values of const data members for constraint dimensions
    void reset_constraint_dim_vars()
    {
      m_ncs.clear();
      m_indices.clear();
      int cursor = 0;
      for (std::size_t i = 0; i < m_cstrs.size(); i++)
      {
        const ConstraintPtr cstr = m_cstrs[i];
        m_ncs.push_back(cstr->nr());
        m_indices.push_back(cursor);
        cursor += cstr->nr();
      }
      m_nc_total = cursor;
    }

  };

  namespace helpers
  {
    /// @brief   Allocate a set of multipliers (or residuals) for a given problem instance.
    template<typename Scalar>
    void allocateMultipliersOrResiduals(
      const ProblemTpl<Scalar>& prob,
      typename math_types<Scalar>::VectorXs& data,
      typename math_types<Scalar>::VectorOfRef& out)
    {
      data.resize(prob.getTotalConstraintDim());
      data.setZero();
      using Problem = ProblemTpl<Scalar>;
      out.reserve(prob.getNumConstraints());
      int cursor = 0;
      int nr = 0;
      for (std::size_t i = 0; i < prob.getNumConstraints(); i++)
      {
        typename Problem::ConstraintPtr cstr = prob.getConstraint(i);
        nr = cstr->nr();
        out.emplace_back(data.segment(cursor, nr));
        cursor += nr;
      }
    }
    
  } // namespace helpers
  

} // namespace proxnlp


