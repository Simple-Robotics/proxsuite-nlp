#pragma once

#include "proxnlp/helpers-base.hpp"
#include "proxnlp/solver-base.hpp"


namespace proxnlp
{
  namespace helpers
  {
    /** @brief  Store the history of results.
     */
    template<typename Scalar>
    struct history_callback : base_callback<Scalar>
    {
      history_callback(bool store_pd_vars=true,
                       bool store_values=true,
                       bool store_residuals=true)
                       : store_primal_dual_vars_(store_pd_vars)
                       , store_values_(store_values)
                       , store_residuals_(store_residuals)
      {}

      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

      struct
      {
        std::vector<VectorXs> xs;
        std::vector<VectorXs> lams;
        std::vector<VectorOfRef> lams_view;
        std::vector<Scalar> values;
        std::vector<Scalar> prim_infeas;
        std::vector<Scalar> dual_infeas;
        std::vector<VectorXs> ls_alphas;
        std::vector<VectorXs> ls_values;
        std::vector<Scalar> d1_s;
      } storage;

      void call(const WorkspaceTpl<Scalar>& workspace,
                const ResultsTpl<Scalar>& results)
      {
        if (store_primal_dual_vars_)
        {
          storage.xs.push_back(results.xOpt);
          storage.lams.push_back(results.lamsOpt_data);
          storage.lams_view.push_back(results.lamsOpt);
        }
        if (store_values_)
          storage.values.push_back(results.value);
        if (store_residuals_)
        {
          storage.prim_infeas.push_back(results.primalInfeas);
          storage.dual_infeas.push_back(results.dualInfeas);
        }
        const std::size_t asize = workspace.ls_alphas.size();
        storage.ls_alphas.push_back(Eigen::Map<const VectorXs>(&workspace.ls_alphas[0], asize));
        storage.ls_values.push_back(Eigen::Map<const VectorXs>(&workspace.ls_values[0], asize));
        storage.d1_s.push_back(workspace.d1);
      }

    protected:
      const bool store_primal_dual_vars_;
      const bool store_values_;
      const bool store_residuals_;
    };
    
  } // namespace helpers
} // namespace proxnlp

