#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"


namespace lienlp
{
namespace python
{

  namespace context
  {
    
    using Scalar = double;

    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using VectorXBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    using Problem_t = ProblemTpl<Scalar>;
    using Result_t = SResults<Scalar>;
    using Workspace_t = SWorkspace<Scalar>;
    using Cost_t = CostFunctionBase<Scalar>;
    using Constraint_t = ConstraintSetBase<Scalar>;
    using Function_t = BaseFunction<Scalar>;
    using C1Function_t = C1Function<Scalar>;
    using C2Function_t = C2Function<Scalar>;

    using Manifold = ManifoldAbstractTpl<Scalar>;
    using Solver = SolverTpl<Scalar>;

    using VecFunc_t = void(const ConstVectorRef&, VectorRef) const;
    using VecFuncRet_t = VectorXs(const ConstVectorRef&) const;
    using MatFunc_t = void(const ConstVectorRef&, MatrixRef) const;
    using MatFuncRet_t = MatrixXs(const ConstVectorRef&) const;
    /// Signature of no-allocation vector-hessian product.
    using VHPFunc_t = void(const ConstVectorRef&, const ConstVectorRef&, MatrixRef) const;


  } // namespace context

} // namespace python
} // namespace lienlp


