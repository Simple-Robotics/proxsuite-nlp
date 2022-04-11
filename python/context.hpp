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

    using Problem = ProblemTpl<Scalar>;
    using Results = SResults<Scalar>;
    using Workspace = SWorkspace<Scalar>;
    using Cost = CostFunctionBaseTpl<Scalar>;
    using Constraint = ConstraintSetBase<Scalar>;
    using Function = BaseFunctionTpl<Scalar>;
    using C1Function = C1FunctionTpl<Scalar>;
    using C2Function = C2FunctionTpl<Scalar>;

    using Manifold = ManifoldAbstractTpl<Scalar>;
    using Solver = SolverTpl<Scalar>;

    // func pointer signatures
    using VecFuncType = void(const ConstVectorRef&, VectorRef) const;
    using MatFuncType = void(const ConstVectorRef&, MatrixRef) const;
    using VHPFuncType = void(const ConstVectorRef&, const ConstVectorRef&, MatrixRef) const;

    // allocated func signatures
    using VecFuncRetType = VectorXs(const ConstVectorRef&) const;
    using MatFuncRetType = MatrixXs(const ConstVectorRef&) const;
    using VHPFuncRetType = MatrixXs(const ConstVectorRef&, const ConstVectorRef&) const;


  } // namespace context

} // namespace python
} // namespace lienlp


