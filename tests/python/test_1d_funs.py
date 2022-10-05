import proxnlp
from proxnlp import manifolds, utils, costs
import casadi as cas
import numpy as np
import pytest
import itertools


space = manifolds.R()
nx = space.nx

linesearch_strategies = [proxnlp.LinesearchStrategy.ARMIJO]

linesearch_interp_type = [
    proxnlp.LSInterpolation.BISECTION,
    proxnlp.LSInterpolation.QUADRATIC,
    proxnlp.LSInterpolation.CUBIC,
]

linesearch_opts = itertools.product(linesearch_strategies, linesearch_interp_type)


@pytest.mark.parametrize(("ls_strat", "ls_interp_type"), linesearch_opts)
def test_quad1d(ls_strat, ls_interp_type):
    print("OPTIONS:", ls_strat, ls_interp_type)
    x_sm = cas.SX.sym("x", 1)
    a = 0.01
    b = -0.53
    f = x_sm**2 * a + b * x_sm
    fs = utils.CasadiFunction(1, 1, f, x_sm, True)

    cost_fun = costs.CostFromFunction(fs)
    problem = proxnlp.Problem(space, cost_fun)
    solver = proxnlp.Solver(problem, 1e-5)
    ws = proxnlp.Workspace(problem)
    rs = proxnlp.Results(problem)
    solver.ls_strat = ls_strat
    solver.ls_options.interp_type = ls_interp_type
    flag = solver.solve(ws, rs, space.neutral(), [])

    real_solution = -b / (2 * a)

    print(rs.xopt)
    print("real sol:", real_solution)
    assert flag == proxnlp.ConvergenceFlag.success
    assert rs.xopt[0] == real_solution


@pytest.mark.parametrize(("ls_strat", "ls_interp_type"), linesearch_opts)
def test_cubic1d(ls_strat, ls_interp_type):
    print("OPTIONS:", ls_strat, ls_interp_type)
    x_sm = cas.SX.sym("x", 1)
    b = -10
    c = 10
    d = -24
    f = x_sm**3 + b * x_sm**2 + c * x_sm + d
    fs = utils.CasadiFunction(1, 1, f, x_sm, True)

    cost_fun = costs.CostFromFunction(fs)
    problem = proxnlp.Problem(space, cost_fun)
    solver = proxnlp.Solver(problem, 1e-5)
    ws = proxnlp.Workspace(problem)
    rs = proxnlp.Results(problem)
    x0 = np.array([1.0])
    solver.ls_strat = ls_strat
    solver.ls_options.interp_type = ls_interp_type
    flag = solver.solve(ws, rs, x0, [])
    assert flag == proxnlp.ConvergenceFlag.success
    assert rs.num_iters <= 7
    print(rs)
    # obtained from wolfram
    real_sol = 10 / 3 + np.sqrt(70) / 3
    assert np.allclose(real_sol, rs.xopt[0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
