import proxsuite_nlp
from proxsuite_nlp import manifolds, costs
from proxsuite_nlp.casadi_utils import CasadiFunction
import casadi as cas
import numpy as np
import pytest
import itertools


space = manifolds.R()
nx = space.nx

linesearch_strategies = [proxsuite_nlp.LinesearchStrategy.ARMIJO]

linesearch_interp_type = [
    proxsuite_nlp.LSInterpolation.BISECTION,
    proxsuite_nlp.LSInterpolation.QUADRATIC,
    proxsuite_nlp.LSInterpolation.CUBIC,
]

linesearch_opts = itertools.product(linesearch_strategies, linesearch_interp_type)


@pytest.mark.parametrize(("ls_strat", "ls_interp_type"), linesearch_opts)
def test_quad1d(ls_strat, ls_interp_type):
    print("OPTIONS:", ls_strat, ls_interp_type)
    x_sm = cas.SX.sym("x", 1)
    a = 0.01
    b = -0.53
    f = x_sm**2 * a + b * x_sm
    fs = CasadiFunction(1, 1, f, x_sm, True)

    cost_fun = costs.CostFromFunction(fs)
    problem = proxsuite_nlp.Problem(space, cost_fun)
    solver = proxsuite_nlp.ProxNLPSolver(problem, 1e-5)
    solver.ls_strat = ls_strat
    solver.ls_options.interp_type = ls_interp_type
    solver.setup()
    flag = solver.solve(space.neutral(), [])
    rs = solver.getResults()

    real_solution = -b / (2 * a)

    print(rs.xopt)
    print("real sol:", real_solution)
    assert flag == proxsuite_nlp.ConvergenceFlag.success
    assert rs.xopt[0] == real_solution


@pytest.mark.parametrize(("ls_strat", "ls_interp_type"), linesearch_opts)
def test_cubic1d(ls_strat, ls_interp_type):
    print("OPTIONS:", ls_strat, ls_interp_type)
    x_sm = cas.SX.sym("x", 1)
    b = -10
    c = 10
    d = -24
    f = x_sm**3 + b * x_sm**2 + c * x_sm + d
    fs = CasadiFunction(1, 1, f, x_sm, True)

    cost_fun = costs.CostFromFunction(fs)
    problem = proxsuite_nlp.Problem(space, cost_fun)
    solver = proxsuite_nlp.ProxNLPSolver(problem, 1e-5)
    x0 = np.array([1.0])
    solver.ls_strat = ls_strat
    solver.ls_options.interp_type = ls_interp_type
    solver.setup()
    flag = solver.solve(x0, [])
    rs = solver.getResults()
    assert flag == proxsuite_nlp.ConvergenceFlag.success
    assert rs.num_iters <= 7
    print(rs)
    # obtained from wolfram
    real_sol = 10 / 3 + np.sqrt(70) / 3
    assert np.allclose(real_sol, rs.xopt[0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
