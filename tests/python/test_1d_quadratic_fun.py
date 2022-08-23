import proxnlp
from proxnlp import manifolds, utils, costs
import casadi as cas


def test_quad1d():
    space = manifolds.R()
    nx = space.nx

    x_sm = cas.SX.sym("x", 1)
    a = 0.01
    b = -0.53
    f = x_sm ** 2 * a + b * x_sm
    fs = utils.CasadiFunction(1, 1, f, x_sm, True)

    cost_fun = costs.CostFromFunction(fs)
    problem = proxnlp.Problem(cost_fun)
    solver = proxnlp.Solver(space, problem, 1e-5)
    ws = proxnlp.Workspace(nx, nx, problem)
    rs = proxnlp.Results(nx, problem)
    flag = solver.solve(ws, rs, space.neutral(), [])

    real_solution = -b / (2 * a)

    print(rs.xopt)
    print("real sol:", real_solution)
    assert flag == proxnlp.ConvergenceFlag.success
    assert rs.xopt[0] == real_solution


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(sys.argv))
