import proxsuite_nlp
import pytest
import numpy as np


class TestClass:
    space = proxsuite_nlp.manifolds.VectorSpace(1)
    cost = proxsuite_nlp.costs.CostSum(1, 1)
    cost.add_component(
        proxsuite_nlp.costs.QuadraticDistanceCost(
            space, space.neutral(), np.zeros((1, 1))
        )
    )
    problem = proxsuite_nlp.Problem(space, cost, [])

    def test_print_options(self):
        options = proxsuite_nlp.LinesearchOptions()
        print(options)

    def test_print_sum_cost(self):
        print(self.cost)

    def test_print_results(self):
        res = proxsuite_nlp.Results(self.problem)
        print(res)

    def test_print_solver(self):
        solver = proxsuite_nlp.ProxNLPSolver(self.problem)
        print(solver)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
