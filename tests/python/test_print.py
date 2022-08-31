import proxnlp
import pytest
import numpy as np


class TestClass:
    space = proxnlp.manifolds.VectorSpace(1)
    cost = proxnlp.costs.CostSum(1, 1)
    cost.add_component(proxnlp.costs.QuadraticDistanceCost(space, space.neutral(), np.zeros((1, 1))))
    problem = proxnlp.Problem(cost, [])

    def test_print_options(self):
        options = proxnlp.LSOptions()
        print(options)

    def test_print_sum_cost(self):
        print(self.cost)

    def test_print_results(self):
        res = proxnlp.Results(1, self.problem)
        print(res)

    def test_print_solver(self):
        solver = proxnlp.Solver(self.space, self.problem)
        print(solver)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
