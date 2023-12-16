import numpy as np

from proxnlp import manifolds, costs


def test_cost_ops():
    space = manifolds.SE3()
    weights = np.eye(space.ndx)
    cc1 = costs.QuadraticDistanceCost(space, space.rand(), weights)
    cc2 = costs.QuadraticDistanceCost(space, space.rand(), weights)
    cc3 = cc1 + cc2
    assert cc3.weights.tolist() == [1, 1]
    print("sum:", cc3)
    cc4 = cc1 * 0.5
    assert cc4.weights.tolist() == [0.5]
    print("mul:", cc4)
    print(0.5 * cc2)
    cc5 = 0.2 * cc3
    print("rmul:", cc5)
    assert cc5.weights.tolist() == [0.2, 0.2]
    cc5 += cc1
    print(cc5)
    assert cc5.weights.tolist() == [0.2, 0.2, 1.0]


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
