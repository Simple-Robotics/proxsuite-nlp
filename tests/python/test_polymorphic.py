from polymorphic_test import (
    X,
    Y,
    W,
    getY,
    getW,
    poly_use_base,
    poly_store,
    create_vec_poly,
    set_return,
    poly_passthrough,
    call_method,
    get_const_y_poly_ref,
)

assert isinstance(getY(), Y)
assert isinstance(getW(), W)
assert issubclass(Y, X)
assert issubclass(W, Y)

y = Y()
print(y)
call_method(y)
w = W()
print(w)


def test_poly_base():
    print("Use base")
    b = poly_use_base(y)
    print(b.x)
    assert isinstance(b.x, Y)

    b = poly_use_base(w)
    print(b.x)
    assert isinstance(b.x, W)


def test_poly_store():
    print("test_poly_store")
    b = poly_store(y)
    print(b.x)
    assert isinstance(b.x, Y)

    b = poly_store(w)
    # wrong, we get Y
    print(b.x)
    assert isinstance(b.x, W)


test_poly_base()
test_poly_store()

print("Create vector<PolyX>")
vec = create_vec_poly()
assert len(vec) == 3
print(vec[0])
print(vec.tolist())

print("passthrough:")
py = poly_passthrough(y)
assert isinstance(py, Y)
print("y", py)

pw = poly_passthrough(w)
print("w", pw)
assert isinstance(pw, W)  # error
assert isinstance(pw, Y)

print("Get static Y:")
ry_static = get_const_y_poly_ref()
print(ry_static)
assert isinstance(ry_static, Y)

print("Set static and return:")
r_stat = set_return(w)
print("r_stat (W):", r_stat)
assert isinstance(r_stat, W)
