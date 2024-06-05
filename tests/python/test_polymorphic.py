from polymorphic_test import (
    X,
    Y,
    Z,
    getY,
    getZ,
    X_wrap_store,
    poly_use_base,
    poly_store,
    create_vec_poly,
    set_return,
    poly_passthrough,
    call_method,
)


class DerX(X):
    def name(self):
        return "My name is DerX and I'm from Python!"


assert isinstance(getY(), Y)
assert isinstance(getZ(), Z)
assert issubclass(Y, X)
assert issubclass(Z, Y)

y = Y()
print(y)
call_method(y)
z = Z()
print(z)
call_method(z)
d = DerX()
print(d)
call_method(d)

dstore = X_wrap_store(d)
print("dstore.x:", dstore.x)


def test_poly_base():
    print("=== test_poly_base ===")
    b = poly_use_base(y)
    print(b.x)
    assert isinstance(b.x, Y)

    b = poly_use_base(z)
    print(b.x)
    assert isinstance(b.x, Z)

    b = poly_use_base(d)
    x = b.x
    print(x, x.name())
    assert isinstance(x, DerX)


def test_poly_store():
    print("=== test_poly_store ===")
    b = poly_store(y)
    print(b.x)
    assert isinstance(b.x, X)
    assert isinstance(b.x, Y)

    print("poly_store(z)")
    b = poly_store(z)
    print(b.x)
    assert isinstance(b.x, Z)

    b = poly_store(d)
    print(b.x, b.x.name())


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

pz = poly_passthrough(z)
print("z", pz)
assert isinstance(pz, Z)
assert isinstance(pz, Y)

pd = poly_passthrough(d)
print("d", pd)

print("Set static and return:")
r_stat = set_return(z)
print("r_stat (Z):", r_stat)
assert isinstance(r_stat, Z)
r_stat = set_return(d)
print(r_stat, r_stat.name())
assert isinstance(r_stat, DerX)
