from test_polymorphic_ext import (
    X,
    Y,
    Z,
    getY,
    getZ,
    X_wrap_store,
    poly_use_base,
    poly_store,
    set_return,
    poly_passthrough,
    call_method,
    vec_store,
)


class DerX(X):
    def name(self):
        return "My name is DerX and I'm from Python!"


class DerY(Y):
    def __init__(self):
        super().__init__()

    def name(self):
        return "DerivedY"


def test_subclasses():
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
dx = DerX()
print(dx)
call_method(dx)
dy = DerY()
print(dy)
call_method(dy)

dstore = X_wrap_store(dx)
print("dstore.x:", dstore.x)


def test_poly_base():
    y = Y()
    z = Z()
    d = DerX()
    e = DerY()

    print("=== test_poly_base ===")
    b = poly_use_base(y)
    del y
    x = b.x
    print(x)
    assert isinstance(x, Y)

    b = poly_use_base(z)
    del z
    x = b.x
    print(x)
    assert isinstance(x, Z)

    b = poly_use_base(d)
    del d
    x = b.x
    print(x, x.name())
    assert isinstance(x, DerX)

    b = poly_use_base(e)
    del e
    x = b.x
    print(x, x.name())
    assert isinstance(x, DerY)


def test_poly_store():
    y = Y()
    z = Z()
    d = DerX()
    e = DerY()

    print("=== test_poly_store ===")
    b = poly_store(y)
    del y
    x = b.x
    print(x)
    assert isinstance(x, X)
    assert isinstance(x, Y)

    print("poly_store(z)")
    b = poly_store(z)
    del z
    x = b.x
    print(x)
    assert isinstance(x, Z)

    b = poly_store(d)
    del d
    x = b.x
    print(x, x.name())

    b = poly_store(e)
    del e
    x = b.x
    print(x, x.name())


def test_vec_store():
    z = Z()
    d = DerX()
    e = DerY()
    print("=== vec_store ===")
    vs = vec_store([z, d])
    vs.add(e)
    del z
    del d
    del e
    vs_list = vs.get().tolist()
    print(vs_list)
    assert len(vs.get()) == 3
    assert isinstance(vs_list[0], Z)
    assert isinstance(vs_list[1], DerX)
    assert isinstance(vs_list[2], DerY)


test_subclasses()
test_poly_base()
test_poly_store()
test_vec_store()

print("passthrough:")
py = poly_passthrough(y)
assert isinstance(py, Y)
print("y", py)

pz = poly_passthrough(z)
print("z", pz)
assert isinstance(pz, Z)
assert isinstance(pz, Y)


# poly_passthrough() returns a copy, and the to-value converter for Poly does not test for bp::wrapper objects
pd = poly_passthrough(dx)
print("d", pd)
assert isinstance(pd, DerX)
pe = poly_passthrough(dy)
assert isinstance(pe, DerY)

print("Set static and return:")
r_stat = set_return(z)
print("r_stat (Z):", r_stat)
assert isinstance(r_stat, Z)
r_stat = set_return(dx)
print(r_stat, r_stat.name())
assert isinstance(r_stat, DerX)
