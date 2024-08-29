from test_polymorphic_ext import (
    X,
    Y,
    Z,
    getY,
    getZ,
    poly_use_base,
    poly_store,
    # set_return,
    poly_passthrough,
    call_method,
    vec_store,
)


class DerX(X):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

    def name(self):
        return "My name is DerX and I'm from Python! Msg {:s}".format(self.msg)


class DerY(Y):
    def __init__(self):
        super().__init__()

    def name(self):
        return "My name is DerY, derived in Python from Y!"


def test_subclasses():
    assert isinstance(getY(), Y)
    assert isinstance(getZ(), Z)
    assert issubclass(Y, X)
    assert issubclass(Z, Y)


print("======================================")
print("=========== START TESTING ============")
print("======================================")

x = X()
call_method(x)

y = Y()
call_method(y)

z = Z()
call_method(z)

dx = DerX("prout")
call_method(dx)

dy = DerY()
call_method(dy)


def test_poly_base():
    y = Y()
    z = Z()
    dx = DerX("ha")
    e = DerY()

    print("\n===== test_poly_base =====")
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

    b = poly_use_base(dx)
    del dx
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
    d = DerX("hoo")
    dy = DerY()

    print("\n===== test_poly_store =====")
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
    x.msg = "holla"
    assert d.msg == "hoo"
    del d
    x = b.x
    print(x, x.name())
    assert isinstance(x, DerX)

    b = poly_store(dy)
    del dy
    x = b.x
    print(x, x.name())
    assert isinstance(x, DerY)


def test_vec_store():
    z = Z()
    d = DerX("heehee")
    e = DerY()
    print("\n===== test_vec_store =====")
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

py = poly_passthrough(y)
assert isinstance(py, Y)
print("y", py)

pz = poly_passthrough(z)
print("z", pz)
assert isinstance(pz, Z)
assert isinstance(pz, Y)


# poly_passthrough() returns a copy, and the to-value converter for Poly does not test for bp::wrapper objects
# pd = poly_passthrough(dx)
# print("> pass return dx:", pd)
# assert isinstance(pd, DerX)
# pe = poly_passthrough(dy)
# print("> pass return dy:", pe)
# assert isinstance(pe, DerY)

# print("Set static and return:")
# r_stat = set_return(z)
# print("r_stat (Z):", r_stat)
# assert isinstance(r_stat, Z)
# r_stat = set_return(dx)
# print(r_stat, r_stat.name())
# assert isinstance(r_stat, DerX)
