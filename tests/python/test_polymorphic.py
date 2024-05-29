from polymorphic_test import (
    X,
    Y,
    Z,
    getY,
    getZ,
    poly_use_base,
    create_vec_poly,
    poly_passthrough,
    get_const_y_poly_ref,
)

assert isinstance(getY(), Y)
assert isinstance(getZ(), Z)
assert issubclass(Y, X)
assert issubclass(Z, X)

y = Y()
print(y)
z = Z()
print(z)

b = poly_use_base(y)
print(b.x)
assert isinstance(b.x, Y)

b = poly_use_base(z)
print(b.x)
assert isinstance(b.x, Z)

vec = create_vec_poly()
assert len(vec) == 3
print(vec[0])
print(vec.tolist())

print("passthrough:")
py = poly_passthrough(y)
assert isinstance(py, Y)
print(py)
pz = poly_passthrough(z)
assert isinstance(pz, Z)
print(pz)

print("Get static Y:")
ry_static = get_const_y_poly_ref()
print(ry_static)
assert isinstance(ry_static, Y)
