from polymorphic_test import X, Y, Z, getY, getZ, poly_use_base, create_vec_poly

assert isinstance(getY(), Y)
assert isinstance(getZ(), Z)
assert issubclass(Y, X)
assert issubclass(Z, X)

y = Y()
z = Z()

b = poly_use_base(y)
print(b.x)
assert isinstance(b.x, Y)

b = poly_use_base(z)
print(b.x)
assert isinstance(b.x, Z)

vec = create_vec_poly()
print(len(vec))
print(vec[0])
print(vec.tolist())
