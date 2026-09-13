import numpy as np

rng = np.random.default_rng(0)
N = 8

# 1. Make two random decimal arrays of length N.
x = rng.uniform(-1, 1, N)
w = rng.uniform(-1, 1, N)

# 2. The float answer. This is what we're trying to match.
float_result = np.dot(x, w)

# 3. Find the scale for each array: biggest absolute value, divided by 127.
#    This is the "step size" -- the real number that one integer unit represents.
s_x = np.max(np.abs(x)) / 127
s_w = np.max(np.abs(w)) / 127

# 4. Convert to integers: divide by the scale, then round.
#    Use np.floor(v + 0.5) for rounding, NOT np.round (contract section 4).
#    Clip w to [-127, 127]. Then .astype(np.int8).
x_q = np.floor(x / s_x + 0.5).astype(np.int8)
w_q = np.floor(w / s_w + 0.5).astype(np.int8)

# 5. Integer dot product. Cast BOTH to np.int32 first or NumPy
#    will overflow int8 silently and give you nonsense.
acc = np.dot(x_q.astype(np.int32), w_q.astype(np.int32))

# 6. Convert back to a real number by multiplying by the combined scale.
int_result = acc * (s_x * s_w)

# 7. Compare.
print("float:", float_result)
print("int:  ", int_result)
print("relative error:", abs(int_result - float_result) / abs(float_result))