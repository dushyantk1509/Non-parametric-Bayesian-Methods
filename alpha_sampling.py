import numpy as np

# ---- intialize variable -----
a = 1
b = 1
alpha = 1
n = 10
k = 3

# ----- Sampling alpha------
x = np.random.beta(alpha + 1, n)
pi_x = (a+k-1)*1.0 / ((a+k-1) + n*(b - np.log(x)))
pvals = [pi_x, 1-pi_x]
# print(pvals)

z = np.random.multinomial(1, pvals)
# print(z)

if z[0] == 1:
	alpha =  np.random.gamma(a+k, b-np.log(x))
else:
	alpha =  np.random.gamma(a+k-1, b-np.log(x))

print(alpha)