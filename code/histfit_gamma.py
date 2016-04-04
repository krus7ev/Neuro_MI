import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


gamma = stats.gamma

a, loc, scale = 3, 0, 2
size = 20000

y = gamma.rvs(a, loc, scale, size=size)

x = np.linspace(0, y.max(), 1000)

# fit
param = gamma.fit(y, floc=0)
pdf_fitted = gamma.pdf(x, *param)



plt.figure(dpi=200)
plt.plot(x, pdf_fitted, color='black')
plt.hist(y, normed=True,  bins=30, align='right', color='white')
plt.xlabel('t', fontsize=15)
plt.ylabel('$f (t)$', fontsize=15)

plt.show()
