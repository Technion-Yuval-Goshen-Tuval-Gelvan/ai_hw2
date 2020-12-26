import numpy as np
import time

a = np.zeros([10000, 1000])
a[15:17, 17:19] = 1

t = time.time()
r = np.argwhere(a == 1)
print(time.time()-t)