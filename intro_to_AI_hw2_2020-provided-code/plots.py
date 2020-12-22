import matplotlib
import matplotlib.pyplot as plt
import numpy as np

depth_diff = np.array([0, 1, 2])
stage_grade = np.array([4/5, 4/5, 2/5])
fig, ax = plt.subplots()

ax.plot(depth_diff, stage_grade)
plt.axis([0, 2, 0, 1])
plt.xticks([0, 1, 2])
plt.yticks([0, 1/5, 2/5, 3/5, 4/5, 1])

ax.set(xlabel='Depth Difference', ylabel='Stage Grade',
       title='Experiment 1')
ax.grid()

plt.show()



depth_diff = np.array([0, 1, 2])
stage_grade = np.array([5/5, 4/5, 3/5])
fig, ax = plt.subplots()

ax.plot(depth_diff, stage_grade)
plt.axis([0, 2, 0, 1])
plt.xticks([0, 1, 2])
plt.yticks([0, 1/5, 2/5, 3/5, 4/5, 1])

ax.set(xlabel='Depth Difference', ylabel='Stage Grade',
       title='Experiment 2')
ax.grid()

plt.show()