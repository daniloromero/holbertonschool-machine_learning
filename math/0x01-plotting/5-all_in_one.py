#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# data set up
y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Grid set up
fig1 = plt.figure(constrained_layout=True)
gs = fig1.add_gridspec(3, 2)
fig1.suptitle('All in one')
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[1, 0])
ax4 = fig1.add_subplot(gs[1, 1])
ax5 = fig1.add_subplot(gs[2, :])

# plotting data into grid
ax1.plot(y0, 'r')

ax2.scatter(x1, y1, c='magenta')
ax2.set_xlabel('Height (in)', size='x-small')
ax2.set_ylabel('Weight (lbs)', size='x-small')
ax2.set_title('Men\'s Height vs Weight', size='x-small')

ax3.plot(x2, y2)
ax3.set_yscale('log')
ax3.set_xlim(0, 28000)
ax3.set_title('Exponential Decay of C-14', size='x-small')
ax3.set_xlabel('Time (years)', size='x-small')
ax3.set_ylabel('Fraction Remaining', size='x-small')

ax4.plot(x3, y31, 'r--', label='C-14')
ax4.plot(x3, y32, c='g', label='Ra-226')
ax4.set_title('Exponential Decay of Radioactive Elements', size='x-small')
ax4.set_xlabel('Time (years)', size='x-small')
ax4.set_xlabel('Time (years)', size='x-small')
ax4.axis([0, 20000, 0, 1])
ax4.legend()

ax5.hist(student_grades, bins=range(0, 110,10), edgecolor='black')
ax5.set_title('Project A', size='x-small')
ax5.set_xlabel('Grades', size='x-small')
ax5.set_ylabel('Number of students', size='x-small')
ax5.axis([0, 100, 0, 30])
plt.show()
