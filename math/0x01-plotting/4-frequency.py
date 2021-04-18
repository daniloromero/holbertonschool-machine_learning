#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()

ax.hist(student_grades, bins=range(0, 110, 10), edgecolor='black')

ax.set_title('Project A')
ax.set_xlabel('Grades')
ax.set_ylabel('Number of students')
plt.axis([0, 100, 0, 30])
fig.tight_layout()
plt.show()
