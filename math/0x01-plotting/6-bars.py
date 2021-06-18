#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

print(fruit)
# The position of the bars on the x-axis
r = np.arange(len(fruit[0]))
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
people = ['Farrah', 'Fred', 'Felicia']
plt.bar(people, color=colors[0], label=fruits[0], height=fruit[0], width=0.5)
plt.bar(people, color=colors[1], label=fruits[1], height=fruit[1],
        bottom=np.array(fruit[0]), width=0.5)
plt.bar(people, color=colors[2], label=fruits[2], height=fruit[2],
        bottom=np.array(fruit[0])+np.array(fruit[1]), width=0.5)
plt.bar(people, color=colors[3], label=fruits[3], height=fruit[3],
        bottom=np.array(fruit[0])+np.array(fruit[1])+np.array(fruit[2]),
        width=0.5)
plt.xticks(r, people)
plt.ylim(0, 80)
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.legend()
plt.show()
