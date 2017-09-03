"""Check how function shift and scale."""
import numpy as np
import matplotlib.pyplot as plt
from utils import sigmoid, rand_range
import seaborn as sns
sns.set_style('whitegrid')

% matplotlib inline

def get_color(bin_values):
    return ['b' if x == 1 else 'g' for x in bin_values]

# Create 1-dim toy data
size = 100
range1_x_min = -10
range1_x_max = 10

x_shift = 0

np.random.seed(0)


x = rand_range(size, range1_x_min, range1_x_max)
prob = sigmoid(x, shift=x_shift)
z = np.random.binomial(n=1, p=prob)


fig, ax = plt.subplots()
ax.plot(x, prob, ls='', marker='.', color='0.5')
ax.scatter(x, z, color=get_color(z), alpha=0.5)
plt.show()


# Create 2-dim toy data
size = 1000
range1_x_min = -10
range1_x_max = 10

x_shift = 3

x = rand_range(size, range1_x_min, range1_x_max)
prob = sigmoid(x, shift=x_shift)
z = np.random.binomial(n=1, p=prob)

range1_y_min = -20
range1_y_max = 20
y = rand_range(size, range1_y_min, range1_y_max)

fig, ax = plt.subplots()
ax.scatter(x, y, color=get_color(z), alpha=0.5)
plt.show()



# Create 2-dim toy data get crossed

# Range x < 3
cond_1 = x < x_shift
prob_1 = sigmoid(y[cond_1], shift=-12, scale=1)
z_1 = np.random.binomial(n=1, p=prob_1)

z[cond_1] = z_1

# Range x >= 3
cond_2 = x >= x_shift
prob_2 = sigmoid(y[cond_2], shift=8, scale=3)
z_2 = 1 - np.random.binomial(n=1, p=prob_2)

z[cond_2] = z_2

fig, ax = plt.subplots()
ax.scatter(x, y, color=get_color(z), alpha=0.5)
plt.show()
