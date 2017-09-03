"""Check how function shift and scale."""
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline


# Def target function to plot
def sigmoid(x):
    """Return sigmoid."""
    return 1 / (1 + np.exp(-x))


# Set target function
targetfunc = sigmoid

# Make data set
x = np.linspace(-30, 30, num=500)
y = sigmoid(x)

# We can shift function by subtract values from x
shift = 10
y_shift = targetfunc(x - shift)

# We can scale function by devide values from x
scale = 5
y_scale = targetfunc(x / scale)

y_both = targetfunc((x - shift) / scale)

ys = [
      {'y': y, 'args': {'label': 'y', 'color': 'b', 'ls': '-'}},
      {'y': y_shift, 'args': {'label': 'y shift', 'color': 'g', 'ls': '--'}},
      {'y': y_scale, 'args': {'label': 'y scale', 'color': 'y', 'ls': '--'}},
      {'y': y_both, 'args': {'label': 'y both', 'color': 'r', 'ls': '--'}},
      ]


# Plot
figtitle = 'Sample of function shift and scale'
save_fig = 'fig/sample_function_shift.png'


figsize = (10, 5)
fig, ax = plt.subplots(figsize=figsize)

for y_conf in ys:
    y = y_conf['y']
    args = y_conf['args']
    ax.plot(x, y, marker='', **args)

plt.legend()
plt.savefig(save_fig, dpi=200)

plt.suptitle(figtitle)
plt.show()
