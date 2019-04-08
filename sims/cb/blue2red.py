import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# see: https://matplotlib.org/gallery/color/custom_cmap.html

def cm(n=100, cmap_name='blue2red'):
    # Blue -> Grey -> Red
    colors = [(0, 0, 1), (.9, .9, .9), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    cm.set_bad(color='black')
    return cm

# Example:
# cm_b2r = cm(9)
# X = np.random.randn(100,5)
# plt.imshow(X, aspect='auto', vmin=-3, vmax=3, cmap=cm_b2r)
# plt.colorbar()
# plt.show()
