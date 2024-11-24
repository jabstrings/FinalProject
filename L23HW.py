#import modules
import matplotlib.pyplot as plt
import numpy as np

#define x and y values
x = np.arange(0, 5*np.pi, 0.1)
y = np.sin(x)

#plot sine curve
plt.plot(x, y, color='blue')
plt.show()