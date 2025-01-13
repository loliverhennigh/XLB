import numpy as np
import matplotlib.pyplot as plt

# Load data
drag = np.load('/home/oliver/store_xlb/drag.npy')
lift = np.load('/home/oliver/store_xlb/lift.npy')
time = np.load('/home/oliver/store_xlb/time.npy')
print('Drag:', drag)

# Plot
plt.figure()
plt.plot(time, drag, label='Drag')
plt.plot(time, lift, label='Lift')
plt.xlabel('Time (s)')
plt.ylabel('Coefficient')
plt.legend()
plt.grid()
plt.show()


