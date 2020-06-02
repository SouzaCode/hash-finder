import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hashlib

# Data for plotting
inputs = []
for i in range(0,10000,100):
    inputs.append(i)
x = np.array(inputs)
outputs = []
for data in inputs:
    encode = hashlib.sha256()
    encode.update(bytes(data))
    outputs.append(int(encode.hexdigest(),16))
y = np.array(outputs)

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()