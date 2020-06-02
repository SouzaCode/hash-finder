import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hashlib

# Data for plotting
inputs = []
for i in range(0,100000,1000):
    inputs.append(i)
x = np.array(inputs)
outputs = []
for data in inputs:
    encode = hashlib.sha256()
    encode.update(bytes(data))
    outputs.append(int(encode.hexdigest(),16))
y = np.array(outputs)
freq = np.fft.fft(y)

plt.plot(x,y)
plt.show()