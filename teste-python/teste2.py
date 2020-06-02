from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

import hashlib

# Data for plotting
max = 1000
step = 100
n = int(max/step)

inputs = []
for i in range(0,max,step):
    inputs.append(i)
x = np.array(inputs)
outputs = []
for data in inputs:
    encode = hashlib.sha256()
    encode.update(bytes(data))
    outputs.append(int(encode.hexdigest(),16))
    #outputs.append(data * data)
y = np.array(outputs)

x = np.reshape(x ,[n, 1]) 
y = np.reshape(y ,[n ,])
nn = MLPRegressor(
    hidden_layer_sizes=(10,),   activation='tanh', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=10000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


n = nn.fit(x, y)
test_x = np.arange(0, max, int(step/4)).reshape(-1, 1)
test_y = nn.predict(test_x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=1, c='b', marker="s", label='real')
ax1.scatter(test_x,test_y, s=10, c='r', marker="o", label='NN Prediction')
plt.show()