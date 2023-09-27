from dense import Dense
from activation_funcs import Tanh
from loss_funcs import mse, mse_prime
import numpy as np

X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

epochs =10000
alpha = 0.01
for e in range(epochs):
    error = 0
    for x,y in zip(X,Y):
        output = x
        for layer in network:
            output = layer.forward(output)

        error += mse(y, output)

        grad = mse_prime(y,output)

        for layer in reversed(network):
            grad = layer.backward(grad,alpha)
        
    error /= len(X)
    print('%d/%d epochs, error=%f' % (e+1, epochs, error))

print('enter any key: ')
temp = input()