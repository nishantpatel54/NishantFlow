import numpy as np
from activation import Activation
from layer import Layer

class Tanh(Activation):
    def __init__(self) -> None:
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x : 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self) -> None:
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        def sigmoid_prime(x):
            s=sigmoid(x)
            return s * (1-s(x))
        super().__init__(sigmoid, sigmoid_prime)

class SoftMax(Layer):
    def forward(self,input) -> None:
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, alpha):
        n = np.size(self.output)
        return np.dot((np.identity(n))*self.output, output_gradient)