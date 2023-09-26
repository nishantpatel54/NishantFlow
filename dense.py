import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size) -> None:
        self.weights=np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)
    def forward(self, input):
        self.input=input
        return np.dot(self.weights,self.input)+self.bias
    def backward(self, output_gradient, alpha):
        weights_gradient=np.dot(output_gradient,self.input.T)
        input_gradient = np.dot(self.weights.T,output_gradient)
        self.weights -= alpha*weights_gradient
        self.bias -= alpha*output_gradient
        return input_gradient