import numpy as np
from layer import Layer
class Activation(Layer):
    def __init__(self,activation, activation_p) -> None:
        self.activation=activation
        self.activation_p=activation_p

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    def backward(self, output_gradient, alpha):
        return np.multiply(output_gradient,self.activation_p(self.input))