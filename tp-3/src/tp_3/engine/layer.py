import numpy as np
from activation_functions.base_activation import ActivationFunction 

class Layer:

    def __init__(self, in_dimm: int, out_dimm: int, activation_function: ActivationFunction, initialization = None):
        if (initialization == None):
            self.W = np.random.randn(in_dimm, out_dimm) * .01
            self.b = np.zeros(out_dimm)
        else:
            raise NotImplementedError("Only default random initialization implemented")
        
        self.activation_function: ActivationFunction = activation_function

    def feed_forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b # z es la salida de la transformacion lineal de la capa
        return self.activation_function.compute(self.z)


    def backward(self, grad_output):
        # grad_output = dL/d(output)
        grad_z = grad_output * self.activation_function.gradient(self.z)
        grad_W = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0)

        grad_input = grad_z @ self.W.T

        return grad_input, grad_W, grad_b
