from Variable import Variable
import numpy as np

class Function:
    def __call__(self, input : Variable):
        X = input.data
        y = self.forward(X)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        X = self.input.data
        grad = gy * 2 * X
        return grad
        

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        X = self.input.data
        grad = gy * np.exp(X)
        return grad
