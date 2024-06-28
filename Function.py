from Variable import Variable
import Utils as u
import numpy as np

class Function:
    def __call__(self, input : Variable):
        X = input.data
        y = self.forward(X)
        output = Variable(u.as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
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

def square(x):
    return Square()(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        X = self.input.data
        grad = gy * np.exp(X)
        return grad

def exp(x):
    return Exp()(x)
