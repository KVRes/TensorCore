from Variable import Variable
import numpy as np

class Function:
    def __call__(self, input : Variable):
        X = input.data
        y = self.forward(X)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
