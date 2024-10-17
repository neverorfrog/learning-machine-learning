from torch import Tensor, sin, cos, vmap, tensor

def function(x: Tensor) -> Tensor:
    return sin(x[0])*cos(x[1]) + sin(0.5*x[0])*cos(0.5*x[1])

vectorized_function = vmap(function, 0)

x1 = tensor([[1,1],[2,2]])
x2 = tensor([[1,1]])
print(vectorized_function(x1))