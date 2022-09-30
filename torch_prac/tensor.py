import torch
import numpy as np


data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

tensor = torch.rand(3, 4)

if torch.cuda.is_available():
    tensor = tensor.to("cuda")


tensor = torch.ones(4, 4)
tensor[:, 1] = 0
t1 = torch.cat([tensor, tensor, tensor], dim=1)

# matrix multiplication. All same result
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# element-wise operation
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single tensor
agg = tensor.sum()
agg_item = agg.item()
tensor.add_(5)

#Bridge with numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

