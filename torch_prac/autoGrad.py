import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)  # requires_grad for parameters
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

#Computing gradients
loss.backward()
print(f"w.grad: {w.grad}")
print(f"b.grad: {b.grad}")
