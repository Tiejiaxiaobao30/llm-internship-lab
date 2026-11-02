import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
print(y.grad) 
assert x.grad == 4.0
assert y.grad is None

x = torch.tensor(2.0, requires_grad=True)
a = x ** 2
y = a * 3 + 1
y.backward()
print(x.grad)
print(a.grad)
print(y.grad)
assert x.grad == 12.0
assert a.grad is None
assert y.grad is None
print(x.grad_fn) 
print(a.grad_fn) # <function PowBackward0 at 0x...>
print(y.grad_fn) # <function AddBackward0 at 0x...>

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y.detach()  # 取出y的值，但不计算梯度

with torch.no_grad(): #不计算梯度
    result = x * 3 

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
z = x ** 3
z.backward()  
print(x.grad)  # 计算梯度时，梯度会累加 4+12 = 16