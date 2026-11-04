import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)  # Output: torch.Size([2, 3])
print(x.size())   # Output: torch.Size([2, 3])
print(x.ndim)    # Output: 2
print(x[0])    # Output: tensor([1, 2, 3])
print(x[:,1])    # Output: tensor([2, 5])
print(x.reshape(3, 2))  # Output: tensor([[1, 2], [3, 4], [5, 6]])
print(x.transpose(0, 1))  # Output: tensor([[1, 4], [2, 5], [3, 6]])


#广播
A = torch.tensor([[1, 2, 3], [4, 5, 6]]) #A.shape = torch.Size([2, 3])
B = torch.tensor([[10, 20, 30]])  # B.shape = torch.Size([1, 3]) 
C = torch.tensor([[1], [2]])   # C.shape = torch.Size([2, 1])
D = torch.tensor([1, 2])     # D.shape = torch.Size([2]) 也就是（2，）一维向量，2个元素


A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = A.reshape(3, 2)
print(A @ B)  # Output: tensor([[22, 28], [49, 64]])
print(A.matmul(B))  # Output: tensor([[22, 28], [49, 64]])

x = torch.randn(2, 3, 4) 
# Batch Size = 2 Sequence Length = 3 Hidden Size = 4
wq = torch.randn(4, 4)
wk = torch.randn(4, 4)
Q = x @ wq
K = x @ wk
scores = Q @ K.transpose(-2, -1) / (4 ** 0.5) #除以根号下k的维度
print(scores.shape)  # Output: torch.Size([2, 3, 3])
