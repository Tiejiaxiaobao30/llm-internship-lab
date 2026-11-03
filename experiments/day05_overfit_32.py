import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
torch.manual_seed(42)

train_dataset = datasets.MNIST(
    root='data/mnist',
    train=True,
    download=True,
    transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        #self.fc1 = nn.Linear(...) 会将 Linear 注册为当前模型的子模块
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

images, labels = next(iter(train_loader)) # 获取一个批次的数据
x = images.reshape(-1, 28*28)
#或者 x = images.flatten(start_dim=1)(更推荐)
#将图像展平为一维向量 其中 -1 表示让 PyTorch 自动推导该维度大小
for step in range(200):
    logits = model(x)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy = (logits.argmax(dim=1) == labels).float().mean() # 计算准确率
    if step % 20 == 0:
        print(f"Step: {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在评估时不需要计算梯度
    logits = model(x)
    loss = loss_fn(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    print(f"Final Evaluation - Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")