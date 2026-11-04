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
test_dataset = datasets.MNIST(
    root='data/mnist',
    train=False,
    download=True,
    transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train() 
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        x = images.reshape(-1, 28*28) 
        logits = model(x)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        total_loss += loss.item() * images.size(0) 
        total_correct += (logits.argmax(dim=1) == labels).sum().item() 
        total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy

history = []
for epoch in range(5):
    train_loss, train_accuracy = train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn
    )
    history.append((train_loss, train_accuracy))
    print(
    f"Epoch {epoch + 1}/5 | "
    f"Loss: {train_loss:.4f} | "
    f"Accuracy: {train_accuracy:.4f}"
)

def evaluate(model, loader, loss_fn):
    model.eval() 
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            x = images.reshape(-1, 28*28)
            logits = model(x)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy

print("Evaluating on test set...")
test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


import matplotlib.pyplot as plt
from pathlib import Path

# 提取训练数据
epochs = range(1, len(history) + 1)
losses = [x[0] for x in history]
accuracies = [x[1] for x in history]
# 图片保存到当前 Python 文件所在目录
output_dir = Path(__file__).resolve().parent
# 第一张图：Loss
plt.figure()
plt.plot(epochs, losses, marker="o", label="Training Loss")
plt.xlabel("Epoch")
plt.xticks(epochs)
plt.ylabel("Training Loss")
plt.title("MNIST MLP Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "day05_loss.png")
plt.close()
# 第二张图：Accuracy
plt.figure()
plt.plot(epochs, accuracies, marker="o", label="Training Accuracy")
plt.xlabel("Epoch")
plt.xticks(epochs)
plt.ylabel("Training Accuracy")
plt.title("MNIST MLP Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "day05_accuracy.png")
plt.close()
