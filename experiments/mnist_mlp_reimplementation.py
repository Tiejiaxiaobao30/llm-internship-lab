import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
torch.manual_seed(1024)

train_dataset = datasets.MNIST(
    download=True,
    train=True,
    root="data/mnist",
    transform=transforms.ToTensor()  #少了括号
)
train_load=DataLoader(train_dataset,batch_size=32,shuffle=True)

test_dataset = datasets.MNIST(
    download=True,
    train=False,
    root="data/mnist",
    transform=transforms.ToTensor() #少了括号
)
test_load=DataLoader(test_dataset,batch_size=32,shuffle=False)

class MLP(nn.Module):   #大小写 没有s
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
    def forward(self,x):    #缺少输入self和x
        x1=torch.relu(self.fc1(x))
        x2=torch.relu(self.fc2(x1))
        x3=self.fc3(x2)
        return x3

model=MLP()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)  #少了s

def train_one_epoch(model,loader,optimizer,loss_fn):
    model.train()  #缺少训练模式
    total_loss=0.0
    total_correct=0
    total_samples=0
    for images,labels in loader:  #要用输入里的loader
        optimizer.zero_grad()
        x =images.flatten(start_dim=1)  #漏了_dim
        logits = model(x)
        loss = loss_fn(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*images.size(0)  #掉了.size
        total_correct += (logits.argmax(dim=1) == labels).sum().item() #item后掉了括号
        total_samples+=images.size(0)  #掉了.size
    avg_loss = total_loss/total_samples
    avg_acc = total_correct/total_samples
    return avg_loss,avg_acc

history = []
for epoch in range(10):
    train_loss,train_acc= train_one_epoch(model,train_load,optimizer,loss_fn)
    history.append((train_loss,train_acc))
    print(f"epoch:{epoch+1}")
    print(f"Train Loss: {train_loss:.4f},Train Accuracy: {train_acc:.4f}")

def evaluate(model,loader,loss_fn):
    model.eval()
    total_loss=0.0
    total_correct=0
    total_samples=0
    with torch.no_grad():
        for images,labels in loader:  
            x =images.flatten(start_dim=1)  
            logits = model(x)
            loss = loss_fn(logits,labels)
            total_loss += loss.item()*images.size(0)  #掉了.size
            total_correct += (logits.argmax(dim=1) == labels).sum().item() #item后掉了括号
            total_samples+=images.size(0)  #掉了.size
    avg_loss = total_loss/total_samples
    avg_acc = total_correct/total_samples
    return avg_loss,avg_acc

test_loss,test_acc = evaluate(model,test_load,loss_fn)
print(f"Test Loss: {test_loss:.4f},Test Accuracy: {test_acc:.4f}")

            
