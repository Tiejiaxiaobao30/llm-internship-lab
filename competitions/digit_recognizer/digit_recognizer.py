import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

root = Path(__file__).resolve().parents[2]
DATA_DIR = root / "data" / "digit_recognizer"
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_df = pd.read_csv(DATA_DIR / "sample_submission.csv")

X = train_df.drop(columns=["label"])
y = train_df["label"]
X = torch.tensor(X.values, dtype=torch.float32) / 255.0
y = torch.tensor(y.values, dtype=torch.long)
X_test = torch.tensor(test_df.values,dtype=torch.float32) / 255.0

def create_dataloaders():
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataset =  TensorDataset(X,y)
    test_dataset = TensorDataset(X_test)
    train_subset,validation_subset = random_split(train_dataset,[int(len(train_dataset)*0.9),int(len(train_dataset)*0.1)],generator=generator)
    train_load=DataLoader(train_subset,batch_size=32,shuffle=True)
    validation_load=DataLoader(validation_subset,batch_size=32,shuffle=False)
    test_load=DataLoader(test_dataset,batch_size=32,shuffle=False)
    return train_load,validation_load,test_load

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
    def forward(self,x):
        x1=torch.relu(self.fc1(x))
        x2=torch.relu(self.fc2(x1))
        x3=self.fc3(x2)
        return x3

def train_one_epoch(model,loader,optimizer,loss_fn):
    model.train()
    total_loss=0.0
    total_correct=0
    total_samples=0
    for images,labels in loader:
        optimizer.zero_grad()
        x =images.flatten(start_dim=1)
        logits = model(x)
        loss = loss_fn(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples+=images.size(0)
    avg_loss = total_loss/total_samples
    avg_acc = total_correct/total_samples
    return avg_loss,avg_acc

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
            total_loss += loss.item()*images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples+=images.size(0)
    avg_loss = total_loss/total_samples
    avg_acc = total_correct/total_samples
    return avg_loss,avg_acc

config = {
    "model_name": "MLP",
    "input_dim": 784,
    "hidden_dims": [256, 128],
    "num_classes": 10,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 32,
    "global_seed": 42,
    "split_seed": 42,
    "train_val_ratio": [0.9, 0.1],
    "total_epochs": 5
}

if __name__ == "__main__":
    torch.manual_seed(42)
    train_load, validation_load, test_load = create_dataloaders()
    
    model=MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001) 

    history = []
    for epoch in range(5):
        train_loss,train_acc= train_one_epoch(model,train_load,optimizer,loss_fn)
        val_loss,val_acc=evaluate(model,validation_load,loss_fn)
        history.append((train_loss,train_acc,val_loss,val_acc))
        checkpoint = {
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history
    }
        torch.save(checkpoint,  Path(__file__).resolve().parent / "checkpoint.pth")
        print(f"epoch:{epoch+1}")
        print(f"Train Loss: {train_loss:.4f},Train Accuracy: {train_acc:.4f},Val Loss:{val_loss:.4f},Val Acc:{val_acc:.4f}")

    model.eval()
    predictions = []

    with torch.no_grad():
        for (images,) in test_load:
            logits = model(images)
            predicted = logits.argmax(dim=1)
            predictions.extend(predicted.tolist())

    sample_df["Label"] = predictions
    sample_df.to_csv(DATA_DIR / "submission.csv",index=False)
    print("Submission 文件已生成！")