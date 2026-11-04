import torch
import torch.nn as nn
from train_mlp_checkpoint import (MLP,train_one_epoch,evaluate,create_dataloaders)

# 从硬盘读取 Checkpoint
checkpoint = torch.load(
    "experiments/checkpoint.pth",
    map_location="cpu",
    weights_only=True
)
config = checkpoint["config"]
train_load, validation_load, test_load = create_dataloaders()
loss_fn = nn.CrossEntropyLoss()
# 创建一个新模型和新优化器
new_model = MLP()
new_optimizer = torch.optim.Adam(new_model.parameters(), lr=config["learning_rate"])

# 恢复保存的模型参数
new_model.load_state_dict(checkpoint["model_state_dict"])
new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"]

history = checkpoint["history"]
for epoch in range(start_epoch,config["total_epochs"]):
    train_loss,train_acc= train_one_epoch(new_model,train_load,new_optimizer,loss_fn)
    val_loss,val_acc=evaluate(new_model,validation_load,loss_fn)
    history.append((train_loss,train_acc,val_loss,val_acc))
    checkpoint = {
    "epoch": epoch+1,
    "model_state_dict": new_model.state_dict(),
    "optimizer_state_dict": new_optimizer.state_dict(),
    "config": checkpoint["config"],
    "history": history
}
    torch.save(checkpoint, "experiments/checkpoint.pth")
    print(f"epoch:{epoch+1}")
    print(f"Train Loss: {train_loss:.4f},Train Accuracy: {train_acc:.4f},Val Loss:{val_loss:.4f},Val Acc:{val_acc:.4f}")

test_loss, test_acc = evaluate(new_model, test_load, loss_fn)
print(f"Test Loss: {test_loss:.4f},Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    checkpoint = torch.load(
    "experiments/checkpoint.pth",
    map_location="cpu",
    weights_only=True
)

    print(checkpoint["epoch"])
    print(len(checkpoint["history"]))

