import torch

checkpoint = torch.load(
    "experiments/checkpoint.pth",
    map_location="cpu",
    weights_only=True
)

print(checkpoint["epoch"])
print(len(checkpoint["history"]))