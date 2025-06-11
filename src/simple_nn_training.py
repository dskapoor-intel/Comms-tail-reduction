import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
 
try:
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.distributed.hccl
except ImportError:
    print("Habana PyTorch framework (habana_frameworks.torch) not found.")
    exit()
 
def setup(rank, world_size):
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} successfully initialized process group.")
 
def cleanup():
    dist.destroy_process_group()
    print("Process group destroyed.")
 
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
 
def train_loop(rank, world_size):
    print(f"Starting training on rank {rank} with explicit all_reduce...")
    setup(rank, world_size)
 
    # Determine the device for this process
    device = torch.device("hpu")
    print(f"Rank {rank} using device: {device}")
 
    # Create some dummy data
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000, 1)).float()
 
    dataset = TensorDataset(X, y)
 
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
 
    model = SimpleNN().to(device)
    #criterion=torch.nn.CrossEntropyLoss(reduction="mean")
    criterion = nn.BCELoss()
 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    num_epochs = 10
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
 
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
 
            optimizer.step()
 
            running_loss += loss.item()
 
            print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
 
        avg_loss = running_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
 
    print(f"Training on rank {rank} finished.")
    cleanup()
 
def main():
    world_size = 8
    print(f"Starting distributed training with world_size={world_size}")
    mp.spawn(train_loop, args=(world_size,), nprocs=world_size, join=True)
    print("All distributed HPU processes finished.")
 
if __name__ == "__main__":
    main()
