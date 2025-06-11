import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np

downcast_to_bf16 = 1         # If 1, the allreduce tensors are downcasted from fp32 to bf16

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def install_package(package_name):
    try:
        # Use subprocess to call pip and install the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Package '{package_name}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package '{package_name}'. Error: {e}")
        exit()

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
 
def train_loop(rank, world_size, global_loss_list):
    print(f"Starting training on rank {rank} with explicit all_reduce...")
    setup(rank, world_size)
 
    # Determine the device for this process
    device = torch.device("hpu")
    print(f"Rank {rank} using device: {device}")

    set_seed(88)
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
                    print(f"BEFORE: param.grad.dtype = {param.grad.dtype}")
                    if downcast_to_bf16 == 1:
                        mytype=torch.bfloat16 # 16 bit
                        temp = param.grad.data
                        print(f"BEFORE: temp.dtype = {temp.dtype}")
                        temp = temp.to(mytype)
                        print(f"AFTER: temp.dtype = {temp.dtype}")
                        dist.all_reduce(temp, op=dist.ReduceOp.SUM)

                        # Upcast temp to fp32 before assigning back to param.grad.data
                        temp = temp.to(torch.float32)
                        param.grad.data = temp
                    else:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    
                    # This is independent of downcast_to_bf16
                    param.grad.data /= world_size
 
            optimizer.step()
 
            running_loss += loss.item()
 
            print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
 
        #avg_loss = running_loss / len(dataloader)

        # Calculate avg_loss across all ranks
        total_loss_tensor = torch.tensor(running_loss, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / len(dataloader) / world_size
       
        if rank == 0:
            global_loss_list.append(avg_loss)
            print(f"Epoch {epoch+1} finished. Global Average Loss: {avg_loss:.4f}")
 
    print(f"Training on rank {rank} finished.")
    cleanup()
 
def main():
    set_seed(88)
    world_size = 8
    print(f"Starting distributed training with world_size={world_size}")
    global_loss_list = mp.Manager().list()  # Shared list to store global loss
    mp.spawn(train_loop, args=(world_size, global_loss_list), nprocs=world_size, join=True)

    # Plot the global loss on rank 0
    plt.plot(global_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Global Average Loss')
    plt.title(f'Global Loss Over Epochs \n downcast_to_bf16 = {downcast_to_bf16}')
    plt.show()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file=f"loss_curve_{timestamp}.png"
    plt.savefig(out_file, format='png')
    print(f"Loss curve saved to {out_file}")
    print("All distributed HPU processes finished.")
 
if __name__ == "__main__":
    main()
