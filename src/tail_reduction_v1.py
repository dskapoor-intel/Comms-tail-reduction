# This benchmark script is meant to measure the tail latency in G2 and G3 with the latest stack
# e.g. 1.22.0-10

import torch
import habana_frameworks.torch.core as htcore
import torch.nn.functional as F
import torch.distributed as dist
import os
import habana_frameworks.torch as ht
import sys
import numpy as np
import subprocess
from dev_name import get_device_name
import math

install_packages=1

def install_package(package_name):
    try:
        # Use subprocess to call pip and install the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Package '{package_name}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package '{package_name}'. Error: {e}")

class EventSet:
    def __init__(self):
        self.start_comm_event = ht.hpu.Event(enable_timing=True)
        self.end_comm_event = ht.hpu.Event(enable_timing=True)

def my_init(device, world_size, myrank):

    torch.manual_seed(88)
    mytype=torch.float32  # 32 Bit
    dtype_size_bytes=32/8 #1 byte = 8 bits

    # Initialize communication tensor
    #tensor_size2 = 33554432 # 1 GB per rank for FP32
    #tensor_size2 = 3276800 # 100 MB per rank for FP32

    # 32 MB, 128 MB, 512 MB --> target msg sizes
    # With 8 nodes and 64 ranks
    # Data type - FP32
    #msg_size_bytes = 32 * 1024 * 1024 # 32 MB Global
    #msg_size_bytes = 64 * 1024 * 1024 # 64 MB Global
    #msg_size_bytes = 128 * 1024 * 1024 # 128 MB Global
    #msg_size_bytes = 512 * 1024 * 1024 # 512 MB Global
    #msg_size_bytes = 2048 * 1024 * 1024 # 2 GB Global
    msg_size_bytes = 4096 * 1024 * 1024 # 4 GB Global



    size_per_rank_bytes = math.ceil(msg_size_bytes / world_size)
    tensor_size2 = math.ceil(size_per_rank_bytes / dtype_size_bytes)
    msg_size_Mbytes = msg_size_bytes/(1024*1024)

    if myrank == 0:
        print(f"Data type = {mytype}")
        print(f"dtype_size_bytes = {dtype_size_bytes}")
        print(f"size_per_rank_bytes = {size_per_rank_bytes}")
        print(f"tensor_size2 = {tensor_size2}")
        print(f"msg_size_Mbytes = {msg_size_Mbytes}")

    tensor4 = torch.randn(tensor_size2, device=device, dtype=mytype)

    tensor_size_dummy = 2
    tensor_dummy = torch.randn(tensor_size_dummy, device=device, dtype=mytype)

    # Create events for timings
    my_events = EventSet()

    return tensor4, my_events, tensor_dummy,msg_size_Mbytes,mytype 

def main():

    iters = 4000
    arg1 = sys.argv[1]
    #print(f"arg1={arg1}")
    os.environ['MASTER_ADDR'] = arg1      # Replace with the master node's IP address
    os.environ['MASTER_PORT'] = '12345'   # Choose an open port



    # Initialize distributed process group with HCCL backend
    dist.init_process_group(backend='hccl', init_method='env://')
    myrank = dist.get_rank()
    world_size = dist.get_world_size()
    print("I am rank",myrank,". Total ranks=", world_size)

    device = torch.device("hpu")
    if (myrank == 0):
        print("Device is",device)

    # Install dependencies on each node
    if install_packages == 1 and myrank == 0:
        # These additional packages are needed for postprocessing on node 0 only.
        # And, any rank can install packages node-wide, so doing it on rank 0 only.
        install_package("matplotlib")
        install_package("datetime")

    # warmup comms
    tensor4_w, events_w, _, _, _ =  my_init(device, world_size,myrank)
    dist.all_reduce(tensor4_w, op=dist.ReduceOp.SUM)
    htcore.hpu.synchronize()
    #htcore.mark_step()
    print(f"[{myrank}] Warmup and sync completed!")

    # Initialization of tensors and events for the comm loop measurements
    tensor4, events_main, tensor0, msg_size_Mbytes, dtype =  my_init(device, world_size, myrank)

    # 1D array to accumulate the times across all iters per rank.
    #duration_comm_alliters = np.full(iters, np.inf)
    duration_comm_alliters = torch.full((iters,), float('inf'), device=device)

    # Start timing loop
    for ii in range(iters):

        if myrank == 0:
            print(f"Running iter:{ii}")

        # Run a dummy comm to simulate a barrier
        dist.all_reduce(tensor0, op=dist.ReduceOp.SUM)

        # Run and time the actual comm
        events_main.start_comm_event.record()
        dist.all_reduce(tensor4, op=dist.ReduceOp.SUM)
        events_main.end_comm_event.record()

        events_main.end_comm_event.synchronize()

        # Calculate elapsed time using events in ms
        duration_comm = ht.hpu.Event.elapsed_time(events_main.start_comm_event, events_main.end_comm_event)

        # Accumulate across ranks
        duration_comm_alliters[ii] = duration_comm
        htcore.hpu.synchronize()

    # Accumalte times across ranks for post-processing in rank 0
    # Prepare gather_list only on rank 0
    gather_list = None
    if myrank == 0:
        gather_list = [torch.zeros(iters, dtype=torch.float32, device=device) for _ in range(world_size)]

    # Accumalte times across ranks for post-processing in rank 0
    dist.gather(duration_comm_alliters, gather_list=gather_list, dst=0)

    # Stats
    if myrank == 0:
        import matplotlib.pyplot as plt
        from datetime import datetime 
       

        gathered_data = torch.stack(gather_list)
        data_min = torch.min(gathered_data)
        data_mean = torch.mean(gathered_data)
        data_max = torch.max(gathered_data)
        print(f"gathered_data.shape= {gathered_data.shape}")
        print(f"[{myrank}]data_min = {data_min}")
        print(f"[{myrank}]data_mean = {data_mean}")
        print(f"[{myrank}]data_max = {data_max}")

        gathered_data = gathered_data.to('cpu')
        flattened_data = gathered_data.flatten().numpy()
        plt.hist(flattened_data, bins=100, alpha=0.7, color='blue')
        #plt.hist(flattened_data, bins=25, range=(0.1, 0.4), alpha=0.7, color='blue')

        # Set the y-axis to a logarithmic scale
        plt.yscale('log')

        # Figure out if we are dealig with a Gaudi2 or 3.
        my_dev = get_device_name()

        plt.title(f'Histogram of Comms time w/ Allreduce @ {msg_size_Mbytes} MB w/ {dtype}\ncards={world_size}, {my_dev}, Iters={iters}, v1.22.0-10')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True)
        #plt.show()

        # Text box
        textstr = f'Max time: {data_max:.2f}\nMin time: {data_min:.2f}\nMean time: {data_mean:.2f}'
        plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(facecolor='white', alpha=0.5))

        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"timing_histogram_{timestamp}.png", format='png')

        # Save tensor to file for later post-processing
        out_file=f"timing_tensor_{timestamp}.pt"
        torch.save(gathered_data, out_file)
        print(f"Tensor saved to {out_file}")


if __name__ == '__main__':
    main()
