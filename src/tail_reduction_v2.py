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

save_plot_and_tensor=1       # If 1, generates and saves a histogram. Also saves the timing tensor for later post-processing.
control_growth_in_tensor4=1  # If 1, divides tensor4 in every iteration by a factor. Without this tensor4 eventually grows to -inf to inf.
check_inf_tensor4 = 0        # If 1, check for the presence of inf in tensor4. If 1, slows iters visibly. However, HPU min time doesn't change much!
downcast_to_bf16 = 0         # If 1, the allreduce tensor, i.e. tensor4 is downcasted from fp32 to bf16

def print_conig(save_plot_and_tensor, control_growth_in_tensor4, check_inf_tensor4, iters, arg1, world_size, device):
    print("***********************************************************")
    print("CONFIGURATION USED")
    print("***********************************************************")
    print(f"save_plot_and_tensor = {save_plot_and_tensor}")
    print(f"control_growth_in_tensor4 = {control_growth_in_tensor4}")
    print(f"check_inf_tensor4 = {check_inf_tensor4}")
    print(f"iters = {iters}")
    print(f"MASTER_ADDRESS = {arg1}")
    print(f"world_size = {world_size}")
    print(f"device = {device}")
    print("***********************************************************")

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
    mytype=torch.float32  # 32 bit
    dtype_size_bytes=32/8 # 1 byte = 8 bits

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
    #msg_size_bytes = 512 * 1024 * 1024 * 1024 # 512 GB Global

    size_per_rank_bytes = math.ceil(msg_size_bytes / world_size)
    tensor_size2 = math.ceil(size_per_rank_bytes / dtype_size_bytes)
    msg_size_Mbytes = msg_size_bytes/(1024*1024)

    tensor4 = torch.randn(tensor_size2, device=device, dtype=mytype)

    if downcast_to_bf16 == 1:
        mytype=torch.bfloat16 # 16 bit
        tensor4 = tensor4.to(mytype)

        # Check if the dtype and size are correct
        print(f"tensor4.dtype = {tensor4.dtype}")
        num_elements = tensor4.numel()
        element_size = tensor4.element_size()
        total_size_bytes = num_elements * element_size
        total_size_Mbytes = total_size_bytes / (1024 * 1024)
        print(f"Size of tensor4 = {total_size_Mbytes:.2f} MB")
        assert total_size_Mbytes == (msg_size_Mbytes/world_size)/2 , f"Assertion failed! total_size_Mbytes != (msg_size_Mbytes/world_size)/2. {total_size_Mbytes} != {(msg_size_Mbytes/world_size)/2} "

        # Updating some of the sizes for book keeping purposes
        # Assuming FP32 to bfloat16 downcast 
        dtype_size_bytes=dtype_size_bytes/2
        msg_size_bytes = msg_size_bytes/2 
        size_per_rank_bytes = size_per_rank_bytes/2 
        msg_size_Mbytes = msg_size_Mbytes/2

    tensor_size_dummy = 2
    tensor_dummy = torch.randn(tensor_size_dummy, device=device, dtype=mytype)

    if myrank == 0:
        print(f"Data type = {mytype}")
        print(f"dtype_size_bytes = {dtype_size_bytes}")
        print(f"size_per_rank_bytes = {size_per_rank_bytes}")
        print(f"tensor_size2 = {tensor_size2}")
        print(f"msg_size_Mbytes = {msg_size_Mbytes}")

    # Create events for timings
    my_events = EventSet()

    return tensor4, my_events, tensor_dummy,msg_size_Mbytes,mytype 

def main():

    iters = 4000 
    arg1 = sys.argv[1]
    #print(f"arg1={arg1}")
    os.environ['MASTER_ADDR'] = arg1     
    os.environ['MASTER_PORT'] = '12345'   # Choose an open port

    # Initialize distributed process group with HCCL backend
    dist.init_process_group(backend='hccl', init_method='env://')
    myrank = dist.get_rank()
    world_size = dist.get_world_size()
    print("I am rank",myrank,". Total ranks=", world_size)

    device = torch.device("hpu")
    if (myrank == 0):
        print_conig(save_plot_and_tensor, control_growth_in_tensor4, check_inf_tensor4, iters, arg1, world_size, device)

    # Install dependencies on each node
    if myrank == 0 and save_plot_and_tensor == 1:
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
    duration_comm_alliters_2 = torch.full((iters,), float('inf'), device=device)
    
    # Dummy collector - to simulate work
    dummy_collector = torch.full((iters,), float('inf'),device=device)

    #htcore.mark_step()
    # Start timing loop
    for ii in range(iters):

        if myrank == 0:
            print(f"Running iter:{ii}")

        # Run a dummy comm to simulate a barrier
        dist.all_reduce(tensor0, op=dist.ReduceOp.SUM)
        htcore.hpu.synchronize()
        
        htcore.mark_step()
        
        # Run and time the actual comm
        events_main.start_comm_event.record()
        dist.all_reduce(tensor4, op=dist.ReduceOp.SUM)
        #htcore.hpu.synchronize()
        events_main.end_comm_event.record()

        events_main.end_comm_event.synchronize()

        # Calculate elapsed time using events in ms
        duration_comm = ht.hpu.Event.elapsed_time(events_main.start_comm_event, events_main.end_comm_event)

        htcore.mark_step()

        # Accumulate across iterations
        duration_comm_alliters[ii] = duration_comm
        htcore.hpu.synchronize()

        # Check for the max time across all ranks in this iter and register that as the time taken for allreduce
        duration_comm_tensor = torch.tensor(duration_comm, device=torch.device("hpu"))
        dist.all_reduce(duration_comm_tensor, op=dist.ReduceOp.MAX, async_op=False)
        duration_comm_alliters_2[ii] = duration_comm_tensor


        # Use the result of allreduce to prevent deferred execution beyond this point
        dummy_collector[ii] = torch.max(tensor4)

        # Prevent tensor from growing to -inf or inf
        if control_growth_in_tensor4 == 1:
            # Specify the interval
            a = world_size - 0.05
            if a <= 0: 
                a = 1
            b = world_size
            random_number = torch.rand(1 , device=device) * (b - a) + a
            tensor4 = tensor4 / random_number

    # Accumulate times across ranks for post-processing in rank 0
    # Prepare gather_list only on rank 0
    gather_list = None
    if myrank == 0:
        gather_list = [torch.zeros(iters, dtype=torch.float32, device=device) for _ in range(world_size)]

    # Accumalte times across ranks for post-processing in rank 0
    dist.gather(duration_comm_alliters, gather_list=gather_list, dst=0)

    # Dummy collection to prevent the Graph compiler from optimizing
    gather_list2 = None
    if myrank == 0:
        gather_list2 = [torch.zeros(iters, dtype=torch.float32, device=device) for _ in range(world_size)]

    # Accumalte times across ranks for post-processing in rank 0
    dist.gather(dummy_collector, gather_list=gather_list2, dst=0)

    # Stats
    if myrank == 0:
        gathered_data = torch.stack(gather_list)
        data_min = torch.min(gathered_data)
        data_mean = torch.mean(gathered_data)
        data_max = torch.max(gathered_data)
        print(f"gathered_data.shape= {gathered_data.shape}")
        print(f"[{myrank}]data_min = {data_min}")
        print(f"[{myrank}]data_mean = {data_mean}")
        print(f"[{myrank}]data_max = {data_max}")

        print(f"[{myrank}]: This should match Theoretical times w.r.t global msg size: duration_comm_alliters_2 = {duration_comm_alliters_2}")

        if save_plot_and_tensor == 1:
            import matplotlib.pyplot as plt
            from datetime import datetime 

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

        # Print stats from dummy_collector
        # Not useful, but included to prevent the GC from optimizing
        gathered_data2 = torch.stack(gather_list2)
        dummy_min = torch.min(gathered_data2)
        dummy_mean = torch.mean(gathered_data2)
        dummy_max = torch.max(gathered_data2)

        print(f"gathered_data2.shape= {gathered_data2.shape}")
        print(f"[{myrank}]dummy_min = {dummy_min}")
        print(f"[{myrank}]dummy_mean = {dummy_mean}")
        print(f"[{myrank}]dummy_max = {dummy_max}")

if __name__ == '__main__':
    main()
