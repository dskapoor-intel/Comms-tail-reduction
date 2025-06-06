#!/bin/bash

# SET BEFORE RUNNING
###########################
PROFILE=0
RUN_ENV=CONTAINER
NUM_RANKS=64
CASE=3
MULTINODE=1
MYHOSTFILE=/etc/mpi/hostfile
export PT_HPU_LAZY_MODE=1
###########################

echo "******************************************************"
echo "PROFILE = $PROFILE"
echo "RUN_ENV = $RUN_ENV"
echo "NUM_RANKS = $NUM_RANKS"
echo "CASE = $CASE"
echo "MULTINODE = $MULTINODE"
echo "MYHOSTFILE = $MYHOSTFILE"
echo "PT_HPU_LAZY_MODE = $PT_HPU_LAZY_MODE"
echo "******************************************************"

# Profiling
if [ $PROFILE = 1 ]; then
	hl-prof-config -e off --phase=device-acq -b 256
        #hl-prof-config --use-template profile_api_with_nics
	#hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on #--> Didn't generate a .hltv file on the compute node in container as well!
        #hl-prof-config -e off --use-template profile_api_with_nics --fuser on --trace-analyzer on --> Didn't generate a .hltv file. 
	#hl-prof-config -e off --phase=device-acq -b 256 --use-template profile_api_with_nics --fuser on --trace-analyzer on #--> Didn't generate a .hltv file on the compute node in container as well!
        #export HABANA_PROF_CONFIG=/software/users/askapoor/TLP/1_comp_comm_overlap/benchmark/test2/Overlap-analysis/prof_config.json
	#echo $HABANA_PROF_CONFIG

	export HABANA_PROFILE=1
fi

# MPI launcher to use
if [ $RUN_ENV = "VM" ]; then
	MPI_LAUNCHER=/usr/local/share/openmpi/bin/mpirun
elif [ $RUN_ENV = "CONTAINER" ]; then
	MPI_LAUNCHER=/opt/amazon/openmpi/bin/mpirun
else
	echo "Invalid option provided in RUN_ENV variable! Valid choices are one of VM or CONTAINER"
	exit 1 # Exit with a non-zero status to indicate an error
fi

echo "MPI_LAUNCHER = $MPI_LAUNCHER"

MYPWD="/software/users/askapoor/TLP/2_tail_reduction"
cd $MYPWD
echo "MYPWD = $MYPWD"

# Find the MASTER_IP address
MASTER_NODE=$(head -n 1 $MYHOSTFILE)
echo "MASTER NODE = $MASTER_NODE"

ssh $MASTER_NODE "python3 $MYPWD/src/whats_my_ip.py $MYPWD"
cd $MYPWD
MY_MASTER_ADDR=$(head -n 1 ./ip_address.txt) 
export MY_MASTER_ADDR
echo "MY_MASTER_ADDR = $MY_MASTER_ADDR"

if [ $CASE = 1 ]; then
	#BIN_NAME="./src/overlap_v1.py"
	BIN_NAME="./src/tail_reduction_v1.py"
elif [ $CASE = 2 ]; then
	BIN_NAME="./src/tail_reduction_simplified.py"
elif [ $CASE = 3 ]; then
	BIN_NAME="./src/tail_reduction_v2.py"
elif [ $CASE = 4 ]; then
	BIN_NAME="./src/overlap_v4.py"
else
	echo "Incorrect CASE value selected. Valid range is 1 to 4"
	exit 1
fi

echo "BIN_NAME = $BIN_NAME"

if [ $MULTINODE = 1 ]; then
      # Multi-node run, hence use a hostfile
      # other --mca options: --mca btl openib,self --mca btl_tcp_if_include eth0 
      $MPI_LAUNCHER  --allow-run-as-root  --mca btl_tcp_if_include eth0 --hostfile $MYHOSTFILE  --bind-to core --map-by socket:PE=14 --rank-by core --report-bindings -x LD_LIBRARY_PATH -x HCL_ROOT -x SYNAPSE_ROOT -x BUILD_ROOT_LATEST -x GC_KERNEL_PATH -x ENGINES_FW_RELEASE_BUILD -x HABANA_PROFILE -x PT_HPU_LAZY_MODE -x LOG_LEVEL_PT_BRIDGE -x GRAPH_VISUALIZATION -x HABANA_LOGS -n $NUM_RANKS python3 $BIN_NAME $MY_MASTER_ADDR 
elif [ $MULTINODE = 0 ]; then
      # Single-node run     
      $MPI_LAUNCHER  --allow-run-as-root --mca btl_tcp_if_include eth0 -n $NUM_RANKS -x HABANA_PROFILE -x PT_HPU_LAZY_MODE python3 $BIN_NAME 
      #python3 $BIN_NAME 
else
    echo "Incorect MULTINODE value selected. Valid values 0 , 1"
    exit 1
fi


