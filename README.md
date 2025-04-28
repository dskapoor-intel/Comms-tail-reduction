# Comms-tail-reduction

# Goals
1. Study the behavior of communication times (specifically, tail times) on Habana devices in scale-up and scale-out scenarios
2. Study the effects of dropping random weights from distributed training workloads on model accuracy.
3. If the tail effects are significant (Goal 1) and if the weights can be dropped without significant loss in accuracy (Goal 2), then make a framework for dropping weights in distributed training workloads with the final goal of bringing down the communication volumes when running allreduce, which should speedup the training process.   

# Dependencies
If you already have the Habana stack installed (through VM/QNPU development environment or Habana containers or other means), no additional dependencies need to be installed.

# How to run?
Once the repository is cloned, run the following steps on a Gaudi-based machine.
```
chmod +x ./run.sh

# Edit some of the variables at the top of run.sh based on your preference.
./run.sh
```
