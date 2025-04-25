# Comms-tail-reduction

Goals:
1. Study the behavior of communication times (i.e. tail times) on Haban devices in scale-up and scale-out scenarios
2. Study the effects of dropping random weights from distributed traiing workloads on model accuracy.
3. If the tail effects are significant (Goal 1) and if the weights can be dropped without significant loss in accuracy (Goal 2), then make a framework for dropping weights in distributed training workloads with the final goal of bringing down the communication volumes when running allreduce, which should speedup the training process.   
