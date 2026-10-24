# A Brief Introduction to PyTorch DDP and FSDP Model Training

## Useful Commands

### 1. Allocate Compute Nodes

Single-node allocation (4 GPUs):


```
salloc --nodes=1 --qos=interactive --time=01:00:00 \
  --constraint=gpu --gpus-per-node=4 \
  --account=<acc_name>_g
```

Multi-node allocation (2 nodes, 4 GPUs per node):


```
salloc --nodes=2 --qos=interactive --time=00:30:00 \
  --ntasks-per-node=4 --cpus-per-task=32 \
  --constraint=gpu --gpus-per-node=4 \
  --account=<acc_name>_g
```

### 2. Load PyTorch



```
module load pytorch/2.6.0
```

### 3. Run Multi-Node Training

From the project directory, run:



```
./multinode_setup_and_run.sh <filename>.py
```

For example:



```
./multinode_setup_and_run.sh 7_mnist_ddp_pt_multinode.py
```

### 4. Debugging Tips

- Investigate slow imports: Use Python's import-time profiling to identify modules that take a long time to load.
  bash
  ```
  python -X importtime file.py
  ```
- Flush output immediately: When debugging distributed jobs, use `flush=True` to ensure that print output is written promptly.
  python
  ```
  print("test", flush=True)
  ```

## Course Contents

### Day 1: Introduction to Distributed Training and DDP

The first day introduces PyTorch distributed communication, single-GPU training, and the fundamentals of implementing Distributed Data Parallel (DDP).

#### Files and Examples

- `0_torch_dist.py` — Launches four processes and demonstrates `all_reduce` using distributed communication.
- `1_torch_dist_gpu.py` — Demonstrates GPU-based `all_reduce` across four processes.
- `2_mnist_training.py` — Implements a basic MNIST image classification pipeline using a single GPU.
- `3_mnist_distributed.py` — Launches four processes that independently perform the same MNIST training task, illustrating why distributed execution alone does not guarantee efficient data-parallel training.
- `4_mnist_manual_ddp.py` — Implements DDP manually by synchronizing gradients across processes and using a distributed data sampler.
- `5_mnist_ddp_pt.py` — Uses PyTorch's built-in `DistributedDataParallel` (DDP) wrapper to simplify gradient synchronization, eliminating the need for manual `all_reduce` operations on model gradients.

### Day 2: Multi-Node DDP, Streaming Data, and FSDP

The second day extends distributed training across multiple nodes, introduces streaming data loading, and explores Fully Sharded Data Parallel (FSDP) for memory-efficient model training.

#### Useful Commands

Allocate two nodes with four GPUs per node:



```
salloc --nodes=2 --qos=interactive --time=00:30:00 \
  --ntasks-per-node=4 --cpus-per-task=32 \
  --constraint=gpu --gpus-per-node=4 \
  --account=<acc_name>_g
```

Load PyTorch:



```
module load pytorch/2.6.0
```

Launch a multi-node training script:



```
./multinode_setup_and_run.sh <filename>.py
```

#### Files and Examples

- `6_torch_dist_multi_node.py` — Demonstrates distributed `all_reduce` communication across multiple nodes.
- `7_mnist_ddp_pt_multinode.py` — Implements MNIST training using PyTorch DDP across multiple nodes and GPUs.
- `8_mnist_ddp_pt_multinode_streaming.py` — Extends multi-node DDP training with a streaming data loader that reads data from disk in chunks.
- `create_mnist_h5.py` — Creates a chunked MNIST dataset in HDF5 format for streaming data-loading experiments.
- `9_fsdp_bench.py` — Demonstrates multi-GPU MNIST training with Fully Sharded Data Parallel (FSDP) and profiles memory usage.

## Learning Objectives

By the end of these two sessions, participants should be able to:

- Understand the basics of PyTorch distributed communication and collective operations such as `all_reduce`.
- Distinguish between independent multi-process execution and synchronized data-parallel training.
- Implement distributed training manually and with PyTorch DDP.
- Run distributed training jobs across multiple nodes and GPUs.
- Use streaming data loading to handle datasets stored on disk.
- Understand how FSDP shards model parameters, gradients, and optimizer states to reduce per-GPU memory requirements.
- Profile memory usage and compare distributed training approaches.
