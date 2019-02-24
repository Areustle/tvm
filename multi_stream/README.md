## How to run
1. Install the tvm and read the [tutorial about tuning for CUDA GPUs](https://docs.tvm.ai/tutorials/autotvm/tune_conv2d_cuda.html#sphx-glr-tutorials-autotvm-tune-conv2d-cuda-py)
The following scripts are modified from this tutorial.

2. Run tuning to generate a log file
```bash
python3 tune_conv2d_cuda.py
```
It will output a log file `conv2d.log`


3. Replay the log file and measure the kernels under single-stream setting and multi-stream setting respectively
```bash
python3 replay.py
```

After replay the whole log, this script will output the performance of the best single-stream kernel (greedy kernel) and the best multi-stream kernel (collaborative kernel) (i.e. the table in the paper).
Replay here is required to make sure the accuracy of measurement.

