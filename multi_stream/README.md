## How to run

1. Run tuning to generate a log file
```bash
python3 tune_conv2d_cuda.py
```
It will output a log file `conv2d.log`


2. Replay the log file and measure the kernels under single-stream setting and multi-stream setting respectively
```bash
python3 replay.py
```

After replay the whole log, this script will output the performance of the best single-stream kernel (greedy kernel) and the best multi-stream kernel (collaborative kernel) (i.e. the table in the paper).
Replay here is required to make sure the accuracy of measurement.

