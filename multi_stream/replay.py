import os
from random import getrandbits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.task.nnvm_integration import TaskExtractEnv
from topi.testing import conv2d_nchw_python

import tune_conv2d_cuda

in_file = "conv2d.log"

# create buffers
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

ctx = tvm.gpu()
a_tvm = tvm.nd.array(a_np, ctx=ctx)
w_tvm = tvm.nd.array(w_np, ctx=ctx)
c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)

def measure_func(func):
    try:
        evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=300, repeat=4)
        ss_res = evaluator(a_tvm, w_tvm, c_tvm)

        evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=300, repeat=-5004) 
        # -5003 means use 5 streams and repeat=3
        # if repeat < 0, then number = (-repeat) / 1000, repeat = (-repeat) % 1000
        ms_res = evaluator(a_tvm, w_tvm, c_tvm)
    except Exception:
        return None, None

    ss_res = list(ss_res.results)
    ms_res = list(ms_res.results)
    ss_res.sort()
    ms_res.sort()

    return np.mean(ss_res[1:-1]), np.mean(ms_res[1:-1])


ss = [] # single streams
ms = [] # multi streams

# replay records
for inp, res in autotvm.record.load_from_file(in_file):
    if res.error_no != 0:
        continue
    if np.mean(res.costs) > 0.01:  # filter too slow kernels
        continue

    with inp.target:
        tsk = autotvm.task.create(inp.task.name, inp.task.args,
                                  inp.target, template_key=inp.config.template_key)
        s, args = tsk.instantiate(inp.config)
        func = tvm.build(s, args, 'cuda')

    ss_res, ms_res = measure_func(func)
    if ss_res is None or ms_res is None:
        continue

    ss.append(ss_res)
    ms.append(ms_res)

    log_row = "%.3f\t%.3f" % (1000 * ss[-1], 1000 * ms[-1])
    with open("multi_stream_cmp.txt", "a") as f:
        f.write(log_row + "\n")
    print(log_row)


# sort
l1 = list(range(len(ss)))
l2 = list(range(len(ms)))

l1.sort(key=lambda x:ss[x])
l2.sort(key=lambda x:ms[x])

# print
print("================================================")
print("Test latency      | single stream | multi stream")
print("single stream best| %.3f ms      | %.3f ms" % (1000 * ss[l1[0]], 1000 * ms[l1[0]]))
print("multi stream best | %.3f ms      | %.3f ms" % (1000 * ss[l2[0]], 1000 * ms[l2[0]]))

