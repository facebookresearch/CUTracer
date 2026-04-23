## VectorAdd (CUDA C++) ➕

Location: `tests/vectoradd`

Steps:
```bash
cd ~/CUTracer/tests/vectoradd
make && ./vectoradd
```

## Triton/Proton Vector Add with Histogram + IPC 📈

Location: `tests/proton_tests`

Steps:
```bash
cd ~/CUTracer/tests/proton_tests

# Recommended: use the wrapper
cutracer trace --analysis=proton_instr_histogram --kernel-filters=add_kernel \
  -- python ./vector-add-instrumented.py

# (Equivalent raw env-var form)
# CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
# CUTRACER_ANALYSIS=proton_instr_histogram \
# KERNEL_FILTERS=add_kernel \
# python ./vector-add-instrumented.py

python ./vector-add-instrumented.py

python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

## Hang Detection (Triton) ⛔

Location: `tests/hang_test`

Steps:
```bash
cd ~/CUTracer/tests/hang_test

# Recommended: use the wrapper
cutracer trace --analysis=deadlock_detection -- python ./test_hang.py

# (Equivalent raw env-var form)
# CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
# CUTRACER_ANALYSIS=deadlock_detection \
# python ./test_hang.py
```
