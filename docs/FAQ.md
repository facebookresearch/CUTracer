## Do I need to recompile my app?
No. CUTracer attaches via `CUDA_INJECTION64_PATH`.

## How to reduce overhead?
Prefer `proton_instr_histogram` (auto-enables `opcode_only`), filter kernels, narrow instruction intervals.

## Can I trace registers and memory simultaneously?
Yes. Use `CUTRACER_INSTRUMENT=reg_trace,mem_trace`. Expect higher overhead and larger outputs.

## Why are regions empty?
Your kernel may not execute clock instructions. Insert scopes (e.g., Triton `pl.scope`) or ensure clock reads occur.

## Where are outputs written?
Current working directory of the traced process.

## Does CUTracer support CUDA Graph / Stream Capture?
Yes. Instrumentation and data flushing handle capture paths. For captured graphs, flushing occurs at `cuGraphLaunch` exit; ensure appropriate stream synchronization.

## Why does enabling deadlock detection increase overhead?
`deadlock_detection` auto-enables `reg_trace` to capture PCs/opcodes per warp; use `KERNEL_FILTERS` and `INSTR_BEGIN/INSTR_END` to reduce impact.


