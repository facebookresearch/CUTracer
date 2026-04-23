## High-level 🧱
1) NVBit hooks CUDA launches and enumerates SASS instructions.
2) CUTracer decides whether to instrument a kernel based on filters and modes.
3) Device-side injected calls push typed messages to a GPU→CPU channel.
4) A host thread receives messages, dispatches analyses, and writes outputs.

## Key components 🔑
- NVBit callbacks: `nvbit_at_init`, `nvbit_at_ctx_init/term`, `nvbit_at_cuda_event`
- Instrumentation entry: `instrument_function_if_needed` (iterates SASS, inserts device calls)
- Device funcs: `instrument_opcode`, `instrument_reg_val`, `instrument_mem` (push packets)
- Channel: `ChannelDev` (device) / `ChannelHost` (host)
- Host receiver: `recv_thread_fun` (dispatch by message type, run analyses)
- Logging + filenames: kernel log open/close, per-kernel iteration index

## Data flow 🔄
- At launch enter: optionally instrument; open kernel log; set `kernel_launch_id`; enable instrumented code.
- During kernel execution: device pushes packets per instruction/wrap.
- After kernel: host flush + synchronize; receiver drains channel; analysis writes files.

## Graph / Stream capture paths 🕸️
- For stream capture, kernels are instrumented during capture; data flushing occurs at `cuGraphLaunch` exit with proper stream synchronization.
- Manual graph builds are instrumented at `cuGraphAddKernelNode`; the same flush at `cuGraphLaunch` applies.

## Clock detection ⏱️
- At instrumentation time, CUTracer scans SASS and records opcode IDs of `CS2R SR_CLOCKLO` for each function.
- At analysis time, those IDs mark region boundaries.


