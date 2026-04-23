
- **Kernel**: A GPU function executed in parallel by many threads.
- **Warp**: A group of 32 threads that execute in lockstep.
- **SASS / Opcode**: NVIDIA GPU assembly instruction and its mnemonic.
- **Clock instruction**: SASS `CS2R SR_CLOCK*` reads the GPU clock. CUTracer uses these as region markers.
- **Region (clock-to-clock)**: A segment of dynamic instruction execution delimited by consecutive clock reads.
- **Histogram (instruction)**: Count of instruction mnemonics within a region per warp.
- **Instrumentation**: Injecting lightweight device calls into SASS via NVBit to emit data.
- **Analysis**: Host-side processing of emitted data into logs/CSVs.
- **Kernel filter**: Substring match on kernel names (mangled/unmangled) to limit instrumentation.

Limitations to note:
- CUTracer's instruction histogram analysis does not support nested regions; it treats consecutive clock reads as a flat sequence of start/stop markers.
- Graph/stream-capture launches are supported but have distinct synchronization paths.


