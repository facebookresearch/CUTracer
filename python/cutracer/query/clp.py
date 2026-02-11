import yscope_clp_core

from cutracer.query.grouper import StreamingGrouperBase
from cutracer.query.reader import TraceReaderBase

class StreamingGrouperCLP(StreamingGrouperBase):
    """
    Stream-based grouper for trace records in CLP.

    Processes records in a single pass, maintaining bounded memory
    per group using deque for tail operations.

    Design principles:
    - Single-pass: records iterator is consumed only once
    - Bounded memory: uses deque(maxlen=N) for tail operations
    - Memory complexity: O(groups × N) for head/tail, O(groups) for count

    Example:
        >>> records = CLPArchive("archive_path.clp")
        >>> grouper = StreamingGrouperBase(records, "warp")
        >>> groups = grouper.tail_per_group(10)
        >>> for warp, records in groups.items():
        ...     print(f"Warp {warp}: {len(records)} records")
    """
    pass

class TraceReaderCLP(TraceReaderBase):
    """
    Reader for CUTracer trace files.

    Supports CLP format.
    Provides efficient iteration over trace records.

    Example:
        >>> reader = TraceReader("trace.clp")
        >>> for record in reader.iter_records():
        ...     print(record["sass"])
    """
    pass
