import yscope_clp_core
from yscope_clp_core import ClpQuery

from cutracer.query.reader import TraceReaderBase

from typing import Generator

class TraceReaderCLP(TraceReaderBase):
    """
    Reader for CUTracer trace files.

    Supports CLP format.
    Provides efficient iteration over trace records.

    Example:
        >>> reader = TraceReaderCLP("trace.clp")
        >>> for record in reader.iter_records():
        ...     print(record["sass"])
    """
    def __init__(self, file):
        assert file.exists(), f"Non-exist clp archive file: {file.absolute()}"
        self._archive = file

    def _filter_expr_to_clp_query(self, filter_exprs: str) -> str:
        pass

    def iter_records(self, filter_exprs: str) -> Generator:
        clp_query: ClpQuery = self._filter_expr_to_clp_query(filter_exprs)
        with yscope_clp_core.search_archive(self._archive. clp_query) as iter:
            for next_record in iter:
                yield next_record

