"""Auditable AML transaction-risk modelling and investigation utilities."""

from aml_evidence_graph.data.splits import TimeSplit, assign_time_split

__version__ = "1.0.0"

__all__ = ["TimeSplit", "__version__", "assign_time_split"]
