from __future__ import annotations

from typing import List, Optional, Sequence

DEFAULT_MAX_RAM_PERCENTAGE_ARG = "-XX:MaxRAMPercentage=80"
EXIT_ON_OUT_OF_MEMORY_ARG = "-XX:+ExitOnOutOfMemoryError"


def _as_list(java_args: Optional[Sequence[str]]) -> List[str]:
    return list(java_args) if java_args else []


def _has_explicit_heap_cap(java_args: Sequence[str]) -> bool:
    return any(arg.startswith("-Xmx") or arg.startswith("-XX:MaxRAMPercentage") for arg in java_args)


def normalize_pipeline_jvm_options(java_args: Optional[Sequence[str]]) -> List[str]:
    normalized = _as_list(java_args)
    if not _has_explicit_heap_cap(normalized):
        normalized.append(DEFAULT_MAX_RAM_PERCENTAGE_ARG)
    return normalized


def normalize_runtime_jvm_args(java_args: Optional[Sequence[str]]) -> List[str]:
    normalized = normalize_pipeline_jvm_options(java_args)
    if EXIT_ON_OUT_OF_MEMORY_ARG not in normalized:
        normalized.append(EXIT_ON_OUT_OF_MEMORY_ARG)
    return normalized
