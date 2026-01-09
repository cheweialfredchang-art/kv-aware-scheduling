from __future__ import annotations
GiB = 1024 ** 3
def gib(x: float) -> int:
    return int(x * GiB)
def bytes_to_gib(b: int) -> float:
    return b / GiB
