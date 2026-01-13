from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List
import csv
import math

@dataclass
class FitResult:
    gpu: str
    alpha_ms: float
    beta_ms_per_token: float
    gamma_ms_per_batch: float
    rmse_ms: float
    n: int

def _solve_3x3(A: List[List[float]], b: List[float]) -> Tuple[float, float, float]:
    # Gaussian elimination (no external deps)
    M = [A[0] + [b[0]], A[1] + [b[1]], A[2] + [b[2]]]
    for i in range(3):
        # pivot
        piv = i
        for r in range(i+1, 3):
            if abs(M[r][i]) > abs(M[piv][i]):
                piv = r
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-12:
            raise ValueError("Singular system while fitting (need more diverse samples).")
        # normalize row
        div = M[i][i]
        for c in range(i, 4):
            M[i][c] /= div
        # eliminate
        for r in range(3):
            if r == i:
                continue
            factor = M[r][i]
            for c in range(i, 4):
                M[r][c] -= factor * M[i][c]
    return (M[0][3], M[1][3], M[2][3])

def fit_linear_latency(csv_path: str, gpu: str | None = None) -> List[FitResult]:
    """Fit alpha + beta*seq_len + gamma*batch from a microbench CSV.

    Expected CSV columns:
      - gpu (string)
      - seq_len (int)
      - batch (int)
      - latency_ms (float)
    Optional columns are ignored.
    """
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if gpu and r.get("gpu") != gpu:
                continue
            if not r.get("latency_ms"):
                continue
            rows.append(r)
    if not rows:
        raise ValueError("No valid rows found (check gpu name filter and latency_ms column).")

    # group by gpu
    by = {}
    for r in rows:
        by.setdefault(r["gpu"], []).append(r)

    out: List[FitResult] = []
    for g, rs in by.items():
        xs = []
        ys = []
        for r in rs:
            try:
                seq = float(r["seq_len"])
                bat = float(r["batch"])
                y = float(r["latency_ms"])
            except Exception:
                continue
            xs.append((1.0, seq, bat))
            ys.append(y)
        n = len(ys)
        if n < 6:
            raise ValueError(f"Need more samples for {g} (got {n}, need >= 6).")

        # normal equations: (X^T X) theta = X^T y
        # theta = [alpha, beta, gamma]
        s00 = sum(x[0]*x[0] for x in xs)
        s01 = sum(x[0]*x[1] for x in xs)
        s02 = sum(x[0]*x[2] for x in xs)
        s11 = sum(x[1]*x[1] for x in xs)
        s12 = sum(x[1]*x[2] for x in xs)
        s22 = sum(x[2]*x[2] for x in xs)

        t0 = sum(x[0]*y for x, y in zip(xs, ys))
        t1 = sum(x[1]*y for x, y in zip(xs, ys))
        t2 = sum(x[2]*y for x, y in zip(xs, ys))

        A = [
            [s00, s01, s02],
            [s01, s11, s12],
            [s02, s12, s22],
        ]
        b = [t0, t1, t2]
        alpha, beta, gamma = _solve_3x3(A, b)

        # rmse
        mse = 0.0
        for (c, seq, bat), y in zip(xs, ys):
            yhat = alpha + beta*seq + gamma*bat
            mse += (y - yhat) ** 2
        rmse = math.sqrt(mse / max(1, n))

        out.append(FitResult(gpu=g, alpha_ms=alpha, beta_ms_per_token=beta, gamma_ms_per_batch=gamma, rmse_ms=rmse, n=n))
    return out
