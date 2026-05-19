"""Quick wall-time benchmark: vectorised vs per-sample BP on Sudoku.
"""

import time

import torch

from mnlearn.inference.belief_prop import _bp_batched, _bp_single
from mnlearn.models import sudoku_edges


def _per_sample_decode(unary, pairwise, edges, num_iters):
    return torch.stack(
        [_bp_single(unary[b], pairwise, edges, num_iters)
         for b in range(unary.shape[0])],
        dim=0,
    )


def bench(B, T, K, num_iters, edges, label):
    torch.manual_seed(0)
    unary    = torch.randn(B, T, K)
    pairwise = torch.randn(K, K) * 0.3

    # Warmup
    _bp_batched(unary[:1], pairwise, edges, 1)
    _bp_single(unary[0], pairwise, edges, 1)

    t0 = time.perf_counter()
    y_batched = _bp_batched(unary, pairwise, edges, num_iters)
    t_batched = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_single = _per_sample_decode(unary, pairwise, edges, num_iters)
    t_single = time.perf_counter() - t0

    mismatch = (y_batched != y_single).float().mean().item()
    print(f"{label:>22}  B={B:>3} iter={num_iters:>3}  "
          f"per-sample={t_single*1000:>7.1f}ms  "
          f"batched={t_batched*1000:>7.1f}ms  "
          f"speedup={t_single/t_batched:>5.1f}x  "
          f"node mismatch={mismatch*100:.2f}%")


def main():
    sudoku = sudoku_edges()
    print("Sudoku (T=81, K=9, E=810):")
    for B in (1, 8, 32, 64):
        for num_iters in (15, 30):
            bench(B, T=81, K=9, num_iters=num_iters, edges=sudoku,
                  label="sudoku")


if __name__ == "__main__":
    main()
