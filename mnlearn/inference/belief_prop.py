"""
Loopy max-sum belief propagation for MAP inference on general graphs.

Given a graph with nodes V, edges E, and K classes per node:

    y* = argmax_y  sum_v unary[v, y_v]  +  sum_{(u,v) in E} pairwise[y_u, y_v]

Belief propagation iteratively passes messages between neighbouring nodes.
On tree-structured graphs (e.g. chains) it is *exact* and agrees with Viterbi.
On graphs with cycles it is an approximation — but often very good in practice.

Implementation note: the iterative message-passing is vectorised over the
batch dimension, so the full ``[B, T, K]`` tensor is decoded in one pass
rather than ``B`` separate kernel launches. ``_bp_single`` is preserved as
a reference per-sample implementation, used by the test suite to confirm
that the batched version produces identical labelings.
"""

import torch

from mnlearn.models.graph import chain_edges


@torch.no_grad()
def _bp_single(unary: torch.Tensor, pairwise: torch.Tensor,
               edges: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Reference per-sample max-sum BP. Kept for test parity with ``_bp_batched``.

    Args:
        unary:    [T, K]    node potentials
        pairwise: [K, K]    pairwise[i, j] = score for (src=i, dst=j)
        edges:    [E, 2]    undirected edge list (each edge stored once)
        num_iters: int

    Returns:
        labels: [T] LongTensor
    """
    T, K = unary.shape
    E = edges.shape[0]
    device = unary.device

    src = edges[:, 0]   # [E]
    dst = edges[:, 1]   # [E]

    # msg[0, e, k] = message from src[e] -> dst[e] for class k
    # msg[1, e, k] = message from dst[e] -> src[e] for class k
    msg = torch.zeros(2, E, K, device=device, dtype=unary.dtype)

    for _ in range(num_iters):
        # Aggregate all incoming messages at each node
        node_msg = torch.zeros(T, K, device=device, dtype=unary.dtype)
        node_msg.scatter_add_(0, dst.unsqueeze(1).expand(E, K), msg[0])
        node_msg.scatter_add_(0, src.unsqueeze(1).expand(E, K), msg[1])

        # Forward messages: src[e] -> dst[e]
        # belief at src excluding the message it already sent to dst
        b_src = unary[src] + node_msg[src] - msg[1]            # [E, K]
        new_fwd = (b_src.unsqueeze(2) + pairwise.unsqueeze(0)).max(dim=1).values  # [E, K]
        new_fwd = new_fwd - new_fwd.max(dim=1, keepdim=True).values

        # Backward messages: dst[e] -> src[e]
        b_dst = unary[dst] + node_msg[dst] - msg[0]            # [E, K]
        new_bwd = (b_dst.unsqueeze(1) + pairwise.unsqueeze(0)).max(dim=2).values  # [E, K]
        new_bwd = new_bwd - new_bwd.max(dim=1, keepdim=True).values

        msg = torch.stack([new_fwd, new_bwd], dim=0)

    # Final beliefs
    node_msg = torch.zeros(T, K, device=device, dtype=unary.dtype)
    node_msg.scatter_add_(0, dst.unsqueeze(1).expand(E, K), msg[0])
    node_msg.scatter_add_(0, src.unsqueeze(1).expand(E, K), msg[1])

    belief = unary + node_msg                  # [T, K]
    return belief.argmax(dim=1)                # [T]


@torch.no_grad()
def _bp_batched(unary: torch.Tensor, pairwise: torch.Tensor,
                edges: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Batch-vectorised max-sum BP. Identical algorithm to ``_bp_single``
    but with all per-iteration tensors carrying an additional leading
    batch dimension, so the entire ``[B, T, K]`` batch is processed in
    a single sequence of GPU kernels.

    Args:
        unary:    [B, T, K]  node potentials
        pairwise: [K, K]     pairwise[i, j] = score for (src=i, dst=j)
        edges:    [E, 2]     undirected edge list (each edge stored once)
        num_iters: int

    Returns:
        labels: [B, T] LongTensor
    """
    B, T, K = unary.shape
    E = edges.shape[0]
    device = unary.device

    src = edges[:, 0]   # [E]
    dst = edges[:, 1]   # [E]

    # Pre-broadcast index tensors for scatter_add (shared across iters).
    dst_idx = dst.view(1, E, 1).expand(B, E, K)   # [B, E, K]
    src_idx = src.view(1, E, 1).expand(B, E, K)   # [B, E, K]
    pair_b  = pairwise.view(1, 1, K, K)           # broadcast over B and E

    # msg[0, b, e, k] = src[e] -> dst[e] for sample b, class k
    # msg[1, b, e, k] = dst[e] -> src[e] for sample b, class k
    msg = torch.zeros(2, B, E, K, device=device, dtype=unary.dtype)

    for _ in range(num_iters):
        # Aggregate all incoming messages at each node.
        node_msg = torch.zeros(B, T, K, device=device, dtype=unary.dtype)
        node_msg.scatter_add_(1, dst_idx, msg[0])
        node_msg.scatter_add_(1, src_idx, msg[1])

        # Forward messages: src[e] -> dst[e].
        # b_src[b, e, k] = unary[b, src[e], k] + node_msg[b, src[e], k] - msg[1, b, e, k]
        b_src   = unary[:, src, :] + node_msg[:, src, :] - msg[1]            # [B, E, K_src]
        new_fwd = (b_src.unsqueeze(-1) + pair_b).max(dim=2).values            # [B, E, K_dst]
        new_fwd = new_fwd - new_fwd.max(dim=-1, keepdim=True).values

        # Backward messages: dst[e] -> src[e].
        b_dst   = unary[:, dst, :] + node_msg[:, dst, :] - msg[0]            # [B, E, K_dst]
        new_bwd = (b_dst.unsqueeze(-2) + pair_b).max(dim=3).values            # [B, E, K_src]
        new_bwd = new_bwd - new_bwd.max(dim=-1, keepdim=True).values

        msg = torch.stack([new_fwd, new_bwd], dim=0)

    # Final beliefs.
    node_msg = torch.zeros(B, T, K, device=device, dtype=unary.dtype)
    node_msg.scatter_add_(1, dst_idx, msg[0])
    node_msg.scatter_add_(1, src_idx, msg[1])

    belief = unary + node_msg                  # [B, T, K]
    return belief.argmax(dim=-1)                # [B, T]


def bp_decode(unary: torch.Tensor, pairwise: torch.Tensor,
              edges=None, num_iters: int = 50) -> torch.Tensor:
    """Batch MAP decoding via max-sum belief propagation.

    Args:
        unary:     [batch, T, K]  node potentials
        pairwise:  [K, K]         pairwise potentials
        edges:     [E, 2] LongTensor or None (chain if None)
        num_iters: number of BP iterations

    Returns:
        y: [batch, T] LongTensor — MAP labeling per sample
    """
    _, T, _ = unary.shape
    device = unary.device

    if edges is None:
        edges = chain_edges(T, device)

    return _bp_batched(unary.detach(), pairwise.detach(), edges, num_iters)
