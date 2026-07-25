"""Pure-PyTorch CSC neighbor sampling used when pyg-lib/torch-sparse are absent."""

from __future__ import annotations

import torch
from torch import Tensor


def pure_torch_neighbor_sample(
    colptr: Tensor,
    row: Tensor,
    seed: Tensor,
    num_neighbors: list[int],
    replace: bool = False,
    directed: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Match torch_sparse.neighbor_sample outputs: node, row, col, edge.

    Graph is CSC: neighbors of node ``v`` live in ``row[colptr[v]:colptr[v+1]]``.
    """
    del directed  # induced vs directed: we always emit sampled tree edges
    if seed.numel() == 0:
        empty = seed.new_empty(0, dtype=torch.long)
        return empty, empty, empty, empty

    seed = seed.to(torch.long).contiguous()
    node_list: list[Tensor] = [seed]
    edge_rows: list[Tensor] = []
    edge_cols: list[Tensor] = []
    edge_ids: list[Tensor] = []

    # Local id map built incrementally; seeds occupy [0, len(seed)).
    global_to_local: dict[int, int] = {int(n): i for i, n in enumerate(seed.tolist())}
    nodes_ordered: list[int] = seed.tolist()
    frontier = seed

    for hop_count in num_neighbors:
        if hop_count == 0 or frontier.numel() == 0:
            node_list.append(frontier.new_empty(0))
            continue
        next_frontier_globals: list[int] = []
        hop_rows: list[int] = []
        hop_cols: list[int] = []
        hop_edges: list[int] = []

        for local_src, global_src in enumerate(frontier.tolist()):
            start = int(colptr[global_src].item())
            end = int(colptr[global_src + 1].item())
            degree = end - start
            if degree <= 0:
                continue
            if hop_count < 0 or (not replace and hop_count >= degree):
                chosen = torch.arange(start, end, dtype=torch.long)
            elif replace:
                offsets = torch.randint(0, degree, (hop_count,), dtype=torch.long)
                chosen = start + offsets
            else:
                offsets = torch.randperm(degree)[:hop_count]
                chosen = start + offsets

            neighbors = row[chosen].to(torch.long)
            for edge_pos, neighbor in zip(chosen.tolist(), neighbors.tolist()):
                if neighbor not in global_to_local:
                    global_to_local[neighbor] = len(nodes_ordered)
                    nodes_ordered.append(neighbor)
                    next_frontier_globals.append(neighbor)
                hop_rows.append(global_to_local[neighbor])
                hop_cols.append(global_to_local[int(global_src)])
                hop_edges.append(edge_pos)

        if hop_rows:
            edge_rows.append(torch.tensor(hop_rows, dtype=torch.long))
            edge_cols.append(torch.tensor(hop_cols, dtype=torch.long))
            edge_ids.append(torch.tensor(hop_edges, dtype=torch.long))
        else:
            empty = frontier.new_empty(0, dtype=torch.long)
            edge_rows.append(empty)
            edge_cols.append(empty)
            edge_ids.append(empty)

        if next_frontier_globals:
            frontier = torch.tensor(next_frontier_globals, dtype=torch.long)
        else:
            frontier = frontier.new_empty(0, dtype=torch.long)
        node_list.append(frontier)

    node = torch.tensor(nodes_ordered, dtype=torch.long)
    if edge_rows:
        row_out = torch.cat(edge_rows)
        col_out = torch.cat(edge_cols)
        edge_out = torch.cat(edge_ids)
    else:
        empty = node.new_empty(0)
        row_out = col_out = edge_out = empty
    return node, row_out, col_out, edge_out
