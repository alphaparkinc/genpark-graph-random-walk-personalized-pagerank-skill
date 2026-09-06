"""
Graph Random Walk Personalized PageRank Skill Client
Pure Python Standard Library implementation of Personalized PageRank (PPR).
Calculates personalized stationary distribution vectors over heterogeneous memory graphs
for context-dependent entity ranking and associative knowledge retrieval.
"""

from typing import List, Dict, Any, Tuple, Set


class PersonalizedPageRank:
    """
    Personalized PageRank engine using power iteration.
    p = (1 - alpha) * W * p + alpha * v_teleport
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 40, tol: float = 1e-6):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.adjacency: Dict[str, Dict[str, float]] = {}
        self.nodes: Set[str] = set()

    def add_edge(self, u: str, v: str, weight: float = 1.0, bidirectional: bool = False):
        """Add weighted directed edge between memory nodes."""
        u = u.strip()
        v = v.strip()
        self.nodes.add(u)
        self.nodes.add(v)

        if u not in self.adjacency:
            self.adjacency[u] = {}
        self.adjacency[u][v] = self.adjacency[u].get(v, 0.0) + weight

        if bidirectional:
            if v not in self.adjacency:
                self.adjacency[v] = {}
            self.adjacency[v][u] = self.adjacency[v].get(u, 0.0) + weight

    def compute_ppr(self, seed_nodes: Dict[str, float]) -> Dict[str, float]:
        """
        Compute Personalized PageRank scores given teleport seed distribution.
        :param seed_nodes: Dict mapping seed node names to positive teleport weights.
        :return: Dict mapping each graph node to its stationary PPR probability.
        """
        if not self.nodes:
            return {}

        total_nodes = len(self.nodes)
        node_list = sorted(list(self.nodes))

        # Normalize seed teleport vector
        teleport_sum = sum(seed_nodes.get(n, 0.0) for n in node_list)
        if teleport_sum > 0:
            teleport = {n: seed_nodes.get(n, 0.0) / teleport_sum for n in node_list}
        else:
            # Fallback to uniform teleport if seeds not found
            teleport = {n: 1.0 / total_nodes for n in node_list}

        # Initialize uniform rank vector
        rank = {n: teleport[n] for n in node_list}

        # Compute out-degree weights
        out_weights = {}
        for u in node_list:
            neighbors = self.adjacency.get(u, {})
            out_weights[u] = sum(neighbors.values())

        # Power iteration
        alpha = 1.0 - self.damping

        for _ in range(self.max_iter):
            next_rank = {n: alpha * teleport[n] for n in node_list}
            dangling_mass = 0.0

            for u in node_list:
                w_total = out_weights[u]
                if w_total > 0:
                    for v, w in self.adjacency.get(u, {}).items():
                        next_rank[v] += self.damping * rank[u] * (w / w_total)
                else:
                    dangling_mass += rank[u]

            # Distribute dangling node probability mass according to teleport distribution
            if dangling_mass > 0:
                for n in node_list:
                    next_rank[n] += self.damping * dangling_mass * teleport[n]

            # Check convergence L1 norm
            diff = sum(abs(next_rank[n] - rank[n]) for n in node_list)
            rank = next_rank
            if diff < self.tol:
                break

        return rank

    def top_k_associates(self, seed_nodes: Dict[str, float], k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-K most relevant nodes relative to seed context."""
        ppr = self.compute_ppr(seed_nodes)
        sorted_nodes = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]
