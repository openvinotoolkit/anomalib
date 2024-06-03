import itertools as it

import networkx as nx
import numpy as np
import torch
from mpmath import mp
from skimage.morphology import max_tree

mp.dps = 15


class NFATree:
    def __init__(self, zi):
        self.n_channels = zi.shape[0]
        self.zi2_rav = zi.reshape(self.n_channels, -1).cpu().numpy() ** 2

        score = torch.mean(zi**2, dim=0).cpu().numpy()
        self.original_shape = score.shape
        self.tree = self.build_tree(score)

    def compute_log_prob_map(self):
        self.compute_log_prob()

        self.pfa_prune()
        keep_merging = self.pfa_merge()
        while keep_merging:
            self.pfa_prune()
            keep_merging = self.pfa_merge()
        self.pfa_prune()

        log_prob_map = np.empty(self.original_shape[0] * self.original_shape[1], dtype=np.float32)
        log_prob_map[:] = np.nan

        final_clusters = self.get_final_clusters()

        for log_prob, pixels in final_clusters.items():
            log_prob_map[pixels] = log_prob

        log_prob_map = log_prob_map.reshape(self.original_shape)

        return log_prob_map

    def compute_log_prob(self):
        zi2_sum = np.sum(self.zi2_rav, axis=0)

        for n in self.tree.nodes:
            region = self.tree.nodes[n]["pixels"]
            zi2_min = zi2_sum[region].min()

            # Chernoff bound for one Chi2 distribution of `self.n_channels` degrees of freedom
            log_prob = (
                -(self.n_channels / 2)
                * (zi2_min / self.n_channels - 1 - np.log(zi2_min / self.n_channels))
                / np.log(10)
            )

            # Log prob for the whole region
            self.tree.nodes[n]["log_prob"] = len(region) * log_prob

    def build_tree(self, score):
        parents, pixel_indices = max_tree(score, connectivity=1)
        parents_rav = parents.ravel()
        score_rav = score.ravel()

        tree = nx.DiGraph()
        tree.add_nodes_from(pixel_indices)
        for node in tree.nodes():
            tree.nodes[node]["score"] = score_rav[node]
        tree.add_edges_from([(n, parents_rav[n]) for n in pixel_indices[1:]])

        self.prune(tree, pixel_indices[0])
        self.accumulate(tree, pixel_indices[0])

        return tree

    def prune(self, graph, starting_node):
        """Transform a canonical max tree to a max tree."""
        value = graph.nodes[starting_node]["score"]
        cluster_nodes = [starting_node]
        for p in [p for p in graph.predecessors(starting_node)]:
            if graph.nodes[p]["score"] == value:
                cluster_nodes.append(p)
                graph.remove_node(p)
            else:
                self.prune(graph, p)
        graph.nodes[starting_node]["pixels"] = cluster_nodes

    def accumulate(self, graph, starting_node):
        """Transform a max tree to a component tree."""
        pixels = graph.nodes[starting_node]["pixels"]
        for p in graph.predecessors(starting_node):
            pixels.extend(self.accumulate(graph, p))
        return pixels

    def get_branch(self, starting_node):
        branch = [starting_node]
        successors = [s for s in self.tree.successors(starting_node)]

        if len(successors) == 0:
            return branch
        assert len(successors) == 1, "Node has more than one successor"

        is_only_child = len([p for p in self.tree.predecessors(successors[0])]) == 1
        if is_only_child:
            branch.extend(self.get_branch(successors[0]))
        return branch

    def get_final_clusters(self):
        leaves = [p for p in self.tree.pred if len(self.tree.pred[p]) == 0]
        final_clusters = {}
        for l in leaves:
            branch_nodes = self.get_branch(l)
            branch_log_probs = [self.tree.nodes[b]["log_prob"] for b in branch_nodes]
            branch_chosen_node = branch_nodes[np.argmin(branch_log_probs)]
            final_clusters[self.tree.nodes[branch_chosen_node]["log_prob"]] = self.tree.nodes[branch_chosen_node][
                "pixels"
            ]
        return final_clusters

    def pfa_prune(self):
        leaves = [p for p in self.tree.pred if len(self.tree.pred[p]) == 0]
        for l in leaves:
            branch_nodes = self.get_branch(l)
            branch_log_probs = [self.tree.nodes[b]["log_prob"] for b in branch_nodes]
            chosen_node = np.argmin(branch_log_probs)
            for i in range(len(branch_nodes)):
                if i != chosen_node:
                    self.tree.add_edges_from(
                        it.product(self.tree.predecessors(branch_nodes[i]), self.tree.successors(branch_nodes[i])),
                    )
                    self.tree.remove_node(branch_nodes[i])

    def pfa_merge(self):
        merged = False
        bifurcations = [p for p in self.tree.pred if len(self.tree.pred[p]) > 1]
        for b in bifurcations:
            # if predecessors are not leaves, continue. We only merge leaves.
            if np.sum([len([pp for pp in self.tree.predecessors(p)]) for p in self.tree.predecessors(b)]) > 0:
                continue
            preds = [p for p in self.tree.predecessors(b)]
            preds_nfas = [self.tree.nodes[p]["log_prob"] for p in preds]
            if self.tree.nodes[b]["log_prob"] <= np.min(preds_nfas):  # TODO: check
                merged = True
                for p in preds:
                    self.tree.add_edges_from(
                        it.product(self.tree.predecessors(p), self.tree.successors(p)),
                    )
                    self.tree.remove_node(p)
        return merged


def compute_number_of_tests(polyominoes_sizes):
    alpha = mp.mpf(0.316915)
    beta = mp.mpf(4.062570)

    if not isinstance(polyominoes_sizes, list):
        polyominoes_sizes = [polyominoes_sizes]

    n_test = mp.mpf(0)
    for region_size in polyominoes_sizes:
        n_test_i = mp.mpf(0)
        for r in range(1, region_size + 1):
            region_size_mp = mp.mpf(r)
            n_test_i += alpha * beta**region_size_mp / region_size_mp
        n_test += n_test_i * region_size

    return float(np.array(mp.log10(n_test), dtype=np.float32))
