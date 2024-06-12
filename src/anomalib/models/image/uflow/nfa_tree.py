"""Description:
    This module contains the implementation of the NFA tree algorithm, used in the U-Flow model to compute the
    log-probability map from the latent variables, and then generate a mask, performing an automatic segmentation of
    anomalies.
    We construct a tree of upper level sets from the latent variables, and compute the log-probability of each region in
    the image. We then perform a prune and merge process to remove nodes with high PFA values. The final tree is used to
    compute the log-NFA, which is used to obtain the segmentation mask with an automatic threshold.

Note:
    This code is part of the U-Flow model, and is not used in the basic functionality of training and testing. It is
    included as it is part of the U-Flow paper, and can be called separately if an unsupervised anomaly segmentation is
    needed.

Licence:
    Copyright (C) 2022-2024 Intel Corporation
    SPDX-License-Identifier: Apache-2.0
"""
import itertools as it
from typing import Union

import networkx as nx
import numpy as np
import torch
from mpmath import mp
from skimage.morphology import max_tree

mp.dps = 15


class NFATree:
    """Class for building a NFA tree of upper level sets, from a latent variable."""

    def __init__(self, zi: torch.Tensor):
        """Args:
        zi (torch.Tensor): Latent variable of shape (C, H, W).
        """
        self.n_channels = zi.shape[0]
        self.zi2_rav = zi.reshape(self.n_channels, -1).cpu().numpy() ** 2

        score = torch.mean(zi**2, dim=0).cpu().numpy()
        self.original_shape = score.shape
        self.tree = self.build_tree(score)

    def compute_log_prob_map(self) -> np.ndarray:
        """Compute the log probability map
        First compute the log probability of each node of the tree. Then, apply the prune and merge steps iteratively
        until no more changes are done in the tree. Finally, get the final clusters and build the log probability map.
        """
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
        """Compute the log probability of each node in the tree. The log probability is computed using the Chernoff bound
        for a Chi2 distribution of `self.n_channels` degrees of freedom.
        """
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

    def build_tree(self, score: np.ndarray) -> nx.DiGraph:
        """Build a tree from the score map."""
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

    def get_branch(self, starting_node: int) -> list[int]:
        """Get a connected section of the tree, starting from `starting_node`, where all nodes have exactly one predecessor
        (except for the starting leaf itself)
        """
        branch = [starting_node]
        successors = [s for s in self.tree.successors(starting_node)]

        if len(successors) == 0:
            return branch
        assert len(successors) == 1, "Node has more than one successor"

        is_only_child = len([p for p in self.tree.predecessors(successors[0])]) == 1
        if is_only_child:
            branch.extend(self.get_branch(successors[0]))
        return branch

    def get_final_clusters(self) -> dict[float, list[int]]:
        """Get the final clusters of the tree.
        The final clusters are the leaves of the final tree, where each leaf is the node with the lowest log probability
        in its branch.
        """
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
        """Procedure 1 in the paper (https://link.springer.com/article/10.1007/s10851-024-01193-y).
        This procedure aims to filter a set of nested connected components, keeping only the most significant one. We
        identify which of these connected components is the most significant one, as it may better delineate the
        anomalous region. After determining which node to preserve, the tree is pruned so that just the chosen node is
        kept, and all other branch nodes are removed.
        """
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
        """Procedure 2 in the paper (https://link.springer.com/article/10.1007/s10851-024-01193-y).
        The second procedure consists of merging leaf nodes with the same successor in case the latter is more
        significant than all others. In this case, all leaf nodes are removed from the tree, and we only keep their
        successor.
        """
        merged = False
        bifurcations = [p for p in self.tree.pred if len(self.tree.pred[p]) > 1]
        for b in bifurcations:
            # if predecessors are not leaves, continue. We only merge leaves.
            if np.sum([len([pp for pp in self.tree.predecessors(p)]) for p in self.tree.predecessors(b)]) > 0:
                continue
            preds = [p for p in self.tree.predecessors(b)]
            preds_nfas = [self.tree.nodes[p]["log_prob"] for p in preds]
            if self.tree.nodes[b]["log_prob"] <= np.min(preds_nfas):
                merged = True
                for p in preds:
                    self.tree.add_edges_from(
                        it.product(self.tree.predecessors(p), self.tree.successors(p)),
                    )
                    self.tree.remove_node(p)
        return merged


def compute_number_of_tests(polyominoes_sizes: Union[int, list[int]]) -> float:
    """Compute the number of tests for the NFA tree, corresponding to all possible regions with arbitrary shape and size in
    the image. Considering 4-connectivity, these groups of connected pixels correspond to the figures called polyominoes
    and a good approximation for the number of polyominoes is given by this formula. See references [60] and [61] in the
    U-Flow paper for more details.
    """
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
