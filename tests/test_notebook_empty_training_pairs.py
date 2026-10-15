"""Learned notebook controls cannot label graphs with no physical pairs."""
import copy

import pytest
import torch
from torch_geometric.data import Data

from data.provenance import hash_payload
from rls.notebook_benchmark import fit_learned_control, load_learned_control


class EdgeSumPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.scored_graph_ids = []

    def forward(self, batch):
        self.scored_graph_ids.extend(batch.cluster_id)
        return batch.edge_attr.sum() * self.weight


def graph(graph_id, *, loop_only=False):
    edges = ([[0, 1], [0, 1]] if loop_only
             else [[0, 1, 0, 1], [1, 0, 0, 1]])
    return Data(x=torch.tensor([[1.], [2.]]), edge_index=torch.tensor(edges),
                edge_attr=torch.ones(len(edges[0]), 1), cluster_id=graph_id)


def run_inputs(train):
    splits = {"train": train, "val": [graph("val")], "test": [graph("test")]}
    context = {"seed": 7, "config_used": {"graph": {"edge_features": ["distance"]}},
               "split_cluster_ids": {name: [g.cluster_id for g in graphs]
                                     for name, graphs in splits.items()}}
    return splits, context


def test_loop_only_training_graph_is_excluded_and_recorded_in_saved_control(tmp_path):
    splits, context = run_inputs([graph("train-a"), graph("train-loop", loop_only=True),
                                  graph("train-b")])
    original_split = copy.deepcopy(context["split_cluster_ids"])
    model = EdgeSumPredictor()

    _, artifact = fit_learned_control(model, splits, context, [.4, 1.], epochs=1)

    assert model.scored_graph_ids == ["train-a", "train-b"]
    assert artifact["normalization_metadata"]["train_ids"] == ["train-a", "train-b"]
    assert artifact["label_provenance"]["total_calls"] == 2
    assert artifact["label_meter"]["scoring"]["forward_calls"] == 2
    assert artifact["training_cohort"] == {
        "population": "train_graphs_with_nonself_physical_pairs",
        "requested_train_ids": ["train-a", "train-loop", "train-b"],
        "train_ids": ["train-a", "train-b"],
        "heldout_ids": ["val", "test"],
        "excluded_graphs": [{"graph_id": "train-loop", "reason": "no_nonself_physical_pairs",
                             "available_pair_count": 0}],
        "split_hash": hash_payload(original_split),
    }
    assert artifact["split_hash"] == hash_payload(original_split)
    assert context["split_cluster_ids"] == original_split

    path = tmp_path / "learned_control.pt"
    torch.save(artifact, path)
    _, restored = load_learned_control(path, context, model)
    assert restored["training_cohort"] == artifact["training_cohort"]


def test_all_loop_only_training_graphs_fail_before_teacher_calls():
    splits, context = run_inputs([graph("loop-a", loop_only=True),
                                  graph("loop-b", loop_only=True)])
    model = EdgeSumPredictor()

    with pytest.raises(ValueError, match="no training graphs with non-self physical pairs"):
        fit_learned_control(model, splits, context, [.4, 1.], epochs=1)

    assert model.scored_graph_ids == []


def test_excluded_loop_only_graph_cannot_hide_train_heldout_overlap():
    splits, context = run_inputs([graph("train-a"), graph("val", loop_only=True)])
    model = EdgeSumPredictor()

    with pytest.raises(ValueError, match="overlap"):
        fit_learned_control(model, splits, context, [.4], epochs=1)

    assert model.scored_graph_ids == []
