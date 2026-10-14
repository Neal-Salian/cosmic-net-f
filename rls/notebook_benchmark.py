"""Shared exact-budget notebook comparisons; no implicit data/model fallbacks.

The learned control distils train-only frozen-predictor edge saliency. It is
not an oracle, attention model, PGExplainer, or a claim of superiority.
"""
import hashlib
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from data.provenance import build_run_provenance, hash_payload
from rls.benchmark_predictor import MeteredPredictor
from rls.benchmark_scorers import PairScorer
from rls.constrained_policy import select_pairs
from rls.constraints import PhysicalPairLayout, pair_budget, selection_diagnostics
from rls.matched_benchmark import BudgetSpec, evaluate_matched_budget
from rls.raw_selector import RawBudgetPairScorer, train_supervised_selector


def as_data(graph):
    if isinstance(graph, Data):
        return graph
    return Data(**{key: value for key, value in graph.items()
                   if key in ("x", "edge_index", "edge_attr", "y", "pos", "stellar_mass",
                              "cluster_id", "lineage_group")})


def fixed_marks(layout, seed, graph_id):
    value = int(hashlib.sha256(f"{seed}:{graph_id}".encode()).hexdigest()[:15], 16)
    return torch.randperm(len(layout.pairs), device=layout.pairs.device,
        generator=torch.Generator(device=layout.pairs.device).manual_seed(value))


def exact_policy_mask(policy, graph, cfg, q):
    """Use the same orientation-averaged scorer and exact decoder as benchmarks.

    Cached embeddings here are for training validation/TTA only; primary
    benchmark timing always uses the actual metered embedding forward.
    """
    data = as_data(graph)
    layout = PhysicalPairLayout.from_edge_index(data.edge_index, len(data.x))
    k = pair_budget(layout, q)
    names = cfg.get("edge_feature_names", ["distance", "delta_v", "cos_theta", "mass_ratio", "proj_sep"])
    cached = SimpleNamespace(call=lambda *args, **kwargs: SimpleNamespace(embeddings=graph["emb"]))
    scores = PairScorer("provided_rl", policy=policy, edge_feature_names=names).score(
        data, layout, k, meter=cached).scores
    marks = fixed_marks(layout, cfg.get("seed", 42), graph.get("cluster_id", "unidentified"))
    constraint = cfg.get("evaluation_constraint", "no_isolates")
    selection = select_pairs(layout, scores, k, constraint, method="direct", sample=False,
                             pair_marks=marks)
    mask = layout.expand(selection.selected)
    diag = selection_diagnostics(layout, selection.selected, k=k, constraint=constraint)
    diag.update(decoder_kind="exact_direct", requested_keep_fraction=float(q),
                requested_pair_count=k, final_pair_count=int(selection.selected.sum()),
                sampled_pair_count=None, budget_excess_pair_count=0)
    probabilities = scores.sigmoid()[layout.inverse]
    stored = torch.ones(data.num_edges, device=data.x.device)
    stored[layout.real] = probabilities
    return mask, stored, diag


def notebook_provenance(context):
    split = {"manifest_sha256": hash_payload(context["split_cluster_ids"])}
    return build_run_provenance(source_files=list(context["dataset"].values()),
        catalog_contract={"research_label": context["research_label"]},
        split_manifest=split, resolved_config=context["config_used"],
        source_revision=context["source_identity"], source_code_files=[],
        graph=context["config_used"]["graph"], augmentation_manifest=None,
        research_label=context["research_label"])


def fit_learned_control(model, split_graphs, context, fractions, *, epochs, hidden_dim=16):
    """Train a raw selector on train-only gradient×input labels with exact costs."""
    train = [as_data(g).to(next(model.parameters()).device) for g in split_graphs["train"]]
    train_ids = [g.cluster_id for g in train]
    heldout = [g.cluster_id for name in ("val", "test") for g in split_graphs[name]]
    split_hash = hash_payload(context["split_cluster_ids"])
    names = list(context["config_used"]["graph"]["edge_features"])
    meter = MeteredPredictor(model)
    teacher = PairScorer("edge_feature_grad_x_input")
    examples = []
    for graph in train:
        layout = PhysicalPairLayout.from_edge_index(graph.edge_index, len(graph.x))
        budgets = sorted({pair_budget(layout, q) for q in fractions})
        # One measured gradient forward per training graph, reused across budgets.
        labels = teacher.score(graph, layout, budgets[0], meter=meter).scores.detach()
        labels = labels / labels.abs().max().clamp_min(1e-8)
        for index, k in enumerate(budgets):
            examples.append(dict(graph={"x": graph.x, "edge_index": graph.edge_index,
                "edge_attr": graph.edge_attr, "cluster_id": graph.cluster_id},
                k=k, labels=labels, label_definition="train_only_normalized_edge_gradient_x_input",
                label_predictor_calls=1 if index == 0 else 0,
                label_provenance={"graph_id": graph.cluster_id, "shared_across_budgets": budgets,
                                  "predictor_state_hash": meter.model_state_hash()}))
    with torch.random.fork_rng(devices=[train[0].x.device.index or 0] if train[0].x.is_cuda else []):
        torch.manual_seed(context["seed"])
        scorer = RawBudgetPairScorer(train[0].x.shape[1], names, hidden_dim).to(train[0].x.device)
    scorer.fit_normalization([(ex["graph"], ex["k"]) for ex in examples], train_ids, heldout, split_hash)
    artifact = train_supervised_selector(scorer, examples, train_ids, heldout, split_hash,
        context["seed"], epochs, 1e-3)
    artifact["architecture"] = dict(node_dim=train[0].x.shape[1], edge_feature_names=names,
                                     hidden_dim=hidden_dim)
    artifact["predictor_state_hash"] = meter.model_state_hash()
    artifact["label_meter"] = meter.snapshot()
    return scorer, artifact


def load_learned_control(path, context, model):
    artifact = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    if artifact.get("predictor_state_hash") != MeteredPredictor(model).model_state_hash():
        raise ValueError("Learned control backbone differs from this run")
    scorer = RawBudgetPairScorer(**artifact["architecture"]).to(next(model.parameters()).device)
    scorer.load_state_dict(artifact["model_state"], strict=True)
    adapter = PairScorer("supervised_raw", raw_scorer=scorer, raw_artifact=artifact,
                         split_hash=hash_payload(context["split_cluster_ids"]))
    adapter.training_cost()  # Validate trained state/schema/split before evaluation.
    return scorer, artifact


def matched_notebook_rows(model, graphs, context, fractions, *, policy=None,
                          learned=None, learned_artifact=None):
    names = context["config_used"]["graph"]["edge_features"]
    scorers = [PairScorer("random"), PairScorer("distance"), PairScorer("degree"),
        PairScorer("stellar_binding_proxy", distance_epsilon=1e-6,
                   distance_unit="Mpc", stellar_mass_unit="M_sun"),
        PairScorer("edge_feature_grad_x_input")]
    if learned is not None:
        scorers.append(PairScorer("supervised_raw", raw_scorer=learned,
            raw_artifact=learned_artifact, split_hash=hash_payload(context["split_cluster_ids"])))
    if policy is not None:
        scorers.append(PairScorer("provided_rl", policy=policy, edge_feature_names=names))
    rows, per_example = [], []
    device = next(model.parameters()).device
    evaluation_graphs = [as_data(g).clone().to(device) for g in graphs]
    cfg = context["config_used"].get("rls", {})
    for q in sorted(set(fractions)):
        budget = BudgetSpec(keep_fraction=q, constraint=cfg.get("evaluation_constraint", "no_isolates"))
        summaries, raw = evaluate_matched_budget(model, evaluation_graphs, scorers,
            budget, seed=context["seed"], provenance=notebook_provenance(context), backbone="frozen")
        for row in summaries:
            own = [r for r in raw if r["method"] == row["method"] and r["status"] == "success"]
            keep = sum(r["stored_retention"] for r in own) / len(own) if own else None
            row.update(frac=q, keep_frac=keep, mean_keep_frac=keep,
                       backbone_sha256=row["model_state_hash"], run_mode=context["run_mode"],
                       method_population_rmse=row["rmse"], method_population_r2=row["r2"],
                       rmse=row["common_rmse"], r2=row["common_r2"], mae=row["common_mae"],
                       fidelity=row["common_fidelity_full_pred"],
                       mean_inference_region_time_s=(sum(r["total_region_time_s"] for r in own) / len(own) if own else None),
                       timing_scope="scoring_including_embedding_and_policy+decoder+prediction; graph construction separate")
        rows.extend(summaries)
        per_example.extend(raw)
    return rows, per_example
