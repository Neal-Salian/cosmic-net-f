"""Supervised projected baseline on explicitly selected content-audited catalogs.

This path uses measured observables and MSE only. It never expands the audited
partial records into the legacy intrinsic-feature/virial-loss training schema.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.astronomy_audit import audit_astronomy_manifest
from data.astronomy_loader import load_audited_catalog
from data.astronomy_manifest import load_astronomy_manifest
from data.astronomy_observations import observe_astronomy_halo
from data.catalog import CatalogContract
from data.observations import ObservationConfig
from data.projected_graph import ProjectedGraphConfig, build_projected_graph, checkpoint_compatibility_record
from data.provenance import (build_run_provenance, hash_payload, model_state_hash,
                             sha256_file, validate_inference_compatibility)
from data.splits import grouped_split
from model.model import build_model


SOURCE_FILES = (
    "training/projected_baseline.py", "training/full_graph_baseline.py",
    "data/astronomy_manifest.py", "data/astronomy_audit.py", "data/astronomy_loader.py",
    "data/astronomy_observations.py", "data/observations.py", "data/projected_graph.py",
    "data/catalog.py", "data/splits.py", "data/provenance.py",
    "graph/graph_builder.py", "model/model.py",
)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=0., inclusive=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (not inclusive and value == minimum)):
        raise ValueError(f"{name} must be finite and {'>=' if inclusive else '>'} {minimum}")
    return float(value)


def _resolve_config(config, allow_physics_proxy):
    config = deepcopy(config)
    # Reject NaN, paths/objects, and ambiguous YAML values before any output exists.
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline config must contain finite JSON-compatible values") from exc
    for name in ("data", "training", "model", "graph", "observation"):
        if not isinstance(config.get(name), dict) or not config[name]:
            raise ValueError(f"audited baseline requires explicit {name} configuration")
    data, training, model = config["data"], config["training"], config["model"]
    if data.get("source") != "audited_astronomy" or data.get("strict_research") is not True:
        raise ValueError("audited baseline requires data.source=audited_astronomy and strict_research=true")
    if not isinstance(data.get("manifest_path"), str) or not data["manifest_path"].strip():
        raise ValueError("data.manifest_path must explicitly identify an astronomy manifest")
    ids = data.get("catalog_ids")
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("data.catalog_ids must be a nonempty unique list of explicit catalog IDs")
    if data.get("split_group_key") not in {"initial_condition_id", "lineage_group", "volume_or_ic_group"}:
        raise ValueError("data.split_group_key must identify initial-condition/lineage groups")
    if training.get("evidence_mode") not in {"scientific", "fixture_smoke"}:
        raise ValueError("training.evidence_mode must explicitly be scientific or fixture_smoke")
    if config.get("augmentation", {}).get("enabled", False):
        raise ValueError("projected baseline does not support legacy intrinsic augmentation")
    if allow_physics_proxy or config.get("physics", {}).get("use_virial_loss", False):
        raise ValueError("projected partial schemas do not support virial physics loss/proxy")
    config["physics"] = {**config.get("physics", {}), "use_virial_loss": False}
    if config.get("wandb", {}).get("enabled", False):
        raise ValueError("audited baseline writes local artifacts; wandb.enabled must be false")
    config["wandb"] = {**config.get("wandb", {}), "enabled": False}
    seed = _integer(config.get("seed", 42), "seed", 0)
    if seed >= 2**32:
        raise ValueError("seed must be less than 2**32")
    config["seed"] = seed
    training["epochs"] = _integer(training.get("epochs"), "training.epochs")
    training["early_stopping_patience"] = _integer(training.get("early_stopping_patience", 20), "early_stopping_patience", 0)
    training["learning_rate"] = _number(training.get("learning_rate", .001), "learning_rate", inclusive=False)
    training["weight_decay"] = _number(training.get("weight_decay", .0001), "weight_decay")
    training["grad_clip"] = _number(training.get("grad_clip", 1.), "grad_clip")
    training.setdefault("optimizer", "adamw")
    training.setdefault("lr_scheduler", "none")
    if training["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("projected baseline optimizer must be adam or adamw")
    if training["lr_scheduler"] not in {"none", "cosine"}:
        raise ValueError("projected baseline lr_scheduler must be none or cosine")
    settings = training.setdefault("lr_scheduler_settings", {})
    if not isinstance(settings, dict) or set(settings) - {"T_max", "eta_min"}:
        raise ValueError("lr_scheduler_settings supports only T_max and eta_min")
    settings["T_max"] = _integer(settings.get("T_max", training["epochs"]), "T_max")
    settings["eta_min"] = _number(settings.get("eta_min", 0.), "eta_min")
    device_name = training.get("device", "cpu")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("training.device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("training.device=cuda requested but CUDA is unavailable")
    training["device"] = device_name
    data["batch_size"] = _integer(data.get("batch_size", 16), "data.batch_size")
    data["num_workers"] = _integer(data.get("num_workers", 0), "data.num_workers", 0)
    model.setdefault("node_features", 4)
    if model["node_features"] != 4:
        raise ValueError("projected node schema requires exactly four node features")
    for name, default, lower in (("hidden_dim", 64, 2), ("output_dim", 64, 1), ("num_layers", 3, 1)):
        model[name] = _integer(model.get(name, default), f"model.{name}", lower)
    model["dropout"] = _number(model.get("dropout", .05), "model.dropout")
    if model["dropout"] >= 1:
        raise ValueError("model.dropout must be less than 1")
    model.setdefault("pooling", "mean")
    if model["pooling"] not in {"mean", "set2set"}:
        raise ValueError("model.pooling must be mean or set2set")
    model.setdefault("activation", "leaky_relu")
    if model["activation"] not in {"relu", "leaky_relu", "elu", "gelu", "tanh", "silu"}:
        raise ValueError("unknown model.activation")
    model.setdefault("residual", True)
    model.setdefault("mc_dropout", False)
    for name in ("residual", "mc_dropout"):
        if type(model[name]) is not bool:
            raise ValueError(f"model.{name} must be boolean")
    graph_values = deepcopy(config["graph"])
    for name in ("node_features", "edge_features"):
        if name in graph_values:
            graph_values[name] = tuple(graph_values[name])
    if "self_loops" in graph_values and type(graph_values["self_loops"]) is not bool:
        raise ValueError("projected self_loops must be boolean")
    try:
        graph = ProjectedGraphConfig(**graph_values)
        observation = ObservationConfig(**config["observation"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid projected graph/observation configuration: {exc}") from exc
    config["graph"] = asdict(graph)
    config["graph"]["node_features"] = list(graph.node_features)
    config["graph"]["edge_features"] = list(graph.edge_features)
    model["edge_features"] = len(graph.edge_features)
    config["observation"] = asdict(observation)
    config["projected_compatibility"] = checkpoint_compatibility_record(graph, observation.line_of_sight)
    config["training_observation_config"] = asdict(observation)
    return config, graph, observation


def _prepare(config, graph_config, observation_config):
    data = config["data"]
    manifest = load_astronomy_manifest(data["manifest_path"])
    lookup = {entry["id"]: entry for entry in manifest.catalogs}
    if set(data["catalog_ids"]) - set(lookup):
        raise ValueError("data.catalog_ids contains a catalog absent from the manifest")
    selected = tuple(lookup[catalog_id] for catalog_id in data["catalog_ids"])
    fixture_mode = config["training"]["evidence_mode"] == "fixture_smoke"
    for entry in selected:
        if entry["role"] != "development":
            raise ValueError("baseline training catalog IDs must all have role=development")
        if entry["source"]["kind"] != "local_hdf5" and not fixture_mode:
            raise ValueError("scientific training rejects synthetic_fixture catalogs")
    # Inspect only the explicitly selected entries; the identity still binds the
    # complete manifest file. The strict loader independently revalidates it.
    audit = audit_astronomy_manifest(replace(manifest, catalogs=selected), inspect_hdf5=True)
    if not audit.content_verified:
        raise ValueError(f"catalog content audit failed: {audit.errors}")
    halos, reports, rejected, contracts = [], [], [], {}
    for catalog_id in data["catalog_ids"]:
        loaded = load_audited_catalog(manifest, audit, catalog_id, allow_synthetic_fixture=fixture_mode)
        if not loaded:
            raise ValueError(f"catalog {catalog_id!r} contains no halos")
        contracts[catalog_id] = loaded[0].metadata["catalog_contract"]
        for halo in loaded:
            observed, report = observe_astronomy_halo(halo, observation_config)
            reports.append(report)
            if observed is None:
                rejected.append({"halo_id": report["halo_id"], "reason": report.get("rejection_reason", "richness_selection")})
                continue
            observed.metadata[data["split_group_key"]] = observed.metadata["initial_condition_id"]
            halos.append(observed)
    ratios = tuple(data.get(f"{name}_ratio", default) for name, default in zip(("train", "val", "test"), (.7, .15, .15)))
    train, val, test, split = grouped_split(halos, group_key=data["split_group_key"], ratios=ratios, seed=config["seed"])
    if not train or not val or not test:
        group_count = len({h.metadata["initial_condition_id"] for h in halos})
        raise ValueError(f"baseline requires enough independent IC groups for nonempty train/val/test splits; got {group_count} groups and ratios {ratios}")
    graphs = {}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        graphs[name] = []
        for halo in rows:
            graph = build_projected_graph(halo, graph_config)
            # Keep rich provenance outside batches: PyG collates mapping values
            # recursively, which is unsuitable for variable-length audit reports.
            graphs[name].append(Data(**{key: graph[key] for key in (
                "x", "edge_index", "edge_attr", "y", "pos", "vel", "cluster_id", "num_nodes")}))
    contracts_list = list(contracts.values())
    combined = deepcopy(contracts_list[0])
    for contract in contracts_list[1:]:
        validate_inference_compatibility(combined, contract)
    combined.update(source="audited_manifest_selection", simulation="multiple_audited_catalogs",
                    volume_or_ic_group=sorted({h.metadata["initial_condition_id"] for h in halos}),
                    source_files=[record for contract in contracts_list for record in contract["source_files"]],
                    synthetic=any(contract["synthetic"] for contract in contracts_list))
    combined["research_label"] = "synthetic" if combined["synthetic"] else "corrected_research"
    CatalogContract.from_mapping(combined).validate(strict_research=not fixture_mode)
    return graphs, split, combined, contracts, reports, rejected, audit.to_dict(), manifest


@torch.no_grad()
def _evaluate(model, loader, device, split_name):
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(device)
        predictions, _ = model(batch)
        for cluster_id, predicted, target in zip(batch.cluster_id, predictions.cpu().tolist(), batch.y.cpu().tolist()):
            if not math.isfinite(predicted):
                raise ValueError("baseline produced nonfinite predictions")
            rows.append({"cluster_id": cluster_id, "split": split_name,
                         "prediction_log10_msun": predicted, "target_log10_msun": target})
    residual = np.asarray([row["prediction_log10_msun"] - row["target_log10_msun"] for row in rows])
    targets = np.asarray([row["target_log10_msun"] for row in rows])
    ss_total = float(np.sum((targets - targets.mean())**2))
    metrics = {f"{split_name}/mse": float(np.mean(residual**2)),
               f"{split_name}/rmse": float(np.sqrt(np.mean(residual**2))),
               f"{split_name}/mae": float(np.mean(np.abs(residual))),
               f"{split_name}/scatter": float(np.std(residual)),
               f"{split_name}/r2": float(1 - np.sum(residual**2) / ss_total) if ss_total > 0 else None}
    return metrics, rows


def run_projected_baseline(config, *, output_dir, allow_physics_proxy=False):
    """Validate/audit before creating output; train MSE and export best-val state."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Baseline requires a new output directory: {output}")
    config, graph_config, observation_config = _resolve_config(config, allow_physics_proxy)
    graphs, split, contract, contracts, reports, rejected, audit, manifest = _prepare(config, graph_config, observation_config)
    seed, training = config["seed"], config["training"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    device = torch.device(training["device"])
    model = build_model(config).to(device)
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[training["optimizer"]]
    optimizer = optimizer_cls(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **training["lr_scheduler_settings"])
                 if training["lr_scheduler"] == "cosine" else None)
    loaders = {name: DataLoader(rows, batch_size=config["data"]["batch_size"], shuffle=name == "train",
        num_workers=config["data"]["num_workers"], generator=torch.Generator().manual_seed(seed))
        for name, rows in graphs.items()}
    # Execute one dry inference to catch architecture/batch incompatibilities
    # before creating files. It does not update or select weights.
    _evaluate(model, loaders["val"], device, "val")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    label = "fixture_smoke" if training["evidence_mode"] == "fixture_smoke" else "corrected_research"
    provenance = build_run_provenance(source_files=contract["source_files"], catalog_contract=contract,
        split_manifest=split, resolved_config=config, source_revision=revision,
        source_code_files=[sha256_file(root / name) for name in SOURCE_FILES],
        graph=graph_config.to_dict(), augmentation_manifest=None, research_label=label)
    provenance.update(training_catalogs=deepcopy(contracts), astronomy_manifest=sha256_file(manifest.path),
        catalog_audit=audit, rng_roles={"model_init_seed": seed, "data_loader_seed": seed,
        "numpy_seed": seed, "torch_seed": seed, "observation_seed": observation_config.seed})
    output.mkdir(parents=True)
    best_loss, best_state, best_epoch, stale, history = math.inf, None, None, 0, []
    for epoch in range(training["epochs"]):
        model.train()
        squared_error, count = 0., 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            prediction, _ = model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y)
            if not torch.isfinite(loss):
                raise ValueError("training loss became nonfinite")
            loss.backward()
            if training["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            squared_error += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
        metrics, _ = _evaluate(model, loaders["val"], device, "val")
        current = metrics["val/mse"]
        history.append({"epoch": epoch, "train/mse": squared_error / count, **metrics})
        if current < best_loss:
            best_loss, best_epoch, stale = current, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step()
        if training["early_stopping_patience"] and stale >= training["early_stopping_patience"]:
            break
    model.load_state_dict(best_state, strict=True)
    val_metrics, val_rows = _evaluate(model, loaders["val"], device, "val")
    test_metrics, test_rows = _evaluate(model, loaders["test"], device, "test")
    provenance["model_state_sha256"] = model_state_hash(best_state)
    artifact = {"artifact_type": "projected_backbone", "model_state_dict": best_state,
        "provenance": provenance, "projected_compatibility": config["projected_compatibility"],
        "training_observation_config": config["training_observation_config"],
        "training_observation_config_sha256": hash_payload(config["training_observation_config"]),
        "best_epoch": best_epoch, "best_validation_mse": best_loss}
    checkpoint = output / "projected_backbone.pt"
    torch.save(artifact, checkpoint)
    checkpoint_record = sha256_file(checkpoint)
    result = {"research_label": label, "scientific_result": False,
        "evidence_mode": training["evidence_mode"], "physics_use_virial_loss": False,
        "metrics": {**val_metrics, **test_metrics}, "best_epoch": best_epoch,
        "output_dir": str(output), "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_record["sha256"], "policy_exported": False}
    payloads = {"results.json": result, "resolved_config.json": config,
        "split_manifest.json": split, "training_history.json": history,
        "heldout_predictions.json": {"val": val_rows, "test": test_rows},
        "observation_reports.json": reports, "rejected_halos.json": rejected,
        "provenance.json": {**provenance, "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_record["sha256"]},
        "projected_backbone.pt.manifest.json": checkpoint_record}
    for filename, payload in payloads.items():
        (output / filename).write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return result
