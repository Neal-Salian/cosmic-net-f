"""Reusable supervised full-graph research baseline entry point."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from data.augmentation import augment_training_split
from data.catalog import CatalogContract, attach_catalog_contract, source_file_record
from data.loaders.base_loader import HaloData, SubhaloData, get_loader
from data.provenance import build_run_provenance
from data.splits import grouped_split
from graph.graph_builder import build_dataloaders
from model.model import build_model
from training.train import Trainer


SOURCE_FILES = (
    "graph/graph_builder.py",
    "data/catalog.py",
    "data/splits.py",
    "data/augmentation.py",
    "data/provenance.py",
    "data/loaders/base_loader.py",
    "data/loaders/tng_loader.py",
    "data/loaders/camels_loader.py",
    "data/loaders/synthetic_loader.py",
    "model/model.py",
    "training/train.py",
    "training/full_graph_baseline.py",
)


def _source_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _smoke_catalog(seed, count=12):
    rng = np.random.default_rng(seed)
    halos = []
    for halo_index in range(count):
        subhalos = []
        center = rng.normal(0, 2, size=3)
        halo_mass = 10 ** rng.uniform(11.5, 14.0)
        for subhalo_index in range(5):
            stellar_mass = 10 ** rng.uniform(8.5, 11.0)
            subhalos.append(SubhaloData(
                subhalo_id=subhalo_index,
                position=(center + rng.normal(0, .4, size=3)).astype(np.float32),
                velocity=rng.normal(0, 150, size=3).astype(np.float32),
                stellar_mass=float(stellar_mass),
                velocity_dispersion=float(rng.uniform(60, 250)),
                half_mass_radius=float(rng.uniform(.001, .02)),
                metallicity=float(rng.uniform(.005, .03)),
            ))
        halos.append(HaloData(
            cluster_id=f"synthetic-smoke-{halo_index:02d}", subhalos=subhalos,
            halo_mass=float(halo_mass),
            metadata={"lineage_group": f"synthetic-smoke-{halo_index:02d}"},
        ))
    contract = CatalogContract(
        research_label="synthetic", source="synthetic_smoke", suite=None,
        simulation=None, snapshot=None, volume_or_ic_group="seeded_fixture",
        target_field="synthetic_halo_mass",
        target_definition="log10(synthetic_halo_mass/M_sun)",
        mass_units="M_sun", position_units="synthetic_distance",
        radius_units="synthetic_distance", velocity_units="synthetic_velocity",
        coordinate_frame="synthetic_cartesian", velocity_convention="synthetic",
        hubble_param=None, scale_factor=None,
        radius_source_field="synthetic_half_mass_radius",
        radius_semantic="synthetic_half_mass_radius", synthetic=True,
        fallback=False, source_files=[],
        field_mapping={"halo_mass": "generated_halo_mass"},
        conversion_record={"cosmology": "not_applicable_synthetic"},
    )
    attach_catalog_contract(halos, contract)
    return halos, contract


def run_baseline(config, *, output_dir, smoke=False, allow_physics_proxy=False):
    """Train and evaluate an unpruned GNN with durable data/run provenance."""
    config = deepcopy(config)
    data_config = config.setdefault("data", {})
    if not smoke:
        if not data_config.get("strict_research") or not data_config.get(
                "catalog_contract"):
            raise ValueError(
                "corrected non-smoke baselines require strict_research=true "
                "and an explicit catalog_contract"
            )
        contract = CatalogContract.from_mapping(data_config["catalog_contract"])
        contract.validate(strict_research=True)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Baseline requires a new output directory: {output}")
    output.mkdir(parents=True)

    seed = int(config.get("seed", 42))
    if smoke:
        halos, contract = _smoke_catalog(seed)
        group_key = "lineage_group"
        data_config["source"] = "synthetic_smoke"
        config.setdefault("training", {})["train_source"] = "synthetic_smoke"
        config["training"]["test_source"] = "synthetic_smoke"
    else:
        halos = get_loader(config).load()
        contract = CatalogContract.from_mapping(
            halos[0].metadata["catalog_contract"])
        group_key = data_config.get("split_group_key")
        if not group_key:
            raise ValueError("corrected baseline requires data.split_group_key")

    ratios = ((.5, .25, .25) if smoke else (
        float(data_config.get("train_ratio", .7)),
        float(data_config.get("val_ratio", .15)),
        float(data_config.get("test_ratio", .15)),
    ))
    train, val, test, split_manifest = grouped_split(
        halos, group_key=group_key, ratios=ratios, seed=seed
    )
    if not train or not val or not test:
        raise ValueError("baseline requires nonempty grouped train/val/test splits")

    augmentation_config = config.get("augmentation", {})
    augmentation_manifest = None
    if augmentation_config.get("enabled", False):
        train, augmentation_manifest = augment_training_split(
            train, split_manifest,
            copies=int(augmentation_config.get("copies", 1)),
            seed=int(augmentation_config.get("seed", seed)),
            jitter_std=float(augmentation_config.get("jitter_std", 0.0)),
            rotate=bool(augmentation_config.get("rotate", True)),
            line_of_sight=augmentation_config.get("line_of_sight", "z"),
        )
    config.setdefault("augmentation", {})["enabled"] = False
    config.setdefault("physics", {})["use_virial_loss"] = bool(
        allow_physics_proxy and config.get("physics", {}).get("use_virial_loss", False)
    )
    config.setdefault("model", {})["edge_features"] = len(
        config.get("graph", {}).get("edge_features", [])
    )
    if smoke:
        config.setdefault("training", {})["epochs"] = 1
        config["training"]["save_every"] = 1
        config.setdefault("data", {})["num_workers"] = 0
    checkpoint_dir = output / "checkpoints"
    config.setdefault("training", {})["checkpoint_dir"] = str(checkpoint_dir)
    config.setdefault("wandb", {})["enabled"] = False

    source_code_files = [source_file_record(path) for path in SOURCE_FILES]
    provenance = build_run_provenance(
        source_files=contract.source_files,
        source_code_files=source_code_files,
        catalog_contract=contract.to_dict(), split_manifest=split_manifest,
        resolved_config=config, source_revision=_source_revision(),
        graph={"backend": "portable_torch", "tie_policy": "inclusive",
               "method": config.get("graph", {}).get("method")},
        augmentation_manifest=augmentation_manifest,
        research_label=contract.research_label,
    )
    if smoke:
        provenance["data_generator"] = {
            "name": "training.full_graph_baseline._smoke_catalog",
            "seed": seed,
            "halo_count": 12,
            "subhalos_per_halo": 5,
            "generator": "numpy.random.Generator(PCG64)",
        }
    provenance["rng_roles"] = {
        "torch_seed": seed,
        "numpy_seed": seed,
        "cuda_seed": seed,
        "data_loader_seed": seed,
        "model_init_seed": seed,
    }
    config["provenance"] = provenance

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader, test_loader = build_dataloaders(
        config, train, val, test, seed=seed
    )
    model = build_model(config)
    trainer = Trainer(
        config, model, train_loader, val_loader, test_loader,
        device=torch.device("cpu") if smoke else None,
    )
    trainer.train()
    metrics = {
        **trainer.validate(val_loader, prefix="val"),
        **trainer.validate(test_loader, prefix="test"),
    }
    metrics = {key: float(value) for key, value in metrics.items()}

    checkpoint_path = checkpoint_dir / "best_model.pt"
    checkpoint_manifest = json.loads(
        checkpoint_path.with_name(checkpoint_path.name + ".manifest.json").read_text()
    )
    final_provenance = {
        **provenance,
        "train_catalog": contract.to_dict(),
        "evaluation_catalog": contract.to_dict(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
        "model_state_sha256": checkpoint_manifest["model_state_sha256"],
    }
    result = {
        "research_label": contract.research_label,
        "physics_use_virial_loss": config["physics"]["use_virial_loss"],
        "metrics": metrics,
        "output_dir": str(output),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, sort_keys=True, indent=2)
    )
    (output / "provenance.json").write_text(
        json.dumps(final_provenance, sort_keys=True, indent=2, allow_nan=False)
    )
    (output / "results.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-physics-proxy", action="store_true")
    args = parser.parse_args(argv)
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    result = run_baseline(
        config, output_dir=args.output_dir, smoke=args.smoke,
        allow_physics_proxy=args.allow_physics_proxy,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
