"""Execute A -> B -> C -> D, including real training and artifact handoffs."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.test_notebook_a_baselines import ROOT, execute, small_inputs
import rls.notebook_workflow as workflow


def notebook_sources(stage):
    path = next((ROOT / "notebooks").glob(f"notebook_{stage}_*.ipynb"))
    return {cell["id"]: "".join(cell["source"])
            for cell in json.loads(path.read_text())["cells"] if cell["cell_type"] == "code"}


def run_notebook(stage, tmp_path, **overrides):
    sources = notebook_sources(stage)
    namespace = {}
    execute(sources["settings"], namespace)
    namespace.update(REPO_DIR=ROOT, SYNC_REPO=False, INSTALL_MISSING=False,
        INPUT_ROOT=tmp_path / "input", DEVICE="cpu", GRAPH_BACKEND="torch",
        SMOKE_TEST=True, SMOKE_GRAPHS_PER_SPLIT=2, OUTPUT_DIR=tmp_path / stage,
        GUMBEL_EPOCHS=1, POLICY_EPOCHS=1, STAGEB_EPOCHS=1,
        TTA_KS=[0, 1, 2], COVERAGE_SAMPLES=2,
        A_ARTIFACT_DIR=tmp_path / "A/smoke", B_ARTIFACT_DIR=tmp_path / "B/smoke")
    namespace.update(overrides)
    for name, source in sources.items():
        if name != "settings":
            execute(source, namespace, f"Notebook {stage}: {name}")
    return namespace


@pytest.fixture(scope="module")
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_full_pipeline(tmp_path, monkeypatch, small_inputs, single_thread):
    monkeypatch.chdir(tmp_path)
    a = run_notebook("A", tmp_path)
    b = run_notebook("B", tmp_path)
    c = run_notebook("C", tmp_path, EVALUATE_STAGE_B=True)
    calls = []
    original = workflow.adapt_at_test_time

    def checked_adaptation(policy, graph, gnn, cfg, *args, **kwargs):
        assert "y" not in graph
        calls.append(cfg["tta_steps"])
        return original(policy, graph, gnn, cfg, *args, **kwargs)

    monkeypatch.setattr(workflow, "adapt_at_test_time", checked_adaptation)
    d = run_notebook("D", tmp_path)
    assert {1, 2}.issubset(calls)
    assert all(torch.equal(value, d["policy"].state_dict()[key])
               for key, value in d["offline_policy_state"].items())
    assert d["selected_k"] in d["tta_df"].K.values
    assert set(d["tta_df"].K) == set(d["gate_df"].loc[d["gate_df"].enabled, "K"])
    assert b["gnn"] is not b["stageb_model"]
    assert b["backbone_record"] == c["backbone_record"] == d["backbone_record"]
    gumbel = c["results_df"].query("method == 'gumbel'").iloc[0]
    assert gumbel.backbone_stage == "gumbel_joint"
    assert gumbel.backbone_sha256 == a["gumbel_record"]["sha256"]
    assert len(c["results_df"]) == len(a["baseline_df"]) + 1 + (2 if b["stageb_info"].get("accepted") else 0)
    assert set(c["coverage_df"].method) == {"full", "rl_policy"}
    # Exercise optional OOD cells with a tiny catalog-format fixture. These
    # are smoke-test artifacts only, never scientific CAMELS results.
    import h5py
    catalog = tmp_path / "camels_fixture.hdf5"
    with h5py.File(catalog, "w") as stream:
        header = stream.create_group("Header")
        header.attrs["HubbleParam"] = 0.6711
        header.attrs["Redshift"] = 0.0
        data = {"SubhaloMass": np.ones(6), "SubhaloStellarMass": np.ones(6) * 0.1,
                "SubhaloPos": np.arange(18).reshape(6, 3) * 10.,
                "SubhaloVel": np.ones((6, 3)), "SubhaloVelDisp": np.ones(6) * 100,
                "SubhaloHalfmassRad": np.ones(6) * 5, "SubhaloGrNr": np.repeat([0, 1], 3)}
        for name, values in data.items():
            stream.create_dataset("Subhalo/" + name, data=values)
        stream.create_dataset("Group/GroupMass", data=[100., 200.])
    for stage, namespace in (("C", c), ("D", d)):
        namespace.update(RUN_CAMELS=True, CAMELS_HDF5=catalog)
        execute(notebook_sources(stage)["camels"], namespace)
        execute(notebook_sources(stage)["provenance"], namespace)
        assert namespace["dependencies"]["CAMELS"]["sha256"] == workflow.file_info(catalog)["sha256"]
    for stage in "BCD":
        folder = tmp_path / stage / "smoke"
        manifest = json.loads((folder / "provenance.json").read_text())[stage]
        assert manifest["status"] == "complete"
        assert manifest["run_mode"] == "smoke"
        for name in manifest["artifacts"]:
            workflow.checked_artifact(folder, manifest, name)
        assert (folder.parent / f"notebook_{stage}_smoke_outputs.zip").is_file()
    # Run the gate again with a deterministic rejection, then execute the
    # actual test cell: no disallowed test-time adaptation may be attempted.
    monkeypatch.setattr(workflow, "tta_should_enable", lambda *args: False)
    sources = notebook_sources("D")
    execute(sources["validation-gate"], d)
    calls.clear()
    execute(sources["test-evaluation"], d)
    assert calls == []
    assert d["tta_df"].K.tolist() == [0]
    assert d["selected_k"] == 0
    # Metadata errors must stop loading instead of running an untrained policy.
    wrong = copy.deepcopy(c["context"])
    wrong["run_mode"] = "full"
    with pytest.raises(ValueError, match="run_mode"):
        workflow.source_artifacts(tmp_path / "B/smoke", tmp_path, "B", "policy.pt", wrong)
    wrong = copy.deepcopy(c["context"])
    wrong["split_cluster_ids"]["test"] = ["different"]
    with pytest.raises(ValueError, match="split_cluster_ids"):
        workflow.source_artifacts(tmp_path / "B/smoke", tmp_path, "B", "policy.pt", wrong)
    (tmp_path / "B/smoke/policy.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed artifact"):
        workflow.source_artifacts(tmp_path / "B/smoke", tmp_path, "B", "policy.pt", c["context"])


def test_empty_physics_statistics_are_explicit():
    graph = dict(edge_index=torch.tensor([[0, 1], [0, 1]]),
                 stellar_mass=torch.ones(2), pos=torch.zeros(2, 3))
    rows = workflow.physics_diagnostics([graph], [torch.ones(2)], [torch.ones(2, dtype=torch.bool)])
    assert rows[0]["spearman_rho"] is None
    assert rows[0]["binding_mannwhitney_p"] is None
    assert rows[0]["physical_edges"] == 0


def test_camels_requires_explicit_real_catalog(tmp_path):
    import h5py
    with pytest.raises(FileNotFoundError, match="CAMELS_HDF5"):
        workflow.local_camels_graphs(None, {}, None, None, tmp_path)
    fake = tmp_path / "synthetic.hdf5"
    with h5py.File(fake, "w") as stream:
        stream.create_group("Subhalo")
        stream.create_group("Group")
    with pytest.raises(ValueError, match="incomplete or synthetic"):
        workflow.local_camels_graphs(fake, {}, None, None, tmp_path)


def test_episode_rewards_do_not_broadcast(monkeypatch, single_thread):
    import importlib
    from rls.policy import EdgePolicyNet
    from rls.policy_gradient import ValueNet, PolicyGradientTrainer
    module = importlib.import_module("rls.train_policy")
    policy, value = EdgePolicyNet(edge_dim=1, node_emb_dim=2), ValueNet(2)
    cfg = dict(batch_size=2, sparsity_mode="penalty", target_sparsity_start=0.5,
               target_sparsity_end=0.5, sparsity_anneal_epochs=1, w_virial=0)
    graphs = [dict(x=torch.ones(2, 1), edge_attr=torch.ones(2, 1),
        edge_index=torch.tensor([[0, 1], [1, 0]]), ctx=torch.ones(2), emb=torch.ones(2, 2),
        y=torch.tensor([float(i)]), stellar_mass=torch.ones(2), vel_disp=torch.ones(2),
        pos=torch.tensor([[0., 0., 0.], [1., 0., 0.]])) for i in range(2)]
    monkeypatch.setattr(module, "compute_rewards", lambda pred, full, y, *args, **kwargs: y)
    original = module.compute_pg_loss
    seen = []

    def loss(logp, adv, values, rewards, **kwargs):
        assert logp.shape == adv.shape == values.shape == rewards.shape == (2,)
        torch.testing.assert_close(rewards, torch.tensor([0., 1.]))
        seen.append(True)
        return original(logp, adv, values, rewards, **kwargs)

    monkeypatch.setattr(module, "compute_pg_loss", loss)
    trainer = PolicyGradientTrainer(policy, value, torch.optim.Adam(policy.parameters()),
                                    torch.optim.Adam(value.parameters()), cfg)
    module.train_policy(trainer, graphs, lambda g, m: (g["y"], g["y"]), cfg, epochs=1)
    assert seen == [True]
