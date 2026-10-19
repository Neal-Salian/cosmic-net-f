#!/usr/bin/env python3
"""Verify the saved A–D evidence and reproduce the existing exploratory design.

No model execution, training, network access, or writes to experiment directories.
All inputs are lossless, SHA-256 checked snapshots committed under paper/evidence.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import warnings

import numpy as np
import scipy
from scipy import stats

PAPER = Path(__file__).resolve().parents[1]
INPUTS = PAPER / "evidence/inputs"
SEED, BOOTSTRAPS, PERMUTATIONS = 20261016, 50000, 99999
METHODS = ["random", "distance", "degree", "stellar_binding_proxy",
           "edge_feature_grad_x_input", "supervised_raw", "provided_rl"]
LABELS = {"full": "Full graph", "random": "Random", "distance": "Distance",
          "degree": "Degree", "stellar_binding_proxy": "Stellar binding proxy",
          "edge_feature_grad_x_input": "Gradient × input",
          "supervised_raw": "Trained selector", "provided_rl": "RL"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(name):
    return gzip.decompress((INPUTS / (name + ".gz")).read_bytes())


def read_json(name):
    return json.loads(read_bytes(name))


def read_csv(name):
    return list(csv.DictReader(io.StringIO(read_bytes(name).decode())))


def verify_inputs():
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    for record in manifest["files"]:
        data = (INPUTS / record["snapshot"]).read_bytes()
        require(sha(data) == record["snapshot_sha256"], "Snapshot hash mismatch")
        raw = gzip.decompress(data)
        require(sha(raw) == record["original_sha256"] and
                len(raw) == record["original_bytes"], "Source byte identity mismatch")
    return manifest


def align(left, right):
    def keyed(rows):
        result = {r["graph_id"]: r for r in rows}
        require(len(result) == len(rows), "Duplicate graph IDs")
        return result
    a, b = keyed(left), keyed(right)
    require(a.keys() == b.keys() and len(a) > 1, "Different or empty paired populations")
    ids = sorted(a)
    ya = np.array([a[k]["target"] for k in ids], float)
    yb = np.array([b[k]["target"] for k in ids], float)
    require(np.array_equal(ya, yb), "Different targets for the same halo")
    def errors(rows):
        return np.array([rows[k]["full_prediction"] if rows[k]["method"] == "full"
                         else rows[k]["pruned_prediction"] for k in ids], float) - ya
    er, ec = errors(a), errors(b)
    require(np.isfinite(er).all() and np.isfinite(ec).all(), "Non-finite residuals")
    return ids, er, ec


def holm(values):
    p = np.asarray(values, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid p values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def rmse_difference(a, b, axis=-1):
    return np.sqrt(np.mean(a*a, axis=axis)) - np.sqrt(np.mean(b*b, axis=axis))


def relative_improvement(a, b, axis=-1):
    return 100 * (1 - np.sqrt(np.mean(a*a, axis=axis) / np.mean(b*b, axis=axis)))


def mean_difference(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def paired_permutation(er, ec, seed, n_resamples=PERMUTATIONS):
    return float(stats.permutation_test((er*er, ec*ec), mean_difference,
        permutation_type="samples", vectorized=True, axis=-1,
        n_resamples=n_resamples, batch=2500, alternative="two-sided",
        rng=np.random.default_rng(seed)).pvalue)


def validate_mask(r):
    """selected_pairs is a list of indices, never a Boolean selection vector."""
    pairs = np.asarray(r["layout_pairs"], dtype=int)
    p = len(pairs)
    indices = np.asarray(r["selected_pairs"], dtype=int)
    require(len(set(indices.tolist())) == len(indices) and
            ((indices >= 0) & (indices < p)).all(), "Invalid selected-pair indices")
    selected = np.zeros(p, dtype=bool)
    selected[indices] = True
    inverse = np.asarray(r["layout_inverse"], dtype=int)
    real = np.asarray(r["stored_real"], dtype=bool)
    mask = np.asarray(r["stored_mask"], dtype=bool)
    require(len(mask) == len(real) == r["available_stored_columns"] and
            len(inverse) == int(real.sum()), "Invalid stored layout")
    require(((inverse >= 0) & (inverse < p)).all(), "Invalid inverse pair indices")
    require(np.array_equal(mask[real], selected[inverse]) and mask[~real].all(),
            "Reciprocal selection or loop preservation failure")
    count = len(indices)
    require(p == r["available_pair_count"] and count == r["final_pair_count"] ==
            r["requested_pair_count"] == math.ceil(r["requested_keep_fraction"] * p),
            "Incorrect physical-pair budget")
    require(set(pairs[selected].ravel()) == set(pairs.ravel()) and
            r["physical_isolates"] == 0 and r["constraint_satisfied"] and r["budget_satisfied"],
            "Constraint failure or isolated node")
    require((np.bincount(inverse, minlength=p) == 2).all(), "Pairs need reciprocal copies")
    require(int(real.sum()) == r["available_nonself_columns"] and
            int((~real).sum()) == r["available_loop_columns"] == r["loops_kept"] and
            int(mask.sum()) == r["total_kept_columns"] and
            int(mask[real].sum()) == r["retained_nonself_columns"], "Column count mismatch")
    require(np.isclose(r["physical_retention"], count/p, rtol=0, atol=1e-12) and
            np.isclose(r["stored_retention"], mask.mean(), rtol=0, atol=1e-12),
            "Physical and stored-column retention confused")


def validate_records(raw, expected_ids):
    groups, identities = defaultdict(list), defaultdict(set)
    nonfull = 0
    for r in raw:
        require(r["status"] == "success", "Failed evaluations must not be silently dropped")
        groups[(r["requested_keep_fraction"], r["method"])].append(r)
        identities[r["graph_id"]].add((r["target"], r["graph_content_hash"]))
        pred = r["full_prediction"] if r["method"] == "full" else r["pruned_prediction"]
        error = pred - r["target"]
        require(np.isfinite(error) and np.isclose(error*error, r["squared_error"], atol=1e-12)
                and np.isclose(abs(error), r["absolute_error"], atol=1e-12), "Saved error mismatch")
        if r["method"] != "full":
            validate_mask(r)
            nonfull += 1
        else:
            require(r["physical_retention"] == r["stored_retention"] == 1 and
                    r["final_pair_count"] == r["available_pair_count"], "Full reference was pruned")
    require(all(len(v) == 1 for v in identities.values()), "Graph or target changed across methods")
    for group in groups.values():
        ids, _, _ = align(group, group)
        require(set(ids) == set(expected_ids), "Evaluation is not the complete test split")
    return groups, nonfull


def metrics(rows):
    _, e, _ = align(rows, rows)
    y = np.array([r["target"] for r in sorted(rows, key=lambda r: r["graph_id"])])
    result = dict(rmse=float(np.sqrt(np.mean(e*e))), mae=float(np.mean(abs(e))),
                  r2=float(1 - np.sum(e*e)/np.sum((y-y.mean())**2)), bias=float(e.mean()),
                  std_residual=float(e.std()), n=len(rows))
    for field, key in [("physical_retention", "physical_retention"),
                       ("stored_retention", "stored_retention"),
                       ("total_region_time_s", "latency_s"),
                       ("scoring_time_s", "scoring_s"), ("decoder_time_s", "decoder_s"),
                       ("prediction_time_s", "prediction_s"), ("full_time_s", "full_s"),
                       ("live_forward_calls", "live_forward_calls")]:
        result[key] = float(np.mean([r[field] for r in rows]))
    result["latency_median_s"] = float(np.median([r["total_region_time_s"] for r in rows]))
    result["latency_iqr_s"] = np.quantile([r["total_region_time_s"] for r in rows], [.25, .75]).tolist()
    return result


def verify_aggregate(groups, csv_rows):
    mapping = {"rmse": "rmse", "mae": "mae", "r2": "r2", "bias": "bias",
               "std_residual": "std_residual", "mean_keep_frac": "physical_retention",
               "mean_stored_keep_frac": "stored_retention", "mean_inference_time_s": "latency_s",
               "mean_scoring_time_s": "scoring_s", "mean_decoder_time_s": "decoder_s",
               "mean_prediction_time_s": "prediction_s", "mean_full_time_s": "full_s"}
    require(len(groups) == len(csv_rows), "Unexpected method/budget aggregate count")
    result, largest = [], 0.
    for r in csv_rows:
        q, method = float(r["frac"]), r["method"]
        require(int(r["n_failed"]) == 0 and int(r["n_success"]) == 82, "Incomplete aggregate population")
        m = metrics(groups[(q, method)])
        for saved, computed in mapping.items():
            diff = abs(float(r[saved]) - m[computed])
            largest = max(largest, diff)
            require(diff < 3e-7, f"Aggregate mismatch: {method}, q={q}, {saved}: {diff}")
        m.update(q=q, method=method)
        result.append(m)
    return result, largest


def comparisons(groups):
    """Exactly the prior order and RNG streams, with additional descriptive effects."""
    contrasts = []
    controls = sorted(set(METHODS + ["full"]) - {"provided_rl"})
    for q in (.4, .6, .8):
        for method in controls:
            ids, er, ec = align(groups[(q, "provided_rl")], groups[(q, method)])
            idx = len(contrasts)
            kwargs = dict(paired=True, vectorized=True, axis=-1, n_resamples=BOOTSTRAPS,
                          batch=2500, confidence_level=.95, method="BCa")
            with warnings.catch_warnings(record=True) as caught:
                boot = stats.bootstrap((er, ec), rmse_difference,
                    rng=np.random.default_rng(SEED+idx*2), **kwargs)
                rel = stats.bootstrap((er, ec), relative_improvement,
                    rng=np.random.default_rng(SEED+idx*2), **kwargs)
            require(not caught and np.isfinite([*boot.confidence_interval, *rel.confidence_interval]).all(),
                    "Degenerate or unsupported bootstrap interval")
            loo = [float(rmse_difference(np.delete(er, i), np.delete(ec, i))) for i in range(len(ids))]
            wins, losses = int((abs(er) < abs(ec)).sum()), int((abs(er) > abs(ec)).sum())
            win_ci = stats.binomtest(wins, len(ids)).proportion_ci(method="wilson")
            contrasts.append(dict(q=q, control=method, n=len(ids), bootstrap_seed=SEED+idx*2,
                permutation_seed=SEED+idx*2+1, delta_rmse=float(rmse_difference(er, ec)),
                ci_low=float(boot.confidence_interval.low), ci_high=float(boot.confidence_interval.high),
                bootstrap_se=float(boot.standard_error),
                p=paired_permutation(er, ec, SEED+idx*2+1),
                relative_improvement_pct=float(relative_improvement(er, ec)),
                relative_ci_low=float(rel.confidence_interval.low),
                relative_ci_high=float(rel.confidence_interval.high),
                mse_difference=float(np.mean(er*er-ec*ec)),
                delta_mae=float(np.mean(abs(er))-np.mean(abs(ec))),
                wins=wins, losses=losses, ties=len(ids)-wins-losses,
                win_fraction=wins/len(ids), win_fraction_ci=[float(x) for x in win_ci],
                loo_min=min(loo), loo_max=max(loo)))
    for r, p in zip(contrasts, holm([r["p"] for r in contrasts])):
        r["holm21"] = float(p)
        r["holm7"] = None
    for r, p in zip(contrasts[:7], holm([r["p"] for r in contrasts[:7]])):
        r["holm7"] = float(p)
    return contrasts


def verify_previous(contrasts):
    previous = read_json("previous-analysis/paired-comparisons.json")
    fields = {"delta_rmse": "delta_rmse_rl_minus_control", "ci_low": "ci95_bca_low",
              "ci_high": "ci95_bca_high", "p": "paired_permutation_mse_p_two_sided",
              "holm7": "holm_p_q40_family7", "holm21": "holm_p_all_21_pruned_budget_contrasts",
              "bootstrap_se": "bootstrap_standard_error", "mse_difference": "mse_difference",
              "loo_min": "leave_one_graph_out_delta_min", "loo_max": "leave_one_graph_out_delta_max"}
    largest = 0.
    require(len(contrasts) == len(previous["contrasts"]) == 21, "Comparison family changed")
    for r, old in zip(contrasts, previous["contrasts"]):
        require((r["q"], r["control"]) == (old["requested_keep_fraction"], old["control"]),
                "Comparison order or RNG streams changed")
        for new, prior in fields.items():
            if old[prior] is None:
                require(r[new] is None, "Unexpected multiplicity family")
            else:
                difference = abs(r[new]-old[prior])
                largest = max(largest, difference)
                require(difference < 1e-12, f"Prior result not reproduced: {new}")
    return largest


def adaptation(provenance, groups):
    test, val = read_csv("D/tta_results.csv"), read_csv("D/tta_validation.csv")
    histories = read_json("D/tta_histories.json")
    base = {r["graph_id"]: r for r in groups[(.4, "full")]}
    out = []
    for r, v in zip(test, val):
        k = int(r["K"])
        require(k == int(v["K"]) and float(r["requested_keep_fraction"]) == .5,
                "Adaptation rows or budgets mismatched")
        hs = histories[str(k)]
        ids = [h["cluster_id"] for h in hs]
        require(len(ids) == len(set(ids)) == 82 and set(ids) == set(base), "D population mismatch")
        fractions, stored = [], []
        for h in hs:
            b = base[h["cluster_id"]]
            require(h["available_pair_count"] == b["available_pair_count"] and
                    h["final_pair_count"] == h["requested_pair_count"] == math.ceil(.5*b["available_pair_count"])
                    and h["budget_excess_pair_count"] == h["physical_isolates"] == 0
                    and h["constraint_satisfied"] and h["budget_satisfied"], "D budget/constraint mismatch")
            fractions.append(h["final_pair_count"]/h["available_pair_count"])
            stored.append((2*h["final_pair_count"]+b["loops_kept"])/b["available_stored_columns"])
        # D keep_frac was reduced in float32; do not relabel it as physical retention.
        require(abs(np.mean(stored)-float(r["keep_frac"])) < 1e-7, "D stored fractions mismatch")
        require(abs(np.mean([h["steps_run"] for h in hs])-float(r["mean_steps"])) < 1e-12 and
                abs(np.mean([h["inference_time_s"] for h in hs])-float(r["mean_time_s"])) < 1e-12,
                "D aggregate runtime/history mismatch")
        out.append(dict(K=k, selected=r["selected"] == "True", val_rmse=float(v["rmse"]),
                        rmse=float(r["rmse"]), mae=float(r["mae"]), r2=float(r["r2"]),
                        latency_s=float(r["mean_time_s"]), steps=float(r["mean_steps"]),
                        physical_retention=float(np.mean(fractions)), stored_retention=float(np.mean(stored))))
    require(len(out) == 4 and sum(r["selected"] for r in out) == 1 and
            min(out, key=lambda r: r["val_rmse"])["K"] == provenance["D"]["selected_K"] == 20,
            "Validation-selected K changed")
    return out


def run_analysis():
    manifest = verify_inputs()
    provenance = {s: read_json(f"{s}/provenance.json")[s] for s in "ABCD"}
    ref = provenance["C"]
    for s, p in provenance.items():
        require(p["status"] == "complete" and p["repo"]["head"] == manifest["base_commit"], "Wrong campaign")
        for field in ("dataset", "split_cluster_ids", "backbone", "source_identity"):
            require(p[field] == ref[field], f"Stage {s} identity mismatch: {field}")
    splits = ref["split_cluster_ids"]
    require([len(splits[k]) for k in ("train", "val", "test")] == [378, 81, 82], "Wrong split sizes")
    require(len(set(splits["train"]+splits["val"]+splits["test"])) == 541, "Split overlap")
    catalog = read_csv("catalog/tng100_clustered.csv")
    require(sha(read_bytes("catalog/tng100_clustered.csv")) == ref["dataset"]["tng_csv"]["sha256"],
            "Catalog does not match execution input")
    require(len(catalog) == 5378 and len({r["group_id"] for r in catalog}) == 541, "Catalog counts changed")
    a, an = validate_records(read_json("A/baseline_per_example.json"), splits["test"])
    c, cn = validate_records(read_json("C/matched_per_example.json"), splits["test"])
    am, a_diff = verify_aggregate(a, read_csv("A/baselines.csv"))
    cm, c_diff = verify_aggregate(c, read_csv("C/results_table.csv"))
    require(len(am) == 28 and len(cm) == 32 and an == 1968 and cn == 2296, "Unexpected record counts")
    # Verify A baseline accuracy agrees with the unified C evaluation to floating point precision.
    for m in am:
        other = next(x for x in cm if (x["q"], x["method"]) == (m["q"], m["method"]))
        require(abs(other["rmse"]-m["rmse"]) < 3e-7, "A/C baseline inconsistency")
    for left, right in zip(sorted(a[(.4, "full")], key=lambda r: r["graph_id"]),
                           sorted(c[(.4, "full")], key=lambda r: r["graph_id"])):
        require((left["graph_id"], left["target"], left["graph_content_hash"]) ==
                (right["graph_id"], right["target"], right["graph_content_hash"]), "A/C graph identity mismatch")
    b_rows = read_csv("B/policy_results.csv")
    require(len(b_rows) == 4, "Missing B results")
    for b, method in zip(b_rows[:2], ["full", "provided_rl"]):
        m = next(x for x in cm if x["q"] == .4 and x["method"] == method)
        for field in ("rmse", "mae", "r2"):
            require(abs(float(b[field])-m[field]) < 3e-7, "B frozen/C inconsistency")
    stageb = provenance["B"]["stageb"]
    stageb_logs = read_csv("B/stageb_training_log.csv")
    require(len(stageb_logs) == len(stageb["full_hist"]) == len(stageb["pruned_hist"]) == 6,
            "Stage B epoch count mismatch")
    for i, r in enumerate(stageb_logs):
        require(float(r["val_full_rmse"]) == stageb["full_hist"][i] and
                float(r["val_pruned_rmse"]) == stageb["pruned_hist"][i], "Stage B history mismatch")
    require(stageb["best_epoch"] == 2 and stageb["accepted"] and
            stageb["best_pruned"] == min(stageb["pruned_hist"]) and
            stageb["best_full"] <= 1.02*stageb["prefinetune_full"], "Stage B validation gate mismatch")
    cohorts = provenance["B"]["execution_cohorts"]
    require([cohorts[k]["eligible_count"] for k in ["train", "val", "test"]] == [377, 81, 82]
            and stageb["excluded_training_graph_ids"] == ["tng100_0000005"], "Stage B cohort mismatch")
    require(len(read_csv("B/training_log.csv")) == 60 and
            provenance["B"]["policy_validation"]["selected_epoch"] == 42, "Policy selection changed")
    contrasts = comparisons(c)
    previous_diff = verify_previous(contrasts)
    d = adaptation(provenance, c)
    pair_counts = [r["available_pair_count"] for r in c[(.4, "full")]]
    physical = read_csv("C/physics_alignment.csv")
    result = dict(schema=1, analysis="exploratory; conditional on this fitted model and legacy split",
        base_commit=manifest["base_commit"], seed=SEED, bootstrap_resamples=BOOTSTRAPS,
        permutation_resamples=PERMUTATIONS, numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), input_manifest_sha256=sha((INPUTS/"manifest.json").read_bytes()),
        analysis_script_sha256=sha(Path(__file__).read_bytes()),
        setup=dict(catalog_rows=len(catalog), halos=541, train=378, val=81, test=82,
                   pair_min=min(pair_counts), pair_max=max(pair_counts), pair_median=float(np.median(pair_counts)),
                   device=ref["device"], torch=ref["torch"], graph_backend=ref["graph_backend"],
                   backbone_state_sha256=ref["backbone"]["sha256"],
                   policy_selected_epoch=provenance["B"]["policy_validation"]["selected_epoch"],
                   graph_construction_s=ref["graph_construction_time_s"]),
        verification=dict(input_files=len(manifest["files"]), A_records=2296, C_records=2624,
                          A_masks=an, C_masks=cn, A_max_aggregate_difference=a_diff,
                          C_max_aggregate_difference=c_diff, prior_statistics_max_difference=previous_diff),
        metrics=cm, A_metrics=am, B_results=b_rows, stageb=provenance["B"]["stageb"],
        contrasts=contrasts, adaptation=d,
        diagnostics=dict(physics_mean_rho=float(np.mean([float(r["spearman_rho"]) for r in physical])),
                         physics_median_rho=float(np.median([float(r["spearman_rho"]) for r in physical])),
                         physics_positive=sum(float(r["spearman_rho"]) > 0 for r in physical),
                         calibration=read_csv("C/calibration.csv")))
    dest = PAPER / "evidence/analysis.json"
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (PAPER / "evidence/paired-comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    with (PAPER / "evidence/per-halo-error-differences.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q", "control", "graph_id", "rl_residual_dex", "control_residual_dex",
                         "delta_absolute_error_dex", "delta_squared_error_dex2"])
        for x in contrasts:
            ids, er, ec = align(c[(x["q"], "provided_rl")], c[(x["q"], x["control"])])
            writer.writerows((x["q"], x["control"], i, float(a), float(b), float(abs(a)-abs(b)),
                             float(a*a-b*b)) for i, a, b in zip(ids, er, ec))
    print(f"Verified {len(manifest['files'])} evidence files; A/C metrics, {an+cn} masks; 21 contrasts reproduced "
          f"(max difference {previous_diff:.3g}).")
    return result


if __name__ == "__main__":
    run_analysis()
