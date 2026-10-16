"""Explicit amortization scenarios, using seconds from matched measurements."""
import math

import numpy as np


def break_even_reuses(setup_s, full_s, pruned_s):
    """Smallest integer R with setup + R*pruned < R*full, or None."""
    if any(not math.isfinite(v) or v < 0 for v in (setup_s, full_s, pruned_s)):
        raise ValueError('costs must be finite and nonnegative')
    saving = full_s - pruned_s
    if saving <= 0:
        return None
    reuses = math.floor(setup_s / saving) + 1
    # Floating subtraction can move an equality across the integer boundary.
    # Verify the actual inequality represented by the returned model count.
    while not setup_s + reuses * pruned_s < reuses * full_s:
        reuses += 1
    while reuses > 1 and setup_s + (reuses - 1) * pruned_s < (reuses - 1) * full_s:
        reuses -= 1
    return reuses


def build_cost_model(records):
    def mean(name):
        return float(np.mean([np.mean(row['samples_s'][name]) for row in records]))
    full, pruned = mean('full_graph'), mean('pruned_backbone_service')
    setup = sum(mean(name) for name in ('topology_preparation', 'embedding_service',
                                      'policy_and_pair_reduction', 'optimized_decoder'))
    original_setup = sum(mean(name) for name in ('pair_layout', 'embedding_service',
                                               'policy_and_pair_reduction', 'reference_decoder'))
    per_graph = []
    for row in records:
        average = lambda name: float(np.mean(row['samples_s'][name]))
        graph_setup = sum(average(name) for name in ('topology_preparation', 'embedding_service',
                                                  'policy_and_pair_reduction', 'optimized_decoder'))
        per_graph.append(dict(graph_id=row['graph_id'], setup_s=graph_setup,
            full_s=average('full_graph'), pruned_s=average('pruned_backbone_service'),
            break_even_reuses=break_even_reuses(graph_setup, average('full_graph'), average('pruned_backbone_service'))))
    return {
        'scope': 'CPU service costs; resident input graph; frozen backbone; fixed reusable mask',
        'full_s': full, 'pruned_s': pruned, 'setup_s': setup,
        'reference_setup_s': original_setup,
        'optimized_break_even_reuses': break_even_reuses(setup, full, pruned),
        'reference_break_even_reuses': break_even_reuses(original_setup, full, pruned),
        'one_shot_measured_s': mean('optimized_rl_one_shot'),
        'one_shot_additive_model_s': setup + pruned,
        'per_graph': per_graph,
        'reuse_scenarios': [{'reuses': r, 'full_total_s': r * full,
                             'optimized_total_s': setup + r * pruned,
                             'reference_total_s': original_setup + r * pruned}
                            for r in (1, 2, 4, 8, 16, 32, 64, 128)],
        'per_halo_topology_cache_s': mean('topology_preparation'),
        'feature_generation_s': mean('feature_generation'),
    }
