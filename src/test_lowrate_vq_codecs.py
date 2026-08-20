#!/usr/bin/env python3
"""Mechanism tests for low-rate VQ baseline codecs."""

from __future__ import annotations

import torch

from build_fibquant_codebook import orthogonal_matrix
from fw_gsrq.codec import build_fixed_whitener, build_sqrt_fisher_whitener
from lowrate_vq.codecs import (
    fibquant_rate,
    opq_rate,
    quantize_fibquant,
    quantize_fibquant_reference,
    quantize_opq,
)
from lowrate_vq.opq import fit_opq
from lowrate_vq.commvq import (
    CommVQValueCodec,
    commvq_rate,
    fit_commutative_stage,
    quantize_commvq_key,
    quantize_commvq_value,
    theta_to_centers,
)
from lowrate_vq.m20_fibquant import (
    build_block_metric_geometry,
    metric_codebook_diagnostics,
    normalize_fisher,
    post_rope_fisher_samples,
    prepare_m20_fibquant_search,
    quantize_m20_fibquant_key,
    quantize_m20_fibquant_key_reference,
    refine_shared_metric_codebook,
)


def test_rate_ledgers() -> None:
    native = fibquant_rate(dimension=128, block_size=8, codebook_size=64)
    strict = fibquant_rate(dimension=128, block_size=8, codebook_size=32)
    opq = opq_rate(dimension=128, subspaces=16, centroids=64)
    fibquant_paper_lm_min = fibquant_rate(
        dimension=128, block_size=4, codebook_size=256
    )
    opq_paper_min = opq_rate(dimension=128, subspaces=2, centroids=256)
    assert native.index_bpa == 0.75 and native.total_bpa == 0.875
    assert strict.index_bpa == 0.625 and strict.total_bpa == 0.75
    assert opq.index_bpa == 0.75 and opq.total_bpa == 0.75
    assert fibquant_paper_lm_min.index_bpa == 2.0
    assert fibquant_paper_lm_min.metadata_bpa == 0.125
    assert fibquant_paper_lm_min.total_bpa == 2.125
    assert opq_paper_min.index_bpa == 0.125
    assert opq_paper_min.total_bpa == 0.125


def test_fibquant_determinism_and_fusion() -> None:
    torch.manual_seed(3)
    rotation = orthogonal_matrix(8, 29)
    codebook = torch.randn(16, 2)
    codebook /= codebook.norm(dim=-1, keepdim=True)
    states = torch.randn(1, 2, 5, 8)
    decoded1, indices1 = quantize_fibquant(
        states, rotation, codebook, block_size=2, return_indices=True
    )
    decoded2, indices2 = quantize_fibquant(
        states, rotation, codebook, block_size=2, return_indices=True
    )
    assert torch.equal(indices1, indices2)
    assert torch.equal(decoded1, decoded2)
    query = torch.randn_like(states)
    explicit = torch.einsum("bhtd,bhsd->bhts", query, decoded1)
    rotated_query = query.float() @ rotation.float().transpose(0, 1)
    rotated_decoded = decoded1.float() @ rotation.float().transpose(0, 1)
    fused = torch.einsum("bhtd,bhsd->bhts", rotated_query, rotated_decoded)
    assert torch.allclose(explicit.float(), fused, atol=2e-5, rtol=2e-5)


def test_fibquant_batched_matches_flattened_reference() -> None:
    torch.manual_seed(5)
    rotation = orthogonal_matrix(8, 31)
    codebook = torch.randn(16, 2)
    states = torch.randn(1, 2, 7, 8)
    expected, expected_indices = quantize_fibquant_reference(
        states,
        rotation,
        codebook,
        block_size=2,
        chunk_size=3,
        return_indices=True,
    )
    actual, actual_indices = quantize_fibquant(
        states,
        rotation,
        codebook,
        block_size=2,
        chunk_size=3,
        return_indices=True,
        maximum_distance_elements=64,
    )
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual, expected)


def test_m20_fibquant_identity_matches_parent() -> None:
    torch.manual_seed(31)
    rotation = orthogonal_matrix(8, 29)
    codebook = torch.randn(16, 2)
    states = torch.randn(1, 2, 7, 8)
    parent, parent_indices = quantize_fibquant(
        states,
        rotation,
        codebook,
        block_size=2,
        return_indices=True,
    )
    identity = torch.eye(2).expand(2, 4, 2, 2).clone()
    child, child_indices = quantize_m20_fibquant_key(
        states,
        rotation,
        codebook,
        identity,
        block_size=2,
        return_indices=True,
    )
    assert torch.equal(parent_indices, child_indices)
    assert torch.equal(parent, child)


def test_m20_fibquant_batched_matches_serial_reference() -> None:
    torch.manual_seed(33)
    rotation = orthogonal_matrix(8, 43)
    codebook = torch.randn(16, 2)
    whiteners = torch.randn(2, 4, 2, 2)
    whiteners = whiteners @ whiteners.transpose(-1, -2)
    states = torch.randn(1, 2, 7, 8)
    expected, expected_indices = quantize_m20_fibquant_key_reference(
        states,
        rotation,
        codebook,
        whiteners,
        block_size=2,
        chunk_size=3,
        return_indices=True,
    )
    selector_cross_tables, selector_norms = prepare_m20_fibquant_search(
        codebook, whiteners
    )
    actual, actual_indices = quantize_m20_fibquant_key(
        states,
        rotation,
        codebook,
        whiteners,
        block_size=2,
        chunk_size=3,
        return_indices=True,
        selector_cross_codebooks=selector_cross_tables,
        selector_norms=selector_norms,
        maximum_distance_elements=64,
    )
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual, expected)

    zero = torch.zeros_like(states[:, :, :1])
    zero_expected, zero_expected_indices = quantize_m20_fibquant_key_reference(
        zero,
        rotation,
        codebook,
        whiteners,
        block_size=2,
        return_indices=True,
    )
    zero_actual, zero_actual_indices = quantize_m20_fibquant_key(
        zero,
        rotation,
        codebook,
        whiteners,
        block_size=2,
        return_indices=True,
        selector_cross_codebooks=selector_cross_tables,
        selector_norms=selector_norms,
    )
    assert torch.equal(zero_actual_indices, zero_expected_indices)
    assert torch.equal(zero_actual, zero_expected)


def test_m20_fibquant_geometry_sampling_and_refinement() -> None:
    torch.manual_seed(37)
    rotation = orthogonal_matrix(8, 41)
    raw = torch.randn(2, 8, 8)
    fisher = raw @ raw.transpose(-1, -2)
    geometry = build_block_metric_geometry(fisher, rotation, block_size=2)
    assert geometry.diagnostics["finite"]
    assert geometry.diagnostics["block_minimum_eigenvalue"] >= 0.5 - 1e-5
    assert tuple(geometry.block_whiteners.shape) == (2, 4, 2, 2)
    identity_rotation = torch.eye(8)
    default_geometry = build_block_metric_geometry(
        fisher, identity_rotation, block_size=8
    )
    full_geometry = build_block_metric_geometry(
        fisher,
        identity_rotation,
        block_size=8,
        spectral_power=1.0,
        fisher_mixture=0.5,
    )
    assert torch.allclose(
        default_geometry.raw_metrics[0],
        build_sqrt_fisher_whitener(fisher[0]).metric,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        full_geometry.raw_metrics[0],
        build_fixed_whitener(fisher[0]).metric,
        atol=1e-6,
        rtol=0.0,
    )
    assert full_geometry.diagnostics["m20_spectral_power"] == 1.0

    queries = torch.randn(1, 4, 8, 8)
    logits = torch.randn(1, 4, 8, 8).tril()
    causal = torch.ones(8, 8, dtype=torch.bool).tril()
    logits[..., ~causal] = torch.finfo(logits.dtype).min
    probabilities = torch.softmax(logits, dim=-1)
    samples = post_rope_fisher_samples(
        queries, probabilities, kv_heads=2, samples_per_head=16
    )
    outer = torch.einsum("hsd,hse->hde", samples.double(), samples.double()) / 16
    normalized = normalize_fisher(outer)
    traces = normalized.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(traces, torch.full_like(traces, 8.0), atol=1e-4)

    blocks = torch.randn(2, 4, 256, 2) * 0.2
    blocks[:, :, :128] += torch.tensor([0.7, 0.1])
    blocks[:, :, 128:] += torch.tensor([-0.6, -0.2])
    initial = torch.randn(8, 2)
    before = metric_codebook_diagnostics(
        blocks, initial, geometry.block_whiteners, chunk_size=64
    )
    deployed, history = refine_shared_metric_codebook(
        blocks,
        initial,
        geometry.block_whiteners,
        iterations=4,
        chunk_size=64,
    )
    after = metric_codebook_diagnostics(
        blocks, deployed, geometry.block_whiteners, chunk_size=64
    )
    assert len(history) == 4
    assert after["finite"]
    assert after["metric_vector_mse"] < before["metric_vector_mse"]

    # With one codeword, norm-squared weighting must reduce exactly to the
    # weighted centroid update. The per-sample scalar does not enter assignment.
    scalar_blocks = torch.tensor([[[[0.0], [2.0]]]])
    scalar_initial = torch.tensor([[0.0]])
    scalar_whitener = torch.ones(1, 1, 1, 1)
    scalar_weights = torch.tensor([[1.0, 3.0]])
    weighted, weighted_history = refine_shared_metric_codebook(
        scalar_blocks,
        scalar_initial,
        scalar_whitener,
        sample_weights=scalar_weights,
        iterations=1,
        ridge=1e-12,
    )
    assert torch.equal(weighted, torch.tensor([[1.5]]))
    assert weighted_history[0]["sample_weighting"] == "norm_squared_mean_one"


def test_opq_identity_roundtrip_and_fusion() -> None:
    torch.manual_seed(5)
    states = torch.randn(1, 2, 4, 8)
    rotations = torch.stack([orthogonal_matrix(8, 40 + h) for h in range(2)])
    codebooks = torch.randn(2, 4, 8, 2)
    decoded, indices = quantize_opq(
        states, rotations, codebooks, return_indices=True
    )
    assert indices is not None and tuple(indices.shape) == (1, 2, 4, 4)
    query = torch.randn_like(states)
    explicit = torch.einsum("bhtd,bhsd->bhts", query, decoded)
    rotated_query = torch.einsum("hde,bhte->bhtd", rotations, query)
    rotated_decoded = torch.einsum("hde,bhte->bhtd", rotations, decoded)
    fused = torch.einsum("bhtd,bhsd->bhts", rotated_query, rotated_decoded)
    assert torch.allclose(explicit, fused, atol=2e-5, rtol=2e-5)


def test_opq_fit_is_finite_and_orthogonal() -> None:
    torch.manual_seed(7)
    vectors = torch.randn(2, 256, 8)
    fit = fit_opq(
        vectors,
        subspaces=4,
        clusters=8,
        iterations=4,
        chunk_size=64,
        seed=29,
    )
    assert fit.diagnostics["finite"]
    assert fit.diagnostics["rotation_orthogonality_max_abs"] < 2e-5
    assert fit.diagnostics["mean_code_utilization"] > 0.9
    assert len(fit.history) == 4


def test_commvq_geometry_rate_and_determinism() -> None:
    rate = commvq_rate(
        full_dimension=4096, group_pairs=64, levels=64, stages=8
    )
    assert rate.index_bpa == 0.75
    assert rate.metadata_bpa == 16 / 4096
    paper_rate = commvq_rate(
        full_dimension=4096, group_pairs=64, levels=64, stages=11
    )
    assert paper_rate.index_bpa == 1.03125
    theta = torch.randn(2, 2, 8, 2)
    centers = theta_to_centers(theta[0, 0])
    assert tuple(centers.shape) == (16, 4)
    states = torch.randn(1, 1, 5, 8)
    decoded1, indices1 = quantize_commvq_key(
        states, theta, return_indices=True
    )
    decoded2, indices2 = quantize_commvq_key(
        states, theta, return_indices=True
    )
    assert torch.equal(indices1, indices2)
    assert torch.equal(decoded1, decoded2)
    codec = CommVQValueCodec(0.75, 8).eval()
    value1 = quantize_commvq_value(states, codec)
    value2 = quantize_commvq_value(states, codec)
    assert torch.equal(value1, value2)
    assert torch.isfinite(value1).all()


def test_commvq_m_step_survives_degenerate_soft_clusters() -> None:
    # Repeated samples create the same rank/conditioning stress that occurs
    # when an annealed 4096-center layer-0 fit has nearly empty soft clusters.
    samples = torch.zeros(32, 8)
    samples[::2, 0] = 1.0
    theta, centers, history, solvers = fit_commutative_stage(
        samples,
        levels=8,
        iterations=2,
        initial_temperature=1e-3,
        final_temperature=1e-5,
        seed=29,
    )
    assert torch.isfinite(theta).all() and torch.isfinite(centers).all()
    assert all(torch.isfinite(torch.tensor(history)))
    assert len(solvers) == 3
    assert all(row["finite"] for row in solvers)


if __name__ == "__main__":
    test_rate_ledgers()
    test_fibquant_determinism_and_fusion()
    test_m20_fibquant_identity_matches_parent()
    test_m20_fibquant_geometry_sampling_and_refinement()
    test_opq_identity_roundtrip_and_fusion()
    test_opq_fit_is_finite_and_orthogonal()
    test_commvq_geometry_rate_and_determinism()
    test_commvq_m_step_survives_degenerate_soft_clusters()
    print("low-rate VQ codec tests passed")
