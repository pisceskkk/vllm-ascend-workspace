"""Family-resolution tests for attention.

These tests pin two contracts at once:

1. **The category-driven resolver** (``common.resolve_attention_family``)
   maps kernel bags to the right paper-aligned family. This is the
   executable form of the "must_have / must_not_have" signatures in
   ``knowledge/attention_families.yaml``.
2. **The HTML report uses the same resolver.**
   ``html_report.detect_attention_subtype`` is tested directly with
   fake ``Event``-shaped objects, so the test contract and the report
   output cannot drift apart.

Family names follow the DeepSeek papers (``mla`` / ``dsa`` / ``csa`` /
``hca`` / ``gqa`` / …), NOT the CANN backend class name. DSA (V3.2)
and CSA (V4) both route through AscendSFABackend on Ascend, but they
are different paper architectures distinguished by whether a
Compressor kernel is present.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest

_SKILL_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _SKILL_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ascend_profile import common, html_report  # noqa: E402


def _categories_from_kernels(names: Iterable[str]) -> set[str]:
    cats: set[str] = set()
    for n in names:
        c, _ = common.categories_and_roles(n, "", "")
        cats.update(c)
    return cats


def _resolve_attention_family(names: Iterable[str]) -> str:
    """Convenience wrapper that runs the kernel names through the same
    pipeline the HTML report uses: ``categories_and_roles`` to get the
    category set, then ``common.resolve_attention_family``."""
    return common.resolve_attention_family(_categories_from_kernels(names))


# --- Minimal Event stand-in for the html_report.detect_attention_subtype test ---


@dataclass
class _FakeEvent:
    name: str
    rank_id: str = "rank0"
    row_idx: int = 0
    task_type: str = ""
    accel_core: str = ""


@dataclass
class _FakeBlock:
    """Smallest object that ``detect_attention_subtype`` will accept.

    The function only reads ``b.events`` and slices it via
    ``events_in_row_range(b.events, row_start, row_end, rank_id)``.
    ``events_in_row_range`` filters by ``rank_id`` and ``row_idx``, so
    we set both consistently across the fake events.
    """

    events: list


def _events_for(names: Iterable[str]) -> list[_FakeEvent]:
    return [_FakeEvent(name=n, rank_id="rank0", row_idx=i) for i, n in enumerate(names)]


# Real-trace kernel bags. Each list is a *subset* of the kernels that
# appear in one attention block for the named family; the full block is
# bigger (RoPE, norm, BMM, etc.) but the listed ones are the unique
# signature kernels.

_FIXTURES: list[tuple[str, list[str], str]] = [
    # ---------- DeepSeek V2 / V3 MLA decode ----------
    (
        "DSV2_V3_MLA_decode",
        [
            "MlaPreprocess",
            "KvRmsNormRopeCache",
            "FusedInferAttentionScoreV2",
            "TransposeQuantBatchMatmul",
            "InterleaveRope",
        ],
        "mla",
    ),
    (
        "DSV2_V3_MLA_prefill",
        [
            "KvRmsNormRopeCache",
            "FusedInferAttentionScore",
            "InterleaveRope",
        ],
        "mla",
    ),
    (
        "DSV2_V3_MLA_with_canonical_CANN_name",
        # CANN op_list canonical names. We accept all three spellings.
        [
            "MlaProlog",
            "KvRmsNormRopeCache",
            "FusedInferAttentionScore",
        ],
        "mla",
    ),
    # ---------- DeepSeek V3.2 = DSA (NOT csa; NOT mla) ----------
    # DSA = Lightning Indexer + Sparse-SharedKV, NO Compressor.
    # DSA is built on MLA (V3.2 paper §4), so MLAPO and KvRmsNormRopeCache
    # still appear, but the sparse signatures must win.
    (
        "DSV3.2_DSA_decode",
        [
            "MlaPreprocess",
            "KvRmsNormRopeCache",
            "InterleaveRope",
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "BatchMatmulTranspose",
        ],
        "dsa",
    ),
    (
        "DSV3.2_DSA_prefill",
        [
            "KvRmsNormRopeCache",
            "QuantLightningIndexer",
            "KVQuantSparseAttnSharedKV",
            "IndexerCompressEpilogV2",
            "InPlacePartialRotaryMul",
        ],
        "dsa",
    ),
    # ---------- DeepSeek V4 = CSA (main layers) ----------
    # CSA = KV Compressor + Lightning Indexer + Sparse-SharedKV. The
    # presence of the Compressor kernel is what distinguishes V4 CSA
    # from V3.2 DSA.
    (
        "DSV4_CSA_prefill",
        [
            "KVQuantSparseAttnSharedKV",
            "KVQuantSparseAttnSharedKVMetadata",
            "QuantLightningIndexer",
            "QuantLightningIndexerMetadata",
            "Compressor",
            "KVCompressEpilog",
            "IndexerCompressEpilogV2",
            "InPlacePartialRotaryMul",
        ],
        "csa",
    ),
    (
        "DSV4_CSA_decode_with_MLAPO_reuse",
        [
            "MlaPreprocess",
            "KvRmsNormRopeCache",
            "InterleaveRope",
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "Compressor",
            "BatchMatmulTranspose",
        ],
        "csa",
    ),
    # ---------- DeepSeek V4 = HCA (alternating layers, heuristic) ----------
    # HCA = Compressor + dense FIA, no indexer, no sparse-sharedkv.
    (
        "DSV4_HCA_heuristic",
        [
            "Compressor",
            "KVCompressEpilog",
            "FusedInferAttentionScore",
            "InterleaveRope",
        ],
        "hca",
    ),
    # ---------- Dense GQA (Llama / Qwen / Mistral) ----------
    (
        "Qwen3_dense_decode",
        [
            "FusedInferAttentionScore",
            "NpuRotaryEmbedding",
        ],
        "gqa",
    ),
    (
        "Llama_dense_prefill",
        [
            "UnpadFlashAttention",
            "NpuRotaryEmbedding",
        ],
        "gqa",
    ),
    # ---------- Linear / Mamba / GDN ----------
    (
        "Mamba2_attn_layer",
        ["CausalConv1d"],
        "linear",
    ),
    # ---------- KVComp overlays ----------
    (
        "DSV3.2_DSA_with_kvcomp",
        [
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "NpuHammingDistTopK",
            "NpuSignBitsPack",
        ],
        "dsa+kvc",
    ),
    (
        "DSV4_CSA_with_kvcomp",
        [
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "Compressor",
            "NpuHammingDistTopK",
        ],
        "csa+kvc",
    ),
    (
        "DSV2_MLA_with_kvcomp",
        [
            "KvRmsNormRopeCache",
            "FusedInferAttentionScoreV2",
            "NpuHammingDistTopK",
        ],
        "mla+kvc",
    ),
    (
        "Dense_with_kvcomp",
        [
            "FusedInferAttentionScore",
            "NpuHammingDistTopK",
        ],
        "gqa+kvc",
    ),
]


@pytest.mark.parametrize(
    "label,kernels,expected_family",
    _FIXTURES,
    ids=[c[0] for c in _FIXTURES],
)
def test_attention_family_resolution(label, kernels, expected_family):
    """Drives the category-driven resolver (used by the HTML report)."""
    got = _resolve_attention_family(kernels)
    assert got == expected_family, (
        f"fixture {label}: kernels {kernels} resolved to family {got!r}, "
        f"expected {expected_family!r}"
    )


@pytest.mark.parametrize(
    "label,kernels,expected_family",
    _FIXTURES,
    ids=[c[0] for c in _FIXTURES],
)
def test_detect_attention_subtype_matches_resolver(label, kernels, expected_family):
    """The HTML report function must agree with the resolver on every
    fixture. Previously ``detect_attention_subtype`` ran its own raw
    kernel-name substring matcher, which diverged from the resolver on
    e.g. ``UnpadFlashAttention`` (resolver: ``gqa``, old code: ``fa``)
    and on metadata-only sparse blocks. This test pins the contract."""
    block = _FakeBlock(events=_events_for(kernels))
    got = html_report.detect_attention_subtype(
        block,
        row_start=0,
        row_end=len(kernels),
        rank_id="rank0",
    )
    assert got == expected_family, (
        f"fixture {label}: detect_attention_subtype returned {got!r}, "
        f"expected {expected_family!r}"
    )


# ---------------------------------------------------------------------------
# Edge-case regressions surfaced by the PR #50 review (gpt-5.5)
# ---------------------------------------------------------------------------


def test_unpad_flash_attention_resolves_to_gqa_not_fa():
    """``UnpadFlashAttention`` is the long-context branch of vllm-ascend's
    dense ``AscendAttentionBackend`` — NOT a separate FA backend. It
    must report as ``gqa`` so the YAML / category contract holds.
    Regression: PR #50 review point 2.
    """
    bag = ["UnpadFlashAttention", "NpuRotaryEmbedding"]

    assert _resolve_attention_family(bag) == "gqa"
    block = _FakeBlock(events=_events_for(bag))
    assert html_report.detect_attention_subtype(block, 0, len(bag), "rank0") == "gqa"


def test_metadata_only_sparse_block_does_not_satisfy_sparse_signature():
    """A block that only contains the *metadata* sub-kernel must NOT
    classify as ``dsa`` / ``csa``. The main sparse-shared-KV category
    (``attention.sparse_sharedkv``) is required; the metadata category
    (``attention.sparse_sharedkv.metadata``) must not satisfy it.
    Regression: PR #50 review point 3.
    """
    bag = ["KVQuantSparseAttnSharedKVMetadata", "QuantLightningIndexer"]
    cats = _categories_from_kernels(bag)

    assert "attention.sparse_sharedkv" not in cats
    assert "attention.sparse_sharedkv.metadata" in cats
    assert common.resolve_attention_family(cats) != "dsa"
    assert common.resolve_attention_family(cats) != "csa"

    block = _FakeBlock(events=_events_for(bag))
    got = html_report.detect_attention_subtype(block, 0, len(bag), "rank0")
    assert got not in ("dsa", "csa"), (
        f"metadata-only sparse block classified as {got!r}; the main "
        "attention.sparse_sharedkv category must be required."
    )


def test_compressor_plus_dense_fia_alone_resolves_to_hca():
    """V4 HCA-heuristic: Compressor + dense FIA, no indexer, no
    sparse-shared-KV. Verifies the resolver agrees with the cheat-sheet
    step 2.
    """
    bag = ["Compressor", "KVCompressEpilog", "FusedInferAttentionScore"]
    assert _resolve_attention_family(bag) == "hca"
    block = _FakeBlock(events=_events_for(bag))
    assert html_report.detect_attention_subtype(block, 0, len(bag), "rank0") == "hca"


def test_compressor_indexer_sparse_resolves_to_csa():
    """V4 CSA main layer: all three sparse-attention building blocks
    plus a Compressor. Verifies the resolver agrees with the cheat-sheet
    step 1.
    """
    bag = ["Compressor", "QuantLightningIndexer", "KVQuantSparseAttnSharedKV"]
    assert _resolve_attention_family(bag) == "csa"
    block = _FakeBlock(events=_events_for(bag))
    assert html_report.detect_attention_subtype(block, 0, len(bag), "rank0") == "csa"


def test_indexer_plus_sparse_no_compressor_resolves_to_dsa():
    """V3.2 DSA: Lightning Indexer + Sparse-SharedKV, no Compressor.
    Verifies the resolver agrees with the cheat-sheet step 3.
    """
    bag = ["QuantLightningIndexer", "KVQuantSparseAttnSharedKV"]
    assert _resolve_attention_family(bag) == "dsa"
    block = _FakeBlock(events=_events_for(bag))
    assert html_report.detect_attention_subtype(block, 0, len(bag), "rank0") == "dsa"


def test_csa_vs_dsa_distinguished_by_compressor():
    """The Compressor kernel is the *only* difference between a V3.2 DSA
    layer and a V4 CSA layer at the kernel level. Drop the Compressor
    from a CSA bag → it must reclassify as DSA. Add a Compressor back →
    must reclassify as CSA.
    """
    csa_bag = ["KVQuantSparseAttnSharedKV", "QuantLightningIndexer", "Compressor"]
    dsa_bag = ["KVQuantSparseAttnSharedKV", "QuantLightningIndexer"]

    assert _resolve_attention_family(csa_bag) == "csa"
    assert _resolve_attention_family(dsa_bag) == "dsa"


def test_mla_signature_disjoint_from_sparse():
    """A pure MLA bag (no Compressor, no Indexer, no Sparse-SharedKV)
    must resolve to ``mla``. A pure sparse bag must NOT pick up the
    ``mla`` family label even when it shares the MLA preprocess kernel.
    """
    mla_only = ["MlaPreprocess", "KvRmsNormRopeCache", "TransposeQuantBatchMatmul"]
    dsa_with_mla_reuse = [
        "MlaPreprocess",
        "KvRmsNormRopeCache",
        "KVQuantSparseAttnSharedKV",
        "QuantLightningIndexer",
    ]
    csa_with_mla_reuse = dsa_with_mla_reuse + ["Compressor"]

    assert _resolve_attention_family(mla_only) == "mla"
    assert _resolve_attention_family(dsa_with_mla_reuse) == "dsa"
    assert _resolve_attention_family(csa_with_mla_reuse) == "csa"


def test_block_head_hc_prefix_does_not_pollute_attention_family():
    """The HC* block-head prefix kernels appear before BOTH attention
    and MoE blocks. Adding them to a DSA bag must not flip the family,
    must not introduce moe.gating, must not pretend to be SFA-specific.
    """
    dsa_with_hc = [
        "HCPreSinkhorn",
        "HCPreInvRMS",
        "HCPost",
        "KVQuantSparseAttnSharedKV",
        "QuantLightningIndexer",
    ]
    assert _resolve_attention_family(dsa_with_hc) == "dsa"
    cats = _categories_from_kernels(dsa_with_hc)
    assert "block_head.mhc_prefix" in cats
    assert "moe.gating" not in cats
